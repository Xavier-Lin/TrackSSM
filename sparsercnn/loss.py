#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MDR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit
from typing import List
from .util import box_ops
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, matched_boxlist_iou
from detectron2.structures import Boxes, Instances
from scipy.optimize import linear_sum_assignment

#--------------- detection loss --------------------------
class SetCriterion(nn.Module):
    """ This class computes the loss for MDR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, cfg, num_classes, matcher, weight_dict, eos_coef, losses, use_focal):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_focal = use_focal
        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.MDR.ALPHA
            self.focal_loss_gamma = cfg.MODEL.MDR.GAMMA
        else:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        if self.use_focal:
            src_logits = src_logits.flatten(0, 1)
            # prepare one_hot target.
            target_classes = target_classes.flatten(0, 1)
            pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
            labels = torch.zeros_like(src_logits)
            labels[pos_inds, target_classes[pos_inds]] = 1
            # comp focal loss.
            class_loss = sigmoid_focal_loss_jit(
                src_logits,
                labels,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            ) / num_boxes
            losses = {'loss_ce': class_loss}
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
            losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses


    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes_xyxy'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes, target_boxes))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        image_size = torch.cat([v["image_size_xyxy_tgt"] for v in targets])
        src_boxes_ = src_boxes / image_size
        target_boxes_ = target_boxes / image_size

        loss_bbox = F.l1_loss(src_boxes_, target_boxes_, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs:
                {
                    'pred_logits': outputs_class[-1], 
                    'pred_boxes': outputs_coord[-1],
                    'aux_outputs' : [{'pred_logits': a, 'pred_boxes': b},{},...]
                }
             targets: 
                [
                    {
                    "labels" :一张图片上所有的标记类别，
                    "boxes"  :归一化后的标记框坐标CX CY W H
                    "boxes_xyxy" :未归一化的标记框坐标 xyxy 形状为(num_gt, 4)
                    "image_size_xyxy" : 图像尺寸w h w h  
                    "image_size_xyxy_tgt": 图像尺寸，但是张量形状变了, num_gt, 4
                    "area": 每个gt的面积
                    },
                    ....
                ] 长度==bs
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices_without_aux = self.matcher(outputs_without_aux, targets)
        # 返回一个list 长度为bs 每一个元素是个tuple 每个tuple里有2个向量，
        # 分别对应一张图片的proposal与gt之间的最优分配结果的行位置向量和列位置向量
        # [tuple(row_inds:Tensor, col_inds:Tensor), ... ] lendth == bs
        if self.cfg.MODEL.MDR.IS_TRAIN and self.cfg.MODEL.MDR.IS_FIX:
            return None, indices_without_aux
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices_without_aux, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if self.cfg.MODEL.MDR.IS_TRAIN:
            return losses, indices_without_aux         
        return losses

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cfg, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, use_focal: bool = False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.MDR.ALPHA
            self.focal_loss_gamma = cfg.MODEL.MDR.GAMMA
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes_xyxy"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal:
            # Compute the classification cost.
            alpha = self.focal_loss_alpha
            gamma = self.focal_loss_gamma
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        image_size_out = torch.cat([v["image_size_xyxy"].unsqueeze(0) for v in targets])
        image_size_out = image_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
        image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])

        out_bbox_ = out_bbox / image_size_out
        tgt_bbox_ = tgt_bbox / image_size_tgt
        cost_bbox = torch.cdist(out_bbox_, tgt_bbox_, p=1)

        # Compute the giou cost betwen boxes
        # cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        # 返回一个list 长度为bs 每一个元素是个tuple 每个tuple里有2个向量，
        # 分别对应一张图片的proposal与gt之间的最优分配结果的行位置向量和列位置向量
        # （由于是线性规划问题，所以是一对一的）, 
        # 即就是行位置数组(向量)长度 == 列位置数组(向量)长度
        # 而在匈牙利算法中，gt总是要匹配完的，只不过可能顺序不一样
        # 因此 数组(向量)长度 == 获得匹配的proposal数量 == num_gt
        # [tuple(row_inds:Tensor, col_inds:Tensor), ... ] lendth == bs


#--------------- track loss --------------------------
class TrackCriterion(nn.Module):
    def __init__(self, cfg, weight_dict, eos_coef, losses, use_focal):
        super().__init__()
        self.cfg = cfg
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.num_classes  = 1
        self.use_focal = use_focal
    
    def initialize_for_single_clip(self): 
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 1 # 初始化 多帧clip的 第2帧的 idx
        self.losses_dict = {} # 初始化loss

    def _step(self):
        self._current_frame_idx += 1
    
    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes
 
    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):# 仅仅 针对 轨迹部分 进行分类置信度的训练
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        对于运动模块--仅仅只需要2分类的分类任务即可 对于实例的语义感知这些分类都在检测器中完成
        """
        # import pdb;pdb.set_trace()
        b_idx = 0
        det_idx = indices[:, 0]
        tgt_idx = indices[:, 1]

        src_logits = outputs.pred_logits.unsqueeze(0)
        target_classes_o = (tgt_idx == -1).to(torch.int64)# 正样本为 0  负样本为 1
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)# 全是 1 -- bs num_tracks
        target_classes[b_idx, det_idx] = target_classes_o

        if self.use_focal:
            src_logits = src_logits.flatten(0, 1)
            # prepare one_hot target.
            target_classes = target_classes.flatten(0, 1)
            pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]# 挑选出 所有的 正样本的索引 
            labels = torch.zeros_like(src_logits)
            labels[pos_inds, target_classes[pos_inds]] = 1
            # comp focal loss.       
            class_loss = sigmoid_focal_loss_jit(
                src_logits,
                labels,
                alpha=0.25,
                gamma=2,
                reduction="sum",
            ) / num_boxes
            losses = {'loss_ce_track': class_loss}
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
            losses = {'loss_ce_track': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # import pdb; pdb.set_trace()
        b_idx = 0
        det_idx = indices[:, 0]
        tgt_idx = indices[:, 1]
        
        # import pdb; pdb.set_trace()
        src_boxes = (outputs.pred_boxes.unsqueeze(0))[b_idx, det_idx]
        box_mask = tgt_idx != -1
        src_boxes = src_boxes[box_mask]
        
        target_boxes = targets['boxes_xyxy']
        target_boxes = target_boxes[tgt_idx[box_mask]]
        assert src_boxes.shape[0] == target_boxes.shape[0]
 
        losses = {}
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes, target_boxes))
        losses['loss_giou_track'] = loss_giou.sum() / num_boxes

        image_size = targets["image_size_xyxy_tgt"]# 未填充的输入尺寸
        src_boxes_ = src_boxes / image_size[0:1, :].repeat(src_boxes.shape[0], 1)
        target_boxes_ = target_boxes / image_size[0:1, :].repeat(src_boxes.shape[0], 1)

        loss_bbox = F.l1_loss(src_boxes_, target_boxes_, reduction='none')
        losses['loss_bbox_track'] = loss_bbox.sum() / num_boxes

        return losses
            
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'track_labels': self.loss_labels,
            'track_boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
            
    def calc_loss_per_frame(self, pred, tgt, matched_idx):
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # import pdb; pdb.set_trace()
        # num_boxes = len(pred)
        num_boxes = (pred.track_ids >= 0).sum().item()
        self.num_samples += num_boxes # 这里是叠加形态，意味着 要收集 所有视频帧的 参与训练的样本 
        # 每一帧的参与训练的样本数量 == 所有轨迹查询数量 -- 由于不存在检测查询的参与 因此这里全部都是轨迹预测
        self.sample_device = pred.pred_logits.device

        # Compute all the requested losses
        for loss in self.losses:
            self.losses_dict.update(
                    {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in self.get_loss(loss, pred, tgt, matched_idx, num_boxes=1).items()})
        self._step() # 每 运行一个帧 这里 当前帧id 的计数 +1
    
    def forward(self, outputs):
        # losses of each frame are calculated during the model's forwarding and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            losses[loss_name] /= num_samples
        return losses
        
