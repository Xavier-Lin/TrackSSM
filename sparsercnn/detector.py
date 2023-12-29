#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import lap
from typing import List
import cv2
import numpy as np
import torch, torchvision
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from cython_bbox import bbox_overlaps as bbox_ious
from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.modeling.roi_heads import build_roi_heads
from collections import deque
from detectron2.structures import Boxes, ImageList, Instances
from .track_decoder import TrackSSM
from .loss import SetCriterion, HungarianMatcher, TrackCriterion
from .head import DynamicHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_iou, matched_boxlist_iou
from .util.misc import nested_tensor_from_tensor_list
from .util.visual import plot_tracking
from .tracker.byte_tracker import BYTETracker
from .tracker.sort import Sort
__all__ = ["MDR"]


@META_ARCH_REGISTRY.register()
class MDR(nn.Module):
    """
    Implement MDR
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.MDR.NUM_CLASSES
        self.num_proposals = cfg.MODEL.MDR.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.MDR.HIDDEN_DIM
        self.num_heads = cfg.MODEL.MDR.NUM_HEADS
        self.random_drop = 0.2
        self.iou_drop_thresh = 0.5
        # self.mem_bank_len = 5

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        
        # Build Proposals.
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)
        
        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_channels=[256,256,256,256], roi_input_stride=[4, 8, 16, 32])
        
        # Build refine decoder
        if self.cfg.MODEL.MDR.IS_TRAIN:
            self.ssm_decoder = TrackSSM(
                cfg=cfg, 
                roi_input_channels=[256, 256, 256, 256], 
                roi_input_stride=[4, 8, 16, 32]
            )
            
        # Build combined layers
        self.combine = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=1)
        
        # Loss parameters:
        class_weight = cfg.MODEL.MDR.CLASS_WEIGHT
        giou_weight = cfg.MODEL.MDR.GIOU_WEIGHT
        l1_weight = cfg.MODEL.MDR.L1_WEIGHT
        no_object_weight = cfg.MODEL.MDR.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.MDR.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.MDR.USE_FOCAL

        # Build Criterion.
        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=class_weight, 
                                   cost_bbox=l1_weight, 
                                   cost_giou=giou_weight,
                                   use_focal=self.use_focal)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        if self.cfg.MODEL.MDR.IS_TRAIN:
            weight_dict_track = {}
            for i in range(max(cfg.DATASETS.TRAIN_DATA.SAMPLER_LEN)): # 对于每一个视频帧 都要构建一个对于 检测器 最后一个stage 的损失 
                weight_dict_track.update({"frame_{}_loss_ce_track".format(i): class_weight,
                                    'frame_{}_loss_bbox_track'.format(i): l1_weight,
                                    'frame_{}_loss_giou_track'.format(i): giou_weight,
                                    })
 
        losses = ["labels", "boxes"]

        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=no_object_weight,
                                      losses=losses,
                                      use_focal=self.use_focal)
        if self.cfg.MODEL.MDR.IS_TRAIN:
            track_losses = ["track_labels", "track_boxes"]
            self.track_criterion = TrackCriterion(
                cfg = cfg,
                weight_dict=weight_dict_track,
                eos_coef=no_object_weight,
                losses=track_losses,
                use_focal=self.cfg.MODEL.MDR.USE_FOCAL_R
            )
        
        if cfg.MODEL.MDR.TRACKING:
            self.reset()

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
    
    
    def reset(self):
        self.id_count = 0
        self.tracks = []
        self.sort_tracker = BYTETracker()
        
    def _generate_empty_tracks(self):# 初始化目标实例的属性 一个目标对应一个查询 对应一个 目标id。对应一个 目标索引。对应一个 目标分数。对应一个预测的目标框。对应一个预测的目标类别，对应一个轨迹实例或者是一个检测实例 
        track_instances = Instances((1, 1))
        num_queries, dim = self.init_proposal_features.weight.shape  # (300, 256)
        device = self.init_proposal_features.weight.device
        
        track_instances.pred_embedding = torch.zeros((num_queries, dim), device=device)# 300 256 -- 输出的flow embedding 或者是 hidden state embeddding
        track_instances.track_ids = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)# 初始化每一个查询对应的 轨迹id
        track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long, device=device)# 初始化 每一个查询 的 消失帧长度 
        track_instances.iou = torch.zeros((len(track_instances),), dtype=torch.float, device=device)# 初始化实例的轨迹iou。
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)# 初始化 最后一个stage 轨迹查询/检测查询 预测的置信度分数 
        track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)# 初始化 的最后一个stage 轨迹/检测 查询 在 未填充尺寸上 的边界框预测
        track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float, device=device)# 初始化 最后一个stage 轨迹/检测 查询的类别预测

        # mem_bank_len = self.mem_bank_len # 
        # track_instances.mem_bank = torch.zeros((len(track_instances), mem_bank_len, dim // 2), dtype=torch.float32, device=device)
        # track_instances.mem_padding_mask = torch.ones((len(track_instances), mem_bank_len), dtype=torch.bool, device=device)
        # track_instances.save_period = torch.zeros((len(track_instances), ), dtype=torch.float32, device=device)# num_query
        return track_instances.to(device)
    
    def _forward_det_decoder(self, features, imgs_whwh, vid):
        # Prepare Proposals.
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * imgs_whwh[vid:vid+1, None, :]
        outputs_class, outputs_coord, outputs_embeddings = self.head(features, proposal_boxes, self.init_proposal_features.weight)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]} # (bz, num_proposal, 1/4), 
        if self.deep_supervision:
            out['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])] # 所有当前帧的检测信息 
        out['hs'] = outputs_embeddings # 最后一个stage的查询特征结果 
        return out
    
    def _forward_track_decoder(self, features, track_instances):
        outputs_class, outputs_coord, outputs_embeddings= self.ssm_decoder(features,  track_instances)
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        out['hs'] = outputs_embeddings #  hidden_state_embedding
        return out
    
    def _forward_single_image(self, frame, pre_fpn_feats, imgs_whwh, track_instances: Instances, video_id: int):
        # Feature Extraction. -- 获取fpn特征 
        src = self.backbone(frame)
        features = list()        
        for f in self.in_features:
            feature = src[f]
            features.append(feature)
            
        if pre_fpn_feats is not None:
            # marge feats 
            srcs = self.combine_fpn_features(pre_fpn_feats, features)
    
        if video_id == 0:#  对于第一帧 -- 收集初始帧的检测损失 det_loss_dict
            out = self._forward_det_decoder(features, imgs_whwh, video_id)
        else:
            det_out = self._forward_det_decoder(features, imgs_whwh, video_id)
            # random drop
            track_instances = self._random_drop(track_instances)
            track_out = self._forward_track_decoder(srcs, track_instances)
            out = {'det_out':det_out, 'track_out':track_out}
            
        pre_fpn_feats = features
        return out, pre_fpn_feats, track_instances

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                batched_inputs == [{}, {}, {} , ...    ] 每个字典是
                    { "image": C H W,  "height":,  "width":,  "image_id":,
                      "instances":Instances }
        """
        assert len(batched_inputs) == 1,  "Each video clip must corresponds to a worker."
        images, images_whwh = self.preprocess_image(
            batched_inputs[0] if self.training else batched_inputs
        )# 输出 video clip 填充后的 张量图像 以及 未填充的尺寸。
        
        if self.training:
            self.track_criterion.initialize_for_single_clip() # 初始化 clipmatcher 中的帧序列记录，用于 随后的 多帧训练时 使用
        
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
        }
        sampler_len = len(batched_inputs[0])
        
        if self.training:
            pre_fpn_feats = None
            track_instances = self._generate_empty_tracks()# 对于这里一开始是初始化检测查询 而这个检测查询会在多帧训练的过程中 动态的更新为轨迹查询  但是固定的检测查询的设置是不会变的 
            det_loss_clip = []
            for vid in range(sampler_len):
                if vid == sampler_len -1:
                    is_last = True
                else:
                    is_last = False

                frame = images.tensor[vid].unsqueeze(0)
                frame_res, pre_fpn_feats, track_instances = self._forward_single_image(frame, pre_fpn_feats, images_whwh, track_instances, vid)
 
                gt_instances = [x["instances"].to(self.device) for x in [batched_inputs[0][vid]]]
                curr_targets = self.prepare_targets(gt_instances)
           
                if vid == 0:#  对于第一帧 -- 收集初始帧的检测损失 det_loss_dict
                    # get detection loss from first frame
                    det_loss_dict_0, pre_indices_without_aux = self.criterion(frame_res, curr_targets)#获得与gt匹配的1对1匹配对
                    if det_loss_dict_0 is not None:
                        det_loss_clip.append(self.weighted_det_loss(det_loss_dict_0))        
                            
                    track_instances = self._det_convert_tracks(frame_res, curr_targets, track_instances, pre_indices_without_aux)
                    continue
                frame_res = self._post_process_single_image(frame_res, track_instances, curr_targets[0], images_whwh[vid:vid+1, :], is_last)
                
                track_instances = frame_res['track_instances']
                outputs['pred_logits'].append(frame_res['track_out']['pred_logits']) 
                outputs['pred_boxes'].append(frame_res['track_out']['pred_boxes']) # 
                if frame_res['det_loss'] is not None:
                    det_loss_clip.append(self.weighted_det_loss(frame_res['det_loss']))

                if is_last :
                    # import pdb;pdb.set_trace()
                    if not self.training:
                        outputs['track_instances'] = track_instances
                        return outputs
                    else:
                        # process det loss
                        if len(det_loss_clip) != 0:
                            det_loss_dict = det_loss_clip[0]
                            for det_loss_img in det_loss_clip[1:]:
                                for k, _ in self.criterion.weight_dict.items():
                                    det_loss_dict[k] += det_loss_img[k]
                            
                            for k, _ in self.criterion.weight_dict.items():
                                det_loss_dict[k] /= sampler_len
                        else:
                            det_loss_dict = {}

                        outputs['losses_dict'] = self.track_criterion.losses_dict
                        track_loss_dict = self.track_criterion(outputs)
                        # print("iter {} after model".format(cnt-1))
                        # import pdb;pdb.set_trace()
                        weight_dict = self.track_criterion.weight_dict
                        for k in track_loss_dict.keys():
                            if k in weight_dict:
                                track_loss_dict[k] *= weight_dict[k] 
                        # import pdb;pdb.set_trace()
                        tol_losses = {**det_loss_dict, **track_loss_dict}    
                        return tol_losses
            
                # 所有当前帧回归到前一帧（或后一帧）的轨迹预测信息 
                # weight_dict = self.track_criterion.weight_dict
                # for k in track_loss_dict.keys():
                #     if k in weight_dict:
                #         batch_track_loss_dict[f'f{vid}_'+ k] = track_loss_dict[k] * weight_dict[k]
                #       #get track loss dict
                #   out_loss_dict = {**loss_dict, **batch_track_loss_dict}
                #   assert len(out_loss_dict) == len(loss_dict) + len(batch_track_loss_dict)    
        
    
        else:
            # input size with no_pad -- whwh   images are paded batch of input images
            if isinstance(images, (list, torch.Tensor)):
                images = nested_tensor_from_tensor_list(images)
            frame = images.tensor
            # Feature Extraction. -- 获取fpn特征 
            src = self.backbone(frame)
            features = list()        
            for f in self.in_features:
                feature = src[f]
                features.append(feature)
            # Prepare Proposals.
            proposal_boxes = self.init_proposal_boxes.weight.clone()
            proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
            proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]
            # det decoder
            outputs_class, outputs_coord, outputs_feats = self.head(features, proposal_boxes, self.init_proposal_features.weight)
            # 检测decoder的输出             'pred_feats': outputs_feats (bz, num_proposal, C),
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]} # (bz, num_proposal, 1/4), 
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            feat_pred = outputs_feats
            
            results = self.inference(box_cls, box_pred, feat_pred, images.image_sizes)
            if self.cfg.MODEL.MDR.TRACKING:          
                # det postprocess --- resize
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    inp_boxes = results_per_image.pred_boxes.tensor.clone()
                    post_dets, out_mask = detector_postprocess(
                        results_per_image, height, width, is_clip = False if "MOT17" in self.cfg.DATASETS.TEST_DATA.DATA_NAME else True
                    )
                    processed_results.append([post_dets, inp_boxes[out_mask]])
                
                frame_id = batched_inputs[0]['frame_id']
                if frame_id == 1:
                    trks = self.init_track(processed_results[0], features, input_per_image)
                    return trks
                track_res = self.step(processed_results[0], features, images_whwh, input_per_image)
                # det_instances = self.prepare_dets(processed_results[0], images_whwh)
                
                #vis
                # import pdb;pdb.set_trace()
                # bboxes = det_instances.post_boxes.cpu().numpy()# x1y1x2y2 
                # scores = det_instances.scores.cpu().numpy()
                # bboxes[:, [2, 3]] -= bboxes[:, [0, 1]]
                
                # scores_ = scores#[scores < 0.3]
                # bboxes_ = bboxes#[scores < 0.3]
                
                # im = cv2.imread(input_per_image['file_name'])
                # online_im = plot_tracking(
                #     im, bboxes_, scores_, frame_id=frame_id + 1, fps=1. / 0.1
                # )
                # cv2.imwrite('./sss.jpg', online_im)
                # import pdb;pdb.set_trace()
                
                # track_res = self.sort_tracker.update(det_instances)
                return track_res   
            else:
                if do_postprocess:
                    processed_results = []
                    for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                        height = input_per_image.get("height", image_size[0])
                        width = input_per_image.get("width", image_size[1])
                        r, _ = detector_postprocess(
                            results_per_image, height, width, is_clip = False if "MOT17" in self.cfg.DATASETS.TEST_DATA.DATA_NAME else True
                        )
                        processed_results.append({"instances": r})
                    return processed_results
                else:
                    return results
    
    def weighted_det_loss(self, det_loss_dict):
        weight_dict = self.criterion.weight_dict
        for k in det_loss_dict.keys():
            if k in weight_dict:
                det_loss_dict[k] *= weight_dict[k]
        return det_loss_dict
    
    def _post_process_single_image(self, frame_res, track_instances, curr_targets, img_whwh, is_last):
        # frame_res 最后一个 stage 预测的类别 以及归一化的 bbox 坐标，以及最后一个stage 使用的图像尺寸归一化的参考中心点。 收集除最后一个stage外 所有检测头的输出--作为辅助输出。 最后一个stage的查询特征结果
        # 检测查询与轨迹查询的实例  其中检测查询 仍然是 固定的初始化状态  轨迹查询仍然保持着前一帧的状态 
        # import pdb;pdb.set_trace()
        det_res = frame_res['det_out']
        track_res = frame_res['track_out']
        
        with torch.no_grad():
            if self.training:
                track_scores = track_res['pred_logits'][0, :].sigmoid().max(dim=-1).values# 对预测的类别向量的进行归一化 并取 最高的值 作为查询目标的 预测置信度分数 
                det_scores = det_res['pred_logits'][0, :].sigmoid().max(dim=-1).values
            else:
                track_scores = track_res['pred_logits'][0, :, 0].sigmoid()
                det_scores = det_res['pred_logits'][0, :, 0].sigmoid()
 
        # update tracks information
        assert len(track_instances) == track_scores.shape[0]
        track_instances.scores = track_scores # 对于检测查询 更新预测的置信度分数。对于 轨迹查询也是更新 在当前帧 预测的 置信度分数
        track_instances.pred_logits = track_res['pred_logits'][0] # 对于检测和轨迹查询 均更新其在当前帧的预测目标的类别向量 
        track_instances.pred_boxes = track_res['pred_boxes'][0] # 对于检测和轨迹查询 均更新其在当前帧的归一化bbox坐标
        track_instances.pred_embedding = track_res['hs'][0]  # 对于检测和轨迹查询 均更新其在当前帧的查询特征结果 output
        
        # add new dets information
        new_det_instances = self._generate_empty_tracks()
        new_det_instances.scores = det_scores
        new_det_instances.pred_logits = det_res['pred_logits'][0]
        new_det_instances.pred_boxes = det_res['pred_boxes'][0]
        new_det_instances.pred_embedding = det_res['hs'][0]
        num_dets = len(det_res['hs'][0])
        new_det_instances.imgs_whwh = curr_targets["image_size_xyxy_tgt"][0].view(1, -1).repeat(num_dets, 1) if self.training else img_whwh.repeat(num_dets, 1)
        
        if self.training:
            # the track id will be assigned by the mather.
            det_loss_dict, indices_without_aux, matched_idx = \
                self.match_for_single_frame(track_instances, det_res, curr_targets)
            self.track_criterion.calc_loss_per_frame(track_instances, curr_targets, matched_idx)
            new_det_instances = new_det_instances[indices_without_aux[0][0]]
            # compute pairwise iou
            gt_boxes_xyxy = Boxes(curr_targets['boxes_xyxy'][indices_without_aux[0][1]])
            pred_boxes_xyxy =  Boxes(new_det_instances.pred_boxes)
            new_det_instances.iou = matched_boxlist_iou(pred_boxes_xyxy, gt_boxes_xyxy)
            new_det_instances.track_ids = curr_targets['track_ids'][indices_without_aux[0][1]]
            
            # update track instance iou  更新 获得匹配 的 轨迹对象 和 fp 以及 检测对象的 iou 信息
            matched_mask = matched_idx[:, 1] != -1
            valid_mask = track_instances.track_ids >= 0
            pred_boxes_xyxy = Boxes(track_instances.pred_boxes[matched_idx[:, 0][matched_mask]])
            gt_boxes_xyxy = Boxes(curr_targets['boxes_xyxy'][matched_idx[:, 1][matched_mask]])
            track_instances.iou[matched_idx[:, 0][matched_mask]] = matched_boxlist_iou(pred_boxes_xyxy, gt_boxes_xyxy)
            # get new det instance
            matched_tids = track_instances.track_ids[matched_idx[:, 0][matched_mask]]
            new_tids = new_det_instances.track_ids
            midx, _ = self._track_association_with_ids(matched_tids, new_tids)
            midx = torch.from_numpy(np.array(midx)).to(matched_tids.device).view(-1, 2)
            new_det_instances = new_det_instances[midx[:, 0][~(midx[:, 1] != -1)]]
            # get valid track instance
            track_instances = track_instances[valid_mask]
            # all instance for the next step
            track_instances = Instances.cat([track_instances, new_det_instances])
            frame_res['det_loss'] = det_loss_dict
        else:
            # each track will be assigned an unique global id by the track base.
            # import pdb;pdb.set_trace()
            track_instances = Instances.cat([new_det_instances, track_instances])
            self.track_base.update(track_instances, 0.7, self.cfg.MODEL.MDR.TRACKING_SCORE)# 初始化新检测的轨迹 以及 去掉丢失时长超过5帧的老轨迹

        if not is_last:
            frame_res['track_instances'] = track_instances
        else:
            frame_res['track_instances'] = None
                        
        return frame_res
    
    def match_for_single_frame(self, track_ist: Instances, det_res: dict, curr_gts: dict):# 
        matched_idx, num_disappear_trks =  self._track_association_with_ids(curr_gts['track_ids'], track_ist.track_ids)
        matched_idx = torch.from_numpy(np.array(matched_idx)).to(track_ist.pred_boxes.device).view(-1, 2)
        # get detection loss from per frame
        det_loss_dict, indices_without_aux = self.criterion(det_res, [curr_gts])#获得与gt匹配的1对1匹配对
        return det_loss_dict, indices_without_aux, matched_idx

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size # 未填充的输入图像尺寸 
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)# 未填充的输入尺寸
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy # 未填充输入尺寸下的 归一化 box坐标
            gt_track_ids = targets_per_image.gt_track_ids
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)# 未填充输入尺寸的归一化box坐标 - cxcywh
            target["labels"] = gt_classes.to(self.device)
            target['track_ids'] = gt_track_ids.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)# 未填充输入尺寸的归一化box坐标 - cxcywh
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)# 未填充输入尺寸的bbox坐标 - xyxy
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets
    

    def inference(self, box_cls, box_pred, feat_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        C = feat_pred.shape[-1]
        results = []
        # init_feats = self.init_proposal_features.weight.clone()

        if self.use_focal:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device).\
                     unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, feat_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, feat_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]
                feat_pred_per_image = feat_pred_per_image.view(-1, 1, C).repeat(1, self.num_classes, 1).view(-1, C)
                feat_pred_per_image = feat_pred_per_image[topk_indices]
                # init_feats_per_img = init_feats
                # init_feats_per_img = init_feats_per_img.view(-1, 1, C).repeat(1, self.num_classes, 1).view(-1, C)
                # init_feats_per_img = init_feats_per_img[topk_indices]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                result.pred_feats = feat_pred_per_image
                # result.init_feats = init_feats_per_img
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                scores, labels, box_pred, image_sizes
            )):
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results
    
    def _random_drop(self, matched_tracks:Instances): #track_instances.obj_idxes >= 0 
        if self.training:
            active_track_instances = self._random_drop_tracks(matched_tracks)# 随机drop获得匹配的实例
            active_track_instances.track_ids[active_track_instances.iou <= self.iou_drop_thresh] = -1
        else:
            active_track_instances = matched_tracks[matched_tracks.track_ids >= 0]
        return active_track_instances
    
    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        if self.random_drop > 0 and len(track_instances) > 0:
            keep_idxes = torch.rand_like(track_instances.scores) > self.random_drop
            track_instances = track_instances[keep_idxes]
        return track_instances

    def preprocess_image(self, inputs):
        """
        Normalize, pad and batch the input images.
        """
        images_list = [self.normalizer(x["image"].to(self.device)) for x in inputs]
        images = ImageList.from_tensors(images_list, self.size_divisibility)# 对于一个视频clip 填充统一尺寸 这个还是比较友好的
        images_whwh = list()
        for bi in inputs:
            h, w = bi["image"].shape[-2:]# 未填充的尺寸
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)
        return images, images_whwh

    def _det_convert_tracks(self, output, targets, track_instances, indices_without_aux):
        assert len(targets) == len(indices_without_aux) == 1
        targets = targets[0]
        indices_without_aux = indices_without_aux[0]
        outputs_feats = output['hs']
        
        track_instances = track_instances[indices_without_aux[0]]
        track_instances.track_ids = targets["track_ids"][indices_without_aux[1]].long()
        track_instances.pred_boxes = output['pred_boxes'][0][indices_without_aux[0]].detach()
        track_instances.pred_embedding = outputs_feats[0][indices_without_aux[0]].clone()# 300 256 -- 输出的flow embedding 或者是 hidden state embeddding
        track_instances.scores = output['pred_logits'][0][indices_without_aux[0]].sigmoid().squeeze(-1)
        track_instances.pred_logits = output['pred_logits'][0][indices_without_aux[0]]
        track_instances.imgs_whwh = targets["image_size_xyxy_tgt"][indices_without_aux[1]]
        # compute pairwise iou
        gt_boxes_xyxy = Boxes(targets['boxes_xyxy'][indices_without_aux[1]])
        pred_boxes_xyxy =  Boxes(output['pred_boxes'][0][indices_without_aux[0]])
        track_instances.iou = matched_boxlist_iou(pred_boxes_xyxy, gt_boxes_xyxy)

        return track_instances
    
    @torch.no_grad()
    def _track_association_with_ids(self, tgt_ids, prev_track_ids): 
        tgt_ids = tgt_ids.cpu().numpy().tolist()
        matched_idx = []
        num_disappear_track = 0 
        for i, track_id in enumerate(prev_track_ids):
            if track_id >= 0:
                if track_id in tgt_ids:
                    gt_idx = tgt_ids.index(track_id)
                    matched_idx.append([i, gt_idx])
                else:
                    num_disappear_track += 1 
                    matched_idx.append([i, -1]) # 在当前帧丢失的轨迹
            else: 
                matched_idx.append([i, -1])# 之前添加的假阳性轨迹 
        return matched_idx, num_disappear_track
    

    def init_track(self, results: List, fpn_feat: List, input_per_image):
        post_output, nms_out_index, conf_mask = self.postprocess(
            results[0].pred_boxes.tensor, results[0].pred_classes, results[0].scores, 1, self.cfg.MODEL.MDR.TRACKING_SCORE)

        curr_pred_feats = results[0].pred_feats[conf_mask][nms_out_index].cpu().numpy()
        inp_boxes = results[1][conf_mask][nms_out_index].cpu().numpy()
        post_output = post_output.cpu().numpy()
        ret = []
        for i, (det, inp_box) in enumerate(zip(post_output, inp_boxes)):
            item = {}
            item['bbox'] = det[:4] # xyxy
            item['score'] = det[4] # score
            item['bbox_inp'] = inp_box # xyxy inp
            item['active'] = 1
            item['age'] = 1
            self.id_count += 1
            item['track_id'] = self.id_count
            item['pred_feats'] = curr_pred_feats[i]
            item['detla_deque'] = deque([], maxlen = self.cfg.MODEL.MDR.MAX_FRAME_DIST)
            wh = det[2: 4] - det[ :2]
            whwh = np.concatenate([wh, wh],axis = -1)
            item['detla_deque'].append((1 / 100000) * whwh) # detla_x1 detla_y1 detla_x2 detla_y2
            ret.append(item) 
            
        ret.append(fpn_feat)
        self.tracks = ret
        # self.pre_img = cv2.imread(input_per_image['file_name'])
        return ret[:-1]
    
    def combine_fpn_features(self, a_fpn, b_fpn):
        # marge feats 
        srcs = [] # 所有4个level上 当前帧 和 前一帧聚合的特征 
        for i, (a_feat, b_feat) in enumerate(zip(a_fpn, b_fpn)):
            srcs.append(self.combine(torch.cat([a_feat, b_feat], dim=1)))
        return srcs
    
    def prepare_dets(self, curr_results:List, images_whwh):
        curr_results_post, curr_boxes_inp = curr_results[0], curr_results[1]
        post_output, nms_out_index, conf_mask = self.postprocess(
            curr_results_post.pred_boxes.tensor, 
            curr_results_post.pred_classes, 
            curr_results_post.scores, 
            num_classes = 1, 
            conf_thre  =self.cfg.MODEL.MDR.TRACKING_SCORE
            # conf_thre = 0.2,
            # nms_thre = 0.7
        )
        curr_res_instances = Instances((1,1))
        curr_res_instances.pred_feats = curr_results_post.pred_feats[conf_mask][nms_out_index] 
        curr_res_instances.inp_boxes = curr_boxes_inp[conf_mask][nms_out_index] # 获取真正的当前帧检测
        curr_res_instances.post_boxes = post_output[:, :4]
        curr_res_instances.scores = post_output[:, 4]
        curr_res_instances.imgs_whwh  = images_whwh.repeat(len(curr_res_instances), 1)
        return curr_res_instances
    
    def get_coord_scale(self, inp_per_img):
        ori_h, ori_w = inp_per_img['height'], inp_per_img['width']
        inp_h, inp_w = inp_per_img['image'].shape[1:]
        scale = min(inp_h/float(ori_h), inp_w/float(ori_w))
        return scale
    
    def step(self, curr_results:List, fpn_feats:List, images_whwh, inp_per_img):
        device = fpn_feats[0].device
        dtype = fpn_feats[0].dtype
        scale = self.get_coord_scale(inp_per_img)
        
        det_instances = self.prepare_dets(curr_results, images_whwh)
        

        prev_active_tracks = Instances((1,1))
        pre_inactive_tracks = Instances((1,1))
        act_bbox_inp, act_pred_feats, inact_bbox_inp, inact_pred_feats = [], [], [], [] 
        act_tracks_idx, inact_tracks_idx = [], []
        for i, track in enumerate(self.tracks[:-1]):
            if track['active'] > 0:
                act_bbox_inp.append(track['bbox'] * scale)
                act_pred_feats.append(track['pred_feats'])
                act_tracks_idx.append(i)
            else:
                inact_bbox_inp.append(track['bbox'] * scale)
                inact_pred_feats.append(track['pred_feats'])
                inact_tracks_idx.append(i)
                
        prev_active_tracks.inp_boxes = torch.as_tensor(act_bbox_inp, device=device, dtype=dtype)
        prev_active_tracks.pred_feats = torch.as_tensor(act_pred_feats, device=device, dtype=dtype)
        prev_active_tracks.imgs_whwh = images_whwh.repeat(len(prev_active_tracks), 1)
        N = len(prev_active_tracks)
        
        pre_inactive_tracks.inp_boxes = torch.as_tensor(inact_bbox_inp, device=device, dtype=dtype)
        pre_inactive_tracks.pred_feats = torch.as_tensor(inact_pred_feats, device=device, dtype=dtype)
        pre_inactive_tracks.imgs_whwh = images_whwh.repeat(len(pre_inactive_tracks), 1)
        
        pre_tracks = Instances.cat([prev_active_tracks, pre_inactive_tracks])
        track_idxes = act_tracks_idx + inact_tracks_idx

        # marge feats 
        srcs = self.combine_fpn_features(self.tracks[-1], fpn_feats)

        # regressive det to tracks from prev_frame.   i_track_instances, vaild_pre_gt_boxes, vaild_pair_mask
        track_scores, track_coords = self.refine_decoder(
            srcs,  
            pre_tracks
        )

        remain_regress_boxes = (track_coords[0] / scale).cpu().numpy()
        track_coords_cpu = track_coords[0].cpu().numpy()
        
        # import pdb;pdb.set_trace()
        # track_scores = track_scores.sigmoid()
        
        # import pdb;pdb.set_trace()
        # img = cv2.imread(inp_per_img['file_name']).copy()
        # for b in remain_regress_boxes:
        #     cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0,0,255), 2)
        # cv2.imwrite("./sss.jpg", img)
        # import pdb;pdb.set_trace()
        
        # import pdb;pdb.set_trace()
        # img = self.pre_img.copy()
        # for b in self.tracks[:-1]:
        #     b = b['bbox']
        #     cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0,0,255), 2)
        # cv2.imwrite("./sss.jpg", img)
        # import pdb;pdb.set_trace()
 
        det_boxes = det_instances.post_boxes.cpu().numpy()
        dist = self.iou_distance(remain_regress_boxes, det_boxes)
        matched_indices, unmatched_tracks, unmatched_dets = self.linear_assignment(dist, 0.5)
        
        # 检测格式的预处理 
        curr_res_inp_boxes = det_instances.inp_boxes.cpu().numpy()
        curr_res_pred_feats = det_instances.pred_feats.cpu().numpy()
        curr_res_post_output = det_boxes
        curr_res_scores = det_instances.scores.cpu().numpy()
        curr_dets = []
        for i, (det, score, inp_box) in enumerate(zip(curr_res_post_output, curr_res_scores, curr_res_inp_boxes)):
            item = {}
            item['bbox'] = det
            item['bbox_inp'] = inp_box
            item['score'] = score
            item['pred_feats'] = curr_res_pred_feats[i]
            item['detla_deque'] = deque([], maxlen = self.cfg.MODEL.MDR.MAX_FRAME_DIST)
            curr_dets.append(item)
        
        
        ret = []
        for m in matched_indices:
            track = curr_dets[m[1]]
            track['track_id'] = self.tracks[:-1][track_idxes[m[0]]]["track_id"]
            track['age'] = 1
            track['active'] = 1

            pre_track_deque = self.tracks[:-1][track_idxes[m[0]]]["detla_deque"]
            pre_bbox = self.tracks[:-1][track_idxes[m[0]]]['bbox']
            detla_xy = track['bbox'] - pre_bbox
            pre_track_deque.append(detla_xy)
            track['detla_deque'] = pre_track_deque
            ret.append(track)
        
        # import pdb;pdb.set_trace()
        for i in unmatched_dets:
            track = curr_dets[i]
            if track['score'] > self.cfg.MODEL.MDR.NEW_TRACKING_THRES:
                self.id_count += 1
                track['track_id'] = self.id_count
                track['age'] = 1
                track['active'] =  1
                wh = track['bbox'][2:4] - track['bbox'][:2]
                whwh = np.concatenate([wh, wh],axis = -1)
                track['detla_deque'].append( (1 / 100000) * whwh )
                ret.append(track)
                
        for i in unmatched_tracks:# 对于丢失的轨迹 其 embedding不会发生更新 因此 它的embedding最多仅仅只能管10帧长度 这是由训练的方式决定的 ，只有被关联到 embedding的信息才会更新
            track = self.tracks[:-1][track_idxes[i]]
            if track['age'] < self.cfg.MODEL.MDR.MAX_FRAME_DIST * 3:
                if track['active'] > 0 or track['age'] < self.cfg.MODEL.MDR.MAX_FRAME_DIST:
                    track['age'] += 1
                    track['active'] = 0
                    pre_box = track['bbox']
                    track['bbox'] = remain_regress_boxes[i]
                    track['bbox_inp'] = track_coords_cpu[i]
                    new_detla = track['bbox'] - pre_box
                    track['detla_deque'].append(new_detla)
                    ret.append(track)
                else:
                    track['age'] += 1
                    track['active'] = 0
                    pre_box = track['bbox']
                    new_detla = self.tracklets_smooth(track['detla_deque'])
                    track['bbox'] = track['bbox'] + new_detla
                    track['bbox_inp'] = track['bbox'] * scale
                    # track['bbox'] = remain_regress_boxes[i]
                    # track['bbox_inp'] = track_coords_cpu[i]
                    # new_detla = track['bbox'] - pre_box
                    track['detla_deque'].append(new_detla)
                    ret.append(track)
                    
        
            # import pdb;pdb.set_trace()
            # img = cv2.imread(inp_per_img['file_name']).copy()
            # for track in lost_stracks:
            #     b = track['bbox']
            #     cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0,0,255), 2)
            # cv2.imwrite("./sss.jpg", img)
            # import pdb;pdb.set_trace()

        ret.append(fpn_feats)
        self.tracks = ret
        # self.pre_img = cv2.imread(inp_per_img['file_name'])
        return ret[:-1]
    
    def greedy_assignment(self, dist):
        matched_indices = []
        if dist.shape[1] == 0:
            return np.array(matched_indices, np.int32).reshape(-1, 2)
        for i in range(dist.shape[0]):
            j = dist[i].argmin()
            if dist[i][j] < 0.4:
                dist[:, j] = 1.
                matched_indices.append([i, j])
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    
    def postprocess(self, pred_boxes, pred_cls, pred_scores, num_classes, conf_thre=0.01, nms_thre=0.8):
        predictions = torch.cat([pred_boxes, pred_scores.unsqueeze(-1), pred_cls.unsqueeze(-1)], 1)
        predictions = predictions.unsqueeze(0)
        output = [None for _ in range(len(predictions))]
        for i, image_pred in enumerate(predictions):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue

            conf_mask = image_pred[:, 4] >= conf_thre
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_pred)
            detections = image_pred[conf_mask]
            if not detections.size(0):
                continue

            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] ,
                detections[:, 5],
                nms_thre,
            )
            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output[0], nms_out_index, conf_mask
    
    def iou_distance(self, atracks, btracks):
        """
        Compute cost based on IoU
        :type atracks: list[STrack]
        :type btracks: list[STrack]

        :rtype cost_matrix np.ndarray
        """
        def ious(atlbrs, btlbrs):
            """
            Compute cost based on IoU
            :type atlbrs: list[tlbr] | np.ndarray
            :type atlbrs: list[tlbr] | np.ndarray

            :rtype ious np.ndarray
            """
            ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
            if ious.size == 0:
                return ious

            ious = bbox_ious(
                np.ascontiguousarray(atlbrs, dtype=np.float),
                np.ascontiguousarray(btlbrs, dtype=np.float)
            )

            return ious

        if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
            atlbrs = atracks
            btlbrs = btracks
        else:
            atlbrs = [track.tlbr for track in atracks]
            btlbrs = [track.tlbr for track in btracks]
        _ious = ious(atlbrs, btlbrs)
        cost_matrix = 1 - _ious

        return cost_matrix
    
    def tracklets_smooth(self, tracks_deque: deque):
        tracklets_len = len(tracks_deque)
        cs = []
        D = 0
 
        for i in range(tracklets_len):
            s = np.exp(-(tracklets_len - 1 - i + 0.01 ))
            cs.append(s) 
            D += (tracks_deque[i] * s)
        
        new_detla = D / (np.mean(np.array(cs)) * tracklets_len)
        return new_detla
    
    def linear_assignment(self, cost_matrix, thresh):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        matches, unmatched_a, unmatched_b = [], [], []
        cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
        matches = np.asarray(matches)
        return matches, unmatched_a, unmatched_b