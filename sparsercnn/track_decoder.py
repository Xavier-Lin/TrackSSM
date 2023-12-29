import copy
import math
from typing import Optional, List

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes, Instances
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, matched_boxlist_iou
_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)

class TrackSSM(nn.Module):
    def __init__(self, cfg, roi_input_channels, roi_input_stride):
        super().__init__()
        self.cfg = cfg
        self.input_stride = roi_input_stride
        # Build RoI.
        self.box_pooler = self._init_box_pooler(cfg, roi_input_channels, roi_input_stride, cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION)
        
        # Build heads.
        num_classes = cfg.MODEL.MDR.TRACK_NUM_CLS 
        d_model = cfg.MODEL.MDR.HIDDEN_DIM
        nhead = cfg.MODEL.MDR.NHEADS
        dropout = cfg.MODEL.MDR.DROPOUT
        activation = cfg.MODEL.MDR.ACTIVATION
        self.motion_head = MotionHead(cfg, d_model, nhead, dropout, activation)      
        # 
        # self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        self.ref_point_head = nn.Linear(2 * d_model, d_model)
  
        # Init parameters.
        self.boxes_ratio = self.cfg.MODEL.MDR.BOX_RATIO
        self.use_focal = cfg.MODEL.MDR.USE_FOCAL_R
        self.num_classes = num_classes
        self.return_intermediate = cfg.MODEL.MDR.DEEP_SUPERVISION
        if self.use_focal:
            prior_prob = cfg.MODEL.MDR.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            # initialize the bias for focal loss.
            if self.use_focal:
                if p.shape[-1] == self.num_classes:
                    nn.init.constant_(p, self.bias_value)
    
    @staticmethod
    def _init_box_pooler(cfg, roi_input_channels, roi_input_stride, pooler_resolution):
        pooler_scales = tuple(1.0 / k for k in roi_input_stride)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = roi_input_channels
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler
    
    def add_boxes_noise(self, prev_gt_boxes, groups = 5):
        device = prev_gt_boxes.device
        prev_gt_boxes_groups = prev_gt_boxes.unsqueeze(0).repeat(groups, 1, 1)
        
        w, h = prev_gt_boxes_groups[..., 2] - prev_gt_boxes_groups[..., 0],  prev_gt_boxes_groups[..., 3] - prev_gt_boxes_groups[..., 1]
        cx, cy = (prev_gt_boxes_groups[..., 2] + prev_gt_boxes_groups[..., 0])/2, (prev_gt_boxes_groups[..., 3] + prev_gt_boxes_groups[..., 1])/2
   
        w_scale = (torch.rand(groups).to(device))  * (self.boxes_ratio[1] - self.boxes_ratio[0]) + self.boxes_ratio[0]
        h_scale = (torch.rand(groups).to(device))  * (self.boxes_ratio[1] - self.boxes_ratio[0]) + self.boxes_ratio[0]
        new_w = w * w_scale[:, None]
        new_h = h * h_scale[:, None]
        
        detla_x = (0.4 * w / 2) * torch.rand(groups)[:, None].to(device) 
        detla_x *= (torch.randn(groups)[:, None].to(device)  > 0.0).int() * 2 - 1
        detla_y = (0.4 * h / 2) * torch.rand(groups)[:, None].to(device)   
        detla_y *= (torch.randn(groups)[:, None].to(device)  > 0.0).int() * 2 - 1
        new_cx = cx + detla_x
        new_cy = cy + detla_y
        
        new_boxes = torch.stack([new_cx, new_cy, new_w, new_h], dim=-1)# cx cy w h
        return new_boxes
    
    def get_vaild_ratio(self, srcs, img_whwh):
        spatial_whwh = torch.as_tensor([[f.shape[-1], f.shape[-2]] for f in srcs], device=srcs[0].device)
        vaild_whwh = torch.as_tensor(self.input_stride, device=img_whwh.device)
        vaild_whwh = (img_whwh[None, :] / vaild_whwh[:, None])[:, 0:2]
        vaild_ratio = vaild_whwh / spatial_whwh
        return vaild_ratio
    
    def get_atten_mask(self,num_groups, num_noise_tracks, num_tracks, device):
        dim = num_groups * num_noise_tracks + num_tracks
        atten_mask = torch.ones((dim, dim), device=device)
        for i in range(num_groups + 1):
            if i < num_groups:
                atten_mask[i * num_noise_tracks: i * num_noise_tracks + num_noise_tracks, i * num_noise_tracks: i * num_noise_tracks + num_noise_tracks] = 0
            else:
                atten_mask[i * num_noise_tracks: i * num_noise_tracks + num_tracks, i * num_noise_tracks: i * num_noise_tracks + num_tracks] = 0
        return atten_mask
                    
    def _update_boxes_and_feats(self, tr_instances: Instances, vaild_ratio: torch.Tensor):
        # num_trks = len(tr_instances)
        img_whwh_nopad = tr_instances.imgs_whwh[0]

        tr_boxes = tr_instances.pred_boxes.clone() if self.training else tr_instances.inp_boxes.clone()
        all_objs_boxes = box_xyxy_to_cxcywh(tr_boxes)# cx cy w h
        
        # get corrsponding query feature
        query_feats = tr_instances.pred_embedding.clone()
        
        # get atten mask
        # if self.training:
        #     atten_mask = self.get_atten_mask(0, len(vaild_pre_gt_boxes), num_trks, all_objs_boxes.device)
        # else:
        atten_mask = None
        
        # get postion embedding
        vaild_ratio_ = torch.cat([vaild_ratio, vaild_ratio], dim=1)
        normalized_all_obj_boxes = (all_objs_boxes / img_whwh_nopad[None])[None] * vaild_ratio_[:, None, :]
        assert (normalized_all_obj_boxes[0] - normalized_all_obj_boxes[-1]).sum() == 0
        all_pos_embeddding = gen_sineembed_for_position(normalized_all_obj_boxes[0][:, None, :])# n_query, bs, _ = pos_tensor.size()  sineembed_tensor = torch.zeros(n_query, bs, 256)
        
        return all_objs_boxes, query_feats, atten_mask, all_pos_embeddding
       
    def forward(self, srcs: List,  track_instances: Instances):
        # import pdb;pdb.set_trace()
        bs = srcs[0].shape[0]
        
        vaild_ratio = self.get_vaild_ratio(srcs, track_instances.imgs_whwh[0])
        group_boxes, group_query_feats, atten_mask, group_pos_embeddding = self._update_boxes_and_feats(
            track_instances, vaild_ratio
        )
        
        bboxes = box_cxcywh_to_xyxy(group_boxes).unsqueeze(0)
        proposal_features = group_query_feats[None].repeat(1, bs, 1)
        pos_embedding = self.ref_point_head(group_pos_embeddding).permute(1, 0, 2)
        
        vaild_ratio_whwh = torch.cat([vaild_ratio, vaild_ratio], dim =1)
        assert (vaild_ratio_whwh[0] - vaild_ratio_whwh[-1]).sum() == 0
        class_logits, pred_bboxes, hidden_state_embbedding = self.motion_head(srcs, bboxes, proposal_features, self.box_pooler, pos_embedding, atten_mask)

        return class_logits, pred_bboxes, hidden_state_embbedding
    

class DynamicConv(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.MDR.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.MDR.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.MDR.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (7*7, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)# 474 49 C
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)#  N 1 2*64*C

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)# N  C 64
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)# N  64  C

        features = torch.bmm(features, param1)# N 7*7 C @ N C 64
        features = self.norm1(features)# N 7*7 64
        features = self.activation(features)# N 7*7 64

        features = torch.bmm(features, param2)# N 7*7 64 @ N 64  C
        features = self.norm2(features)
        features = self.activation(features)# N 7*7  C

        features = features.flatten(1) # N 7*7*C
        features = self.out_layer(features)# N 7*7*C -> N C
        features = self.norm3(features)
        features = self.activation(features)

        return features


class MotionHead(nn.Module):
    def __init__(self, cfg, d_model, nhead, dropout, activation="relu" ):
        super().__init__()

        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.motion = MotionE(cfg)
        self.inst_interact = DynamicConv(cfg)
        self.forward_ffn  = FFN(d_model, drop_ratio=dropout, output_dim=d_model)
      
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
 
        # cls.
        num_cls = cfg.MODEL.MDR.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.MDR.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)
        
        # pred.
        if cfg.MODEL.MDR.USE_FOCAL:
            self.class_logits = nn.Linear(d_model, 1)
        else:
            self.class_logits = nn.Linear(d_model, 2)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = _DEFAULT_SCALE_CLAMP
        
    def scale_bboxes(self, boxes: torch.Tensor):
        scaled_boxes  = box_xyxy_to_cxcywh(boxes)
        scaled_boxes[:, 2] *= 2
        scaled_boxes[:, 3] *= 2
        scaled_boxes = box_cxcywh_to_xyxy(scaled_boxes)
        return scaled_boxes
    
    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx = deltas[:, 0::4] 
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4] 
        dh = deltas[:, 3::4] 
        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * (widths[:, None] / 2) + ctr_x[:, None]
        pred_ctr_y = dy * (heights[:, None] / 2) + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes

    def forward(self, features, bboxes, embedding_features, pooler, position_embedding, atten_mask):
        """
        features 网络的特征图 features = [ p2.feat,  p3.feat,   p4.feat,    p5.feat] 
                通道统一256 每个元素的大小为 (bz C h w)        C=256
        bboxes 初始化的框 # (bz, 300, 4)                      num_proposal=300
        pro_features 初始化的提议特征  (1, bz*300, 256) 
        pooler ROI池化头  _init_box_pooler(cfg, input_shape)
        """
        # import pdb;pdb.set_trace()
        N, nr_boxes = bboxes.shape[:2]# bz N
        # use self attention entrance embedding 
        q = k = embedding_features = embedding_features.view(N, nr_boxes, self.d_model)
        pro_features2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embedding_features.transpose(0, 1), attn_mask = atten_mask)[0].transpose(0, 1)
        embedding_features = embedding_features + self.dropout1(pro_features2)
        embedding_features = self.norm1(embedding_features)
        
        # get local region flow_feature.
        proposal_boxes = list()# 提议框在refine的过程中也是对应于原图尺寸下的提议
        for b in range(N):
            proposal_boxes.append(Boxes(self.scale_bboxes(bboxes[b])))
        flow_features = pooler(features, proposal_boxes)  
        flow_features = flow_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1) # 根据roi提取流引导特征
        #  features=[ p2.feat,  p3.feat,   p4.feat,    p5.feat]  bz C h w 
        # -> (roi_output_size**2, bz*num_proposal,  C)  
        
        # get instance specific flow_feature
        embedding_features = embedding_features.view(1, N * nr_boxes, self.d_model)
        pro_features2 = self.inst_interact(embedding_features, flow_features)
        embedding_features = embedding_features + self.dropout2(pro_features2)
        flow_features = self.norm2(embedding_features)
        
        # get flow hidden state embeddings
        hidden_embeddings = flow_features.clone()
        
        # motion prediction.
        position_embedding = position_embedding.view(1, N * nr_boxes, self.d_model)
        reg_feature = self.motion(position_embedding, flow_features)
        cls_feature = self.forward_ffn(flow_features)
        
        cls_feature = cls_feature.transpose(0, 1).reshape(N * nr_boxes, -1)
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))
        # pred_bboxes = self.bboxes_delta(reg_feature).sigmoid()# 归一化的 cx cy w h
        # import pdb;pdb.set_trace()
        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), hidden_embeddings.view(N, nr_boxes, -1)


class MotionE(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.MDR.HIDDEN_DIM
        drop_ratio = cfg.MODEL.MDR.DROPOUT 
        d_ffn = 1024
        
        self.stem = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 1)
        self.stem_1 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim // 2, 2)
        self.stem_2 = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim // 2, 2)
        # ffn_1
        self.motion_1 = FFN(self.hidden_dim // 2, d_ffn, drop_ratio, self.hidden_dim // 2)
        # ffn_2
        self.motion_2 = FFN(self.hidden_dim, d_ffn, drop_ratio, self.hidden_dim)

    def forward(self, position_features, flow_features):
        '''
        position_features: (1,  N * nr_boxes, self.d_model)
        flow_features: (1, N * nr_boxes, self.d_model) --- roi区域内针对对应embedding信息的 流特征 这样的话 区域内不同的实例流特征就可以区分开来
        '''
        position_features = position_features.permute(1, 0, 2)# N 1 C
        features = flow_features.permute(1, 2, 0) # N C 1
        features = torch.bmm(features, features.transpose(1,2))
        features = self.stem(features)# N C C -- 生成流引导特征
       
        matrix1 = self.stem_1(features) # N  C    C/2
        matrix2 = self.stem_2(features).permute(0, 2, 1) # N  C/2  C -- 生成引导矩阵

        features = torch.bmm(position_features, matrix1)# N 1 C @ N C C/2 -> N  1  C/2
        features = self.motion_1(features)# N  1  C/2

        features = torch.bmm(features, matrix2)#  N  1  C/2 @ N  C/2  C ->  N  1  C
        features = self.motion_2(features)#  N  1  C

        features = features.flatten(1) # N C
        return features
    
class TAN(nn.Module):
    def __init__(self, d_model, ):
        super().__init__()
        self.combine = nn.Conv2d(d_model * 2, d_model, kernel_size=1)
        
        pass
        
    def forward(self):
        pass

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class FFN(nn.Module):
    def __init__(self, input_dim = 256, d_ffn = 1024, drop_ratio = 0.0, output_dim = 256, act = 'relu'):
        super().__init__()
        assert input_dim == output_dim
        
        self.linear1 = nn.Linear(input_dim, d_ffn)
        self.activation = _get_activation_fn(act)
        self.dropout2 = nn.Dropout(drop_ratio)
        self.linear2 = nn.Linear(d_ffn, output_dim)
        self.dropout3 = nn.Dropout(drop_ratio)
        self.norm2 = nn.LayerNorm(output_dim)
        
    def forward(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src 

def gen_sineembed_for_position(pos_tensor):# normalized cx cy w h
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
