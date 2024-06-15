import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba_encoder import Mamba_encoder
from einops import rearrange, repeat

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, class_token=False):# # l b 256 -- 前五帧的 归一化边界框的位置信息 和 归一化差分运动信息 --- 这里 与视觉的 处理方式 大同小异 -- 都是 在 序列层面上 的编码
        x = x.permute(1, 2, 0) # b 256 l ---  b c l
        num_feats = x.shape[1]# --- 序列的维度
        num_pos_feats = num_feats# --- 序列的维度
        mask = torch.zeros(x.shape[0], x.shape[2], device=x.device).to(torch.bool)# b l --- b l
        batch = mask.shape[0] # batch size
        assert mask is not None
        not_mask = ~mask # b l --- 1
        y_embed = not_mask.cumsum(1, dtype=torch.float32)# 【1 2 3 4 5】--- b l --- 形成 序列方向上的 位置编码

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)# 0 1 2 ... 255 --- [256]
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)# [256]

        pos_y = y_embed[:, :, None] / dim_t # b l 256
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)# b l 256
        return pos_y


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module): # 最朴素的transformer编码器网络： 自注意力+FFN

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):# 256 8 512 0.1 relu F
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask = None,
                     src_key_padding_mask = None,
                     pos = None):
        src2 = self.self_attn(src, src, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask = None,
                    src_key_padding_mask = None,
                    pos = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask = None,
                src_key_padding_mask = None,
                pos = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class History_motion_embedding(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation='relu', normalize_before=False, pos_type='sin'):
        super(History_motion_embedding, self).__init__()
        self.cascade_num = 6
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))# 1 1 256
        self.trca = nn.ModuleList()
        for _ in range(self.cascade_num):
            self.trca.append(TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                   dropout, activation, normalize_before))
        # 6x transformer编码器网络 但是feedforward是512 大大减少了计算量，其实 DETR 本身的 推理速度 就是很快的 可以达到 25FPS
        self.proj = nn.Linear(8, d_model)# 这一部分是将 cx cy w h detla_cx detla_cy detla_w detla_h 投射至高维度线性空间，以获取更加丰富的表示
        if pos_type == 'sin':# 默认使用的是余弦位置编码
            self.pose_encoding = PositionEmbeddingSine(normalize=True)


    def forward(self, x): # 输入的x为condition 也就是 轨迹的 历史帧的位置信息 和 运动信息 b 5 8 ---- 默认 使用的是 历史帧的5帧
        if len(x.shape) == 2:
            x = x.unsqueeze(0).to(self.cls_token)
        else:
            x = x.to(self.cls_token)

        q_patch = self.proj(x).permute(1, 0, 2)# l b 256
        # pos = self.pose_encoding(q_patch).transpose(0, 1)# l b 256
        _, b, d = q_patch.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b).permute(1, 0, 2).contiguous()# 给 每一个 采样的短轨迹( 短序列 ) 附加一个 cls-token --- 1 b 256
        encoder_patch = torch.cat((q_patch, cls_tokens), dim=0)# l+1 b 256
        for i in range(self.cascade_num):
            en_out = self.trca[i](src=encoder_patch)# 这里训练阶段的位置编码没有起到什么作用啊
            encoder_patch = en_out
        out = en_out[0].view(b, 1, d).contiguous()# b 1 256 -- 这里为什么要用【0】呢？--这是因为 将历史帧 的运动和位置信息 合并到cls_token当中，并使用更新后的cls_token作为 融合 历史运动 和 位置信息的 条件embedding
        return out


class Time_info_aggregation(nn.Module):
    def __init__(self, d_model = 256, n_layer = 6, v_size = 8):
        super(Time_info_aggregation, self).__init__()
 
        self.mamba_net = Mamba_encoder(
            d_model = d_model,
            n_layer = n_layer,
            vocab_size = v_size,
            rms_norm  = True,
            fused_add_norm= True
        )
        
    def forward(self, x): # 输入的x为condition 也就是 轨迹的 历史帧的位置信息 和 运动信息 b 5 8 ---- 默认 使用的是 历史帧的5帧
        out = self.mamba_net(x)
        return out
        