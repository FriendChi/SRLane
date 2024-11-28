from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiSegmentAttention(nn.Module):
    """Multi-Segment Attention (MSA) for lane detection.

    Args:
        embed_dim: Channel dimension.
        num_groups: Number of lane segment groups.
        dropout: dropout ratio.
    """
    def __init__(self,
                 embed_dim: int,
                 num_groups: int = 1,
                 dropout: float = 0.0, ):
        super(MultiSegmentAttention, self).__init__()
        self.embed_dim = embed_dim  # 通道维度，表示特征输入的总维度。
        self.num_groups = num_groups  # 分组数量，用于将车道分成若干段。
        self.dropout = dropout  # Dropout比例，避免过拟合。

        # 检查通道数是否可以被分组数整除。
        if embed_dim % num_groups != 0:
            raise ValueError(f"Embed_dim ({embed_dim}) must be "
                             f"divisible by num_groups ({num_groups})")
        
        self.head_dim = embed_dim // num_groups  # 每个分组的维度。
        self.scale = 1 / (self.head_dim ** 0.5)  # 缩放因子，防止注意力权重过大。

        # 查询向量的线性投影层，降维至 head_dim。
        self.q_proj = nn.Linear(embed_dim, self.head_dim)
        # 键向量的1D卷积层，输入通道数与输出通道数相同，按组进行卷积。
        self.k_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=1,
                                groups=num_groups)

    def forward(self,
                x: Tensor,
                attn_mask: Optional[Tensor] = None,
                tau: float = 1.0):
        """The forward function of MSA.

        Args:
            x: The input data with shape (B, N, C).
            attn_mask: Attention mask. Defaults to None.
            tau: Temperature in Softmax. Default: 1.0
        Returns:
            Tensor: The updated feature with shape (B, N, C).
            Tensor: The attention map.
        """
        bs, n_q, _ = x.shape  # 获取 batch size 和查询数量。
        
        # 展平 batch 和查询维度，为键向量的卷积操作做准备。
        kv = x.flatten(0, 1).unsqueeze(-1)  # (B * N, C, 1)

        # 使用 1D 卷积投影键向量，得到 shape 为 (B * N, C, 1)。
        k = self.k_proj(kv)

        # 使用线性投影得到查询向量，并增加一个维度 (B, 1, N, head_dim)。
        q = self.q_proj(x).unsqueeze(1)

        # 将值向量 reshape 成组格式并调整维度顺序，形状为 (B, groups, N, head_dim)。
        v = x.view(bs, n_q, self.num_groups, -1).permute(0, 2, 1, 3)

        # 将键向量 reshape 成组格式并调整维度顺序，形状为 (B, groups, head_dim, N)。
        k = k.view(bs, n_q, self.num_groups, -1).permute(0, 2, 3, 1)

        # 计算注意力权重，点积后乘以缩放因子，形状为 (B, groups, N, N)。
        attn_weight = (q @ k) * self.scale

        # 如果提供了注意力掩码，添加到注意力权重上。
        if attn_mask is not None:
            attn_weight = attn_weight + attn_mask.view(*attn_weight.shape)

        # 应用 Softmax（带温度参数 tau）归一化注意力权重。
        attn_weight = attn_weight.div(tau).softmax(-1)

        # 使用注意力权重加权求和值向量，形状为 (B, groups, N, head_dim)。
        context = attn_weight @ v

        # 调整输出格式，合并最后两维 (B, N, C)。
        context = context.permute(0, 2, 1, 3).contiguous()
        return context.flatten(-2, -1), attn_weight

    @staticmethod
    def loss(pred_lanes: Tensor,
             target_lanes: Tensor,
             pred_attn_weight: Tensor):
        """Loss function.

        Args:
            pred_lanes: Predicted lane xs with shape (n_prior, 72).
            target_lanes: Ground-truth lane xs with shape (n_pos, 72).
            pred_attn_weight: Atten map with shape (groups, n_pos, n_prior).
        Returns:
            Tensor: Cross entropy loss of attention map.
        """
        # 如果没有目标车道线，返回损失为 0。
        if len(target_lanes) == 0:
            return 0

        # 创建目标车道线的副本并翻转 x 坐标顺序。
        target_lanes = target_lanes.detach().clone()
        target_lanes = target_lanes.flip(-1)  # (n_pos, 72)

        # 同样翻转预测车道线的 x 坐标。
        pred_lanes = pred_lanes.clone()
        pred_lanes = pred_lanes.flip(-1)

        # 重塑目标和预测车道线为分组格式。
        groups, n_pos, n_prior = pred_attn_weight.shape
        target_lanes = target_lanes.reshape(n_pos, groups, -1).permute(1, 0, 2)  # (groups, n_pos, 72//groups)
        pred_lanes = pred_lanes.reshape(n_prior, groups, -1).permute(1, 0, 2)  # (groups, n_prior, 72//groups)

        # 计算有效掩码，标识车道线的有效点。
        valid_mask = (0 <= target_lanes) & (target_lanes < 1)

        # 计算预测车道线和目标车道线之间的距离。
        dist = ((pred_lanes.unsqueeze(1) - target_lanes.unsqueeze(2)).abs()
                ) * valid_mask.unsqueeze(2)  # (groups, n_pos, n_prior, 72//groups)

        # 对每个分组的距离求平均，忽略无效点。
        dist = dist.sum(-1) / (valid_mask.sum(-1).unsqueeze(2) + 1e-6)  # (groups, n_pos, n_prior)

        # 找到与每个目标车道线最接近的预测车道线索引。
        _, indices = dist.min(-1)  # (groups, n_pos)

        # 更新掩码，忽略无效目标点。
        valid_mask = valid_mask.any(-1)  # (groups, n_pos)
        indices[~valid_mask] = 255  # 无效索引置为 255。

        # 限制注意力权重的范围，防止 log 输入非法值。
        pred_attn_weight = torch.clamp(pred_attn_weight, 1e-6, 1 - 1e-6)

        # 计算交叉熵损失，忽略无效索引。
        loss = F.nll_loss(torch.log(pred_attn_weight).transpose(1, 2),
                          indices.long(),
                          ignore_index=255)
        return loss
