import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from srlane.models.registry import HEADS
from srlane.models.losses.seg_loss import SegLoss


@HEADS.register_module
class LocalAngleHead(nn.Module):
    """
    局部角度预测头，用于预测车道线的局部角度。

    参数:
        num_points (int): 要预测的车道点数量。
        in_channel (int): 输入特征的通道数。
        cfg (dict): 模型配置字典，包含特定设置。
    """

    def __init__(self,
                 num_points: int = 72,
                 in_channel: int = 64,
                 cfg=None):
        super(LocalAngleHead, self).__init__()
        self.n_offsets = num_points  # 车道点数量
        self.cfg = cfg
        self.img_w = cfg.img_w  # 图像宽度
        self.img_h = cfg.img_h  # 图像高度
        self.aux_seg = self.cfg.get("seg_loss_weight", 0.) > 0.  # 是否启用辅助分割
        self.feat_h, self.feat_w = self.cfg.angle_map_size  # 角度映射的尺寸

        # 构建均匀分布的 y 坐标（用于先验信息）
        self.register_buffer(
            name="prior_ys",
            tensor=torch.linspace(0, self.feat_h, steps=self.n_offsets, dtype=torch.float32)
        )

        # 生成特征图的网格坐标 (grid_x, grid_y)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.feat_h - 0.5, 0, -1, dtype=torch.float32),
            torch.arange(0.5, self.feat_w, 1, dtype=torch.float32),
            indexing="ij"
        )
        grid = torch.stack((grid_x, grid_y), 0)  # 堆叠为 (2, h, w)
        grid.unsqueeze_(0)  # 增加 batch 维度，变为 (1, 2, h, w)
        self.register_buffer(name="grid", tensor=grid)

        # 定义角度预测的卷积层，数量为 FPN 的层数
        self.angle_conv = nn.ModuleList()
        for _ in range(self.cfg.n_fpn):
            self.angle_conv.append(nn.Conv2d(in_channel, 1, 1, 1, 0, bias=False))

        # 如果启用了辅助分割任务，则定义分割相关的卷积层和损失函数
        if self.aux_seg:
            num_classes = self.cfg.max_lanes + 1  # 分割的类别数（车道数量 + 背景）
            self.seg_conv = nn.ModuleList()
            for _ in range(self.cfg.n_fpn):
                self.seg_conv.append(nn.Conv2d(in_channel, num_classes, 1, 1, 0))
            self.seg_criterion = SegLoss(num_classes=num_classes)  # 分割损失函数

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化角度预测卷积层的权重。"""
        for m in self.angle_conv.parameters():
            nn.init.normal_(m, 0., 1e-3)

    def forward(self, feats: List[torch.Tensor]):
        """
        前向传播过程。

        参数:
            feats (List[Tensor]): 多级特征图列表。

        返回:
            dict: 包含车道候选信息、预测角度图和分割图（如果启用分割任务）。
        """
        theta_list = []  # 存储多级特征的角度预测结果

        # 如果处于测试模式，仅使用最深层的特征
        if not self.training:
            feats = feats[-1:]

        # 遍历特征图，逐层进行角度预测
        for i, feat in enumerate(feats, 1):
            theta = self.angle_conv[len(feats) - i](feat).sigmoid()  # 激活为 [0, 1]
            theta_list.append(theta)

        # 如果启用了辅助分割，计算分割结果
        if self.aux_seg:
            seg_list = []
            for i, feat in enumerate(feats, 1):
                seg = self.seg_conv[len(feats) - i](feat)
                seg_list.append(seg)

        # 插值到目标大小 (self.feat_h, self.feat_w)
        angle = F.interpolate(
            theta_list[-1],
            size=[self.feat_h, self.feat_w],
            mode="bilinear",
            align_corners=True
        ).squeeze(1)
        angle = angle.detach()  # 分离计算图
        angle.clamp_(min=0.05, max=0.95)  # 限制角度范围，避免过度倾斜

        # 构建车道候选区域
        k = (angle * math.pi).tan()  # 计算角度的斜率
        bs, h, w = angle.shape
        grid = self.grid
        ws = (
            (self.prior_ys.view(1, 1, self.n_offsets) - grid[:, 1].view(1, h * w, 1))
            / k.view(bs, h * w, 1)
            + grid[:, 0].view(1, h * w, 1)
        )  # (bs, h*w, n_offsets)
        ws = ws / w  # 归一化宽度
        valid_mask = (0 <= ws) & (ws < 1)  # 有效范围的掩码
        _, indices = valid_mask.max(-1)
        start_y = indices / (self.n_offsets - 1)  # 起始点的 y 坐标
        priors = ws.new_zeros(
            (bs, h * w, 2 + 2 + self.n_offsets), device=ws.device
        )
        priors[..., 2] = start_y
        priors[..., 4:] = ws

        return dict(
            priors=priors,
            pred_angle=[theta.squeeze(1) for theta in theta_list] if self.training else None,
            pred_seg=seg_list if (self.training and self.aux_seg) else None
        )

    def loss(self,
             pred_angle: List[torch.Tensor],
             pred_seg: Optional[List[torch.Tensor]],
             gt_angle: List[torch.Tensor],
             gt_seg: Optional[List[torch.Tensor]],
             loss_weight: Tuple[float] = [0.2, 0.2, 1.],
             ignore_value: float = 0.,
             **ignore_kwargs):
        """
        计算角度预测和分割的损失。

        参数:
            pred_angle (List[Tensor]): 预测的角度图。
            gt_angle (List[Tensor]): 真实角度标签。
            pred_seg (List[Tensor], 可选): 预测的分割图。
            gt_seg (List[Tensor], 可选): 真实分割标签。
            loss_weight (Tuple[float]): 每层特征图的损失权重。
            ignore_value (float): 忽略值的占位符。

        返回:
            dict: 包含角度损失和（如果启用）分割损失。
        """
        angle_loss = 0  # 初始化角度损失
        for pred, target, weight in zip(pred_angle, gt_angle, loss_weight):
            valid_mask = target > ignore_value  # 忽略指定值
            angle_loss += (
                ((pred - target).abs() * valid_mask).sum() / (valid_mask.sum() + 1e-4)
            ) * weight

        if self.aux_seg:  # 如果启用了分割任务，计算分割损失
            seg_loss = 0
            for pred, target, weight in zip(pred_seg, gt_seg, loss_weight):
                seg_loss += self.seg_criterion(pred, target) * weight
            return {"angle_loss": angle_loss, "seg_loss": seg_loss}

        return {"angle_loss": angle_loss}

    def __repr__(self):
        num_params = sum(map(lambda x: x.numel(), self.parameters()))
        return f"#Params of {self._get_name()}: {num_params / 10 ** 3:<.2f}[K]"
