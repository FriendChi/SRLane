from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .multi_segment_attention import MultiSegmentAttention
from srlane.ops import nms
from srlane.utils.lane import Lane
from srlane.models.losses.focal_loss import FocalLoss
from srlane.models.utils.dynamic_assign import assign
from srlane.models.utils.a3d_sample import sampling_3d
from srlane.models.losses.lineiou_loss import liou_loss
from srlane.models.registry import HEADS


class RefineHead(nn.Module):
    """Refine head.

    Args:
        stage: Refinement stage index.
        num_points: Number of points to describe a lane.
        prior_feat_channels: Input channel.
        in_channel: Input channels.
        fc_hidden_dim: Hidden channels.
        refine_layers: Total number of refinement stage.
        sample_points: Number of points for sampling lane feature.
        num_groups: Number of lane segment groups.
        cfg: Model config.
    """

    def __init__(self,
                 stage: int,
                 num_points: int,
                 prior_feat_channels: int,
                 fc_hidden_dim: int,
                 refine_layers: int,
                 sample_points: int,
                 num_groups: int,
                 cfg=None):
        super(RefineHead, self).__init__()
        self.stage = stage
        self.cfg = cfg
        self.img_w = self.cfg.img_w
        self.img_h = self.cfg.img_h
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.sample_points = sample_points
        self.fc_hidden_dim = fc_hidden_dim
        self.num_groups = num_groups
        self.num_level = cfg.n_fpn
        self.last_stage = stage == refine_layers - 1

        self.register_buffer(name="sample_x_indexs", tensor=(
                torch.linspace(0, 1,
                               steps=self.sample_points,
                               dtype=torch.float32) * self.n_strips).long())
        self.register_buffer(name="prior_feat_ys", tensor=torch.flip(
            (1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]))

        self.prior_feat_channels = prior_feat_channels
        self.z_embeddings = nn.Parameter(torch.zeros(self.sample_points),
                                         requires_grad=True)

        self.gather_fc = nn.Conv1d(sample_points, fc_hidden_dim,
                                   kernel_size=prior_feat_channels,
                                   groups=self.num_groups)
        self.shuffle_fc = nn.Linear(fc_hidden_dim, fc_hidden_dim)
        self.channel_fc = nn.ModuleList()
        self.segment_attn = nn.ModuleList()
        for i in range(1):
            self.segment_attn.append(
                MultiSegmentAttention(fc_hidden_dim, num_groups=num_groups))
            self.channel_fc.append(
                nn.Sequential(nn.Linear(fc_hidden_dim, 2 * fc_hidden_dim),
                              nn.ReLU(),
                              nn.Linear(2 * fc_hidden_dim, fc_hidden_dim)))
        reg_modules = list()
        cls_modules = list()
        for _ in range(1):
            reg_modules += [nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                            nn.ReLU()]
            cls_modules += [nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                            nn.ReLU()]

        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)
        self.reg_layers = nn.Linear(
            fc_hidden_dim,
            self.n_offsets + 1 + 1)
        self.cls_layers = nn.Linear(fc_hidden_dim, 2)
        self.init_weights()

    def init_weights(self):
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)
        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)
        nn.init.normal_(self.z_embeddings, mean=self.cfg.z_mean[self.stage],
                        std=self.cfg.z_std[self.stage])


    def translate_to_linear_weight(self,
                                   ref: Tensor,
                                   num_total: int = 3,
                                   tau: int = 2.0):
        grid = torch.arange(num_total, device=ref.device,
                            dtype=ref.dtype).view(
            *[len(ref.shape) * [1, ] + [-1, ]])
        ref = ref.unsqueeze(-1).clone()
        l2 = (ref - grid).pow(2.0).div(tau).abs().neg()
        weight = torch.softmax(l2, dim=-1)

        return weight  # (1, 36, 3)

    def pool_prior_features(self,
                            batch_features: List[Tensor],
                            num_priors: int,
                            prior_feat_xs: Tensor, ):
        """Pool prior feature from feature map.
        Args:
            batch_features: Input feature maps.
        """
        batch_size = batch_features[0].shape[0]

        prior_feat_xs = prior_feat_xs.view(batch_size, num_priors, -1, 1)
        prior_feat_ys = self.prior_feat_ys.unsqueeze(0).expand(
            batch_size * num_priors,
            self.sample_points).view(
            batch_size, num_priors, -1, 1)

        grid = torch.cat((prior_feat_xs, prior_feat_ys), dim=-1)
        if self.training or not hasattr(self, "z_weight"):
            z_weight = self.translate_to_linear_weight(self.z_embeddings)
            z_weight = z_weight.view(1, 1, self.sample_points, -1).expand(
                batch_size,
                num_priors,
                self.sample_points,
                self.num_level)
        else:
            z_weight = self.z_weight.view(1, 1, self.sample_points, -1).expand(
                batch_size,
                num_priors,
                self.sample_points,
                self.num_level)

        feature = sampling_3d(grid, z_weight,
                              batch_features)  # (b, n_prior, n_point, c)
        feature = feature.view(batch_size * num_priors, -1,
                               self.prior_feat_channels)
        feature = self.gather_fc(feature).reshape(batch_size, num_priors, -1)
        for i in range(1):
            res_feature, attn = self.segment_attn[i](feature, attn_mask=None)
            feature = feature + self.channel_fc[i](res_feature)
        return feature, attn

    def forward(self, batch_features, priors, pre_feature=None):
        batch_size = batch_features[-1].shape[0]
        num_priors = priors.shape[1]
        prior_feat_xs = (priors[..., 4 + self.sample_x_indexs]).flip(
            dims=[2])  # top to bottom

        batch_prior_features, attn = self.pool_prior_features(
            batch_features, num_priors, prior_feat_xs)

        fc_features = batch_prior_features
        fc_features = fc_features.reshape(batch_size * num_priors,
                                          self.fc_hidden_dim)

        if pre_feature is not None:
            fc_features = fc_features + pre_feature.view(*fc_features.shape)

        cls_features = fc_features
        reg_features = fc_features
        predictions = priors.clone()
        if self.training or self.last_stage:
            for cls_layer in self.cls_modules:
                cls_features = cls_layer(cls_features)
            cls_logits = self.cls_layers(cls_features)
            cls_logits = cls_logits.reshape(
                batch_size, -1, cls_logits.shape[1])  # (B, num_priors, 2)
            predictions[:, :, :2] = cls_logits
        for reg_layer in self.reg_modules:
            reg_features = reg_layer(reg_features)
        reg = self.reg_layers(reg_features)
        reg = reg.reshape(batch_size, -1, reg.shape[1])

        #  predictions[:, :, 2] += reg[:, :, 0]
        # predictions[:, :, 3] = reg[:, :, 1]
        # predictions[..., 4:] += reg[..., 2:]
        predictions[:, :, 2:] += reg

        return predictions, fc_features, attn


@HEADS.register_module
class CascadeRefineHead(nn.Module):
    def __init__(self,
                 num_points: int = 72,  # 每条车道线的点数量
                 prior_feat_channels: int = 64,  # 前一层特征的通道数
                 fc_hidden_dim: int = 64,  # 全连接层隐藏单元数
                 refine_layers: int = 1,  # 精炼的层数
                 sample_points: int = 36,  # 每次采样点的数量
                 num_groups: int = 6,  # 分组的数量
                 cfg=None):  # 配置对象
        super(CascadeRefineHead, self).__init__()
        self.cfg = cfg
        self.img_w = self.cfg.img_w  # 图像宽度
        self.img_h = self.cfg.img_h  # 图像高度
        self.n_strips = num_points - 1  # 点之间的间隔数
        self.n_offsets = num_points  # 偏移点的数量
        self.sample_points = sample_points  # 每次采样点的数量
        self.refine_layers = refine_layers  # 精炼层数
        self.fc_hidden_dim = fc_hidden_dim  # 全连接层的隐藏维度
        self.num_groups = num_groups  # 分组数
        self.prior_feat_channels = prior_feat_channels  # 前一层通道数

        # 注册缓冲区，生成均匀分布的 y 坐标点
        self.register_buffer(name="prior_ys",
                             tensor=torch.linspace(1, 0, steps=self.n_offsets,
                                                   dtype=torch.float32))

        # 创建多个精炼层，每层使用 `RefineHead`
        self.stage_heads = nn.ModuleList()
        for i in range(refine_layers):
            self.stage_heads.append(
                RefineHead(stage=i,
                           num_points=num_points,
                           prior_feat_channels=prior_feat_channels,
                           fc_hidden_dim=fc_hidden_dim,
                           refine_layers=refine_layers,
                           sample_points=sample_points,
                           num_groups=num_groups,
                           cfg=cfg))

        # 定义分类损失函数，使用 FocalLoss
        self.cls_criterion = FocalLoss(alpha=0.25, gamma=2.)


      def forward(self, x, **kwargs):
        batch_features = list(x)  # 获取输入的特征
        batch_features.reverse()  # 逆序特征以便从高到低分辨率处理
        priors = kwargs["priors"]  # 提取先验信息
        pre_feature = None  # 存储前一层输出的特征
        predictions_lists = []  # 用于保存每层的预测结果
        attn_lists = []  # 用于保存注意力特征

        # 逐层精炼
        for stage in range(self.refine_layers):
            # 调用对应的 `RefineHead` 层
            predictions, pre_feature, attn = self.stage_heads[stage](
                batch_features, priors,
                pre_feature)
            predictions_lists.append(predictions)  # 记录预测结果
            attn_lists.append(attn)  # 记录注意力

            # 除了最后一层外，将预测作为下一层的先验
            if stage != self.refine_layers - 1:
                priors = predictions.clone().detach()

        # 如果是训练阶段，返回多层预测和注意力信息
        if self.training:
            output = {"predictions_lists": predictions_lists,
                      "attn_lists": attn_lists}
            return output

        # 测试阶段仅返回最后一层的预测
        return predictions_lists[-1]


    def loss(self,
             output,
             batch):
        predictions_lists = output["predictions_lists"]  # 获取各层预测结果
        attn_lists = output["attn_lists"]  # 获取各层注意力
        targets = batch["gt_lane"].clone()  # 获取目标车道线信息

        cls_loss = 0  # 分类损失
        l1_loss = 0  # 平滑 L1 损失
        iou_loss = 0  # IoU 损失
        attn_loss = 0  # 注意力损失

        # 遍历每一层的预测结果
        for stage in range(0, self.refine_layers):
            predictions_list = predictions_lists[stage]
            attn_list = attn_lists[stage]
            for idx, (predictions, target, attn) in enumerate(
                    zip(predictions_list, targets, attn_list)):
                # 过滤目标值
                target = target[target[:, 1] == 1]
                if len(target) == 0:
                    # 如果没有有效目标，仅计算分类损失
                    cls_target = predictions.new_zeros(
                        predictions.shape[0]).long()
                    cls_pred = predictions[:, :2]
                    cls_loss = cls_loss + self.cls_criterion(
                        cls_pred, cls_target).sum()
                    continue

                # 标准化预测值
                predictions = torch.cat((predictions[:, :2],
                                         predictions[:, 2:4] * self.n_strips,
                                         predictions[:, 4:] * self.img_w),
                                        dim=1)

                # 分配匹配目标
                with torch.no_grad():
                    (matched_row_inds, matched_col_inds) = assign(
                        predictions, target, self.img_w,
                        k=self.cfg.angle_map_size[0])

                # 计算注意力损失
                attn_loss += MultiSegmentAttention.loss(
                    predictions[:, 4:] / self.img_w,
                    target[matched_col_inds, 4:] / self.img_w,
                    attn[:, matched_row_inds])

                # 分类目标值
                cls_target = predictions.new_zeros(predictions.shape[0]).long()
                cls_target[matched_row_inds] = 1
                cls_pred = predictions[:, :2]

                # 回归目标值
                reg_yl = predictions[matched_row_inds, 2:4]
                target_yl = target[matched_col_inds, 2:4].clone()
                with torch.no_grad():
                    reg_start_y = torch.clamp(
                        (reg_yl[:, 0]).round().long(), 0,
                        self.n_strips)
                    target_start_y = target_yl[:, 0].round().long()
                    target_yl[:, 1] -= reg_start_y - target_start_y

                reg_pred = predictions[matched_row_inds, 4:]
                reg_targets = target[matched_col_inds, 4:].clone()

                # 分类损失
                cls_loss = cls_loss + self.cls_criterion(cls_pred, cls_target).sum(
                ) / target.shape[0]

                # 平滑 L1 损失
                l1_loss = l1_loss + F.smooth_l1_loss(reg_yl, target_yl,
                                                       reduction="mean")

                # IoU 损失
                iou_loss = iou_loss + liou_loss(reg_pred, reg_targets,
                                                self.img_w)

        # 损失平均化
        cls_loss /= (len(targets) * self.refine_layers)
        l1_loss /= (len(targets) * self.refine_layers)
        iou_loss /= (len(targets) * self.refine_layers)
        attn_loss /= (len(targets) * self.refine_layers)

        # 返回各类损失
        return_value = {"cls_loss": cls_loss,
                        "l1_loss": l1_loss,
                        "iou_loss": iou_loss,
                        "attn_loss": attn_loss}

        return return_value


def predictions_to_pred(self, predictions):
    """
    将网络预测值转换为实际车道线的坐标点。

    :param predictions: 网络的输出预测，包含分类信息和车道线点坐标
    :return: 车道线点坐标列表
    """
    pred = []  # 用于存储每条车道线的预测结果
    for prediction in predictions:
        # 选择车道线点的有效预测概率最高的类别（背景/车道线）
        lane_cls = prediction[:, :2].argmax(1)

        # 获取当前预测中属于车道线的点索引
        lane_inds = torch.where(lane_cls == 1)[0]

        if len(lane_inds) == 0:
            # 如果没有预测为车道线的点，继续下一条预测
            pred.append(None)
            continue

        # 提取车道线的偏移量
        lanes = prediction[lane_inds, 4:]
        # 提取车道线起始 y 坐标及其范围
        start_ys = prediction[lane_inds, 2]
        delta_ys = prediction[lane_inds, 3]

        # 根据 y 坐标偏移计算实际的 y 值
        lane_ys = torch.arange(0, self.n_strips + 1).to(start_ys) - start_ys[:, None]
        valid = (lane_ys >= 0) & (lane_ys < delta_ys[:, None])  # 保证 y 值在合法范围内
        lane_ys = lane_ys * valid

        # 利用偏移量计算车道线的实际 x 值
        lane_xs = torch.einsum("ij,j->ij", lane_ys, lanes)

        # 拼接 y 和 x 的坐标对
        lane = torch.cat((lane_xs[:, :, None], lane_ys[:, :, None]), dim=2)

        # 将无效值替换为 -2，表示无效点
        lane = torch.where(valid[:, :, None], lane, torch.full_like(lane, -2))

        # 转换为 numpy 格式，并添加到结果列表
        pred.append(lane.cpu().numpy())

    return pred


def get_lanes(self, predictions):
    """
    从网络的预测值中提取实际的车道线表示形式（点集合）。

    :param predictions: 网络输出的预测值
    :return: 车道线的点集合，每条车道线包含其对应的点
    """
    out_lanes = []  # 用于存储提取后的车道线
    out_categories = []  # 用于存储每条车道线的类别

    for prediction in predictions:
        if prediction is None:
            # 如果当前预测为空，继续下一个
            continue

        lanes = []  # 存储当前预测中的每条车道线的点集合
        categories = []  # 存储当前预测的每条车道线的类别

        for lane in prediction:
            # 筛选出有效点的索引（x 和 y 坐标均非 -2）
            valid_points = (lane[:, 0] != -2) & (lane[:, 1] != -2)
            if not valid_points.any():
                # 如果没有有效点，则忽略该车道线
                continue

            # 提取有效点的 x 和 y 坐标
            lane_points = lane[valid_points]
            lanes.append(lane_points)  # 添加到当前车道线列表

            # 这里假设分类信息需要进一步解析
            # categories.append(category) (可以根据需要调整)

        # 将当前预测的车道线和类别添加到最终输出
        out_lanes.append(lanes)
        out_categories.append(categories)

    return out_lanes, out_categories


    def __repr__(self):
        num_params = sum(map(lambda x: x.numel(), self.parameters()))
        return f"#Params of {self._get_name()}: {num_params / 10 ** 3:<.2f}[K]"
