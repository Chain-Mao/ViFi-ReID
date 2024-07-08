# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
from fastreid.modeling.backbones.bert import BertConfig, BertModel
from fastreid.modeling.backbones.vivit import ViViT
from fastreid.modeling.backbones.feature_fusion import FeatureFusion
from fastreid.modeling.losses.supcontrast import supcon_loss

@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            heads,
            pixel_mean,
            pixel_std,
            loss_kwargs=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        # backbone
        # self.backbone = backbone # 不用
        self.vision_backbone = ViViT(256, 128, 16, 16, 25)
        self.wifi_config = BertConfig(csi_size=936, hidden_size=768, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024)
        self.wifi_backbone = BertModel(config=self.wifi_config)
        self.feature_fusion = FeatureFusion(num_layers=3)
        # self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        # head
        self.heads = heads
        # self.icm_heads = nn.Linear(768, 2)

        self.loss_kwargs = loss_kwargs

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)
        return {
            'backbone': backbone,
            'heads': heads,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    }
                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, retrival=False):
        visions = self.preprocess_image(batched_inputs)
        vision_features = self.vision_backbone(visions)
        wifi_features = self.wifi_backbone(self.scale_tensor(batched_inputs['wifis']))
        
        # 特征融合 positive正例
        features = self.feature_fusion(vision_features, wifi_features).unsqueeze(2).unsqueeze(3)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets)

            # # 难例挖掘
            hard_negative_vision_to_wifi, hard_negative_wifi_to_vision = self.hard_negative_mining(vision_features, wifi_features, targets)
            icm_labels = torch.cat([torch.ones(int(targets.size(0)), dtype=torch.long), torch.zeros(2*int(targets.size(0)), dtype=torch.long)], dim=0).to('cuda')
            icm_vision = torch.cat((vision_features, vision_features, hard_negative_wifi_to_vision), dim=0)
            icm_wifi = torch.cat((wifi_features, hard_negative_vision_to_wifi, wifi_features), dim=0)
            icm_loss = self.icm_loss(icm_vision, icm_wifi, icm_labels)

            # 添加对比学习损失函数
            # Supervised对比学习
            sup_loss = (
                supcon_loss(vision_features, wifi_features, targets, targets) +
                supcon_loss(wifi_features, vision_features, targets, targets)
            ) / 2
            
            losses = self.losses(outputs, targets)
            losses['icm_loss'] = icm_loss/10
            losses['sup_loss'] = sup_loss/2
            return losses
        else:
            outputs = self.heads(features)
            if retrival:
                selected_features = torch.zeros_like(vision_features)
                for i in range(len(batched_inputs["camids"])):
                    if batched_inputs["camids"][i] == 2:    # 1 query 2 gallery
                        selected_features[i] = wifi_features[i]
                    elif batched_inputs["camids"][i] == 1:
                        selected_features[i] = vision_features[i]
                return selected_features
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            visions = batched_inputs['visions']
        # elif isinstance(batched_inputs, torch.Tensor):
        #     images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        visions.sub_(self.pixel_mean).div_(self.pixel_std)
        return visions

    def losses(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet'] = triplet_loss(
                pred_features,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')

        return loss_dict

    def hard_negative_mining(self, vision_features, wifi_features, targets):
        # 计算视觉特征和WiFi特征之间的余弦相似度
        similarity_matrix = torch.nn.functional.cosine_similarity(vision_features.unsqueeze(1), wifi_features.unsqueeze(0), dim=2)

        # 初始化难例特征值的索引
        hard_negative_vision_to_wifi_indices = []
        hard_negative_wifi_to_vision_indices = []

        for idx in range(similarity_matrix.size(0)):
            # 找出vision对应的wifi负样本
            mask = targets[idx] != targets
            vision_to_wifi_similarities = similarity_matrix[idx]
            vision_to_wifi_similarities[~mask] = -1  # 忽略正样本
            hard_wifi_idx = vision_to_wifi_similarities.argmax()
            hard_negative_vision_to_wifi_indices.append(hard_wifi_idx)

            # 找出wifi对应的vision负样本
            wifi_to_vision_similarities = similarity_matrix[:, idx]
            wifi_to_vision_similarities[~mask] = -1  # 忽略正样本
            hard_vision_idx = wifi_to_vision_similarities.argmax()
            hard_negative_wifi_to_vision_indices.append(hard_vision_idx)

        # 将索引列表转换为Tensor，并使用这些索引提取难例特征
        hard_negative_vision_to_wifi = wifi_features[torch.tensor(hard_negative_vision_to_wifi_indices)]
        hard_negative_wifi_to_vision = vision_features[torch.tensor(hard_negative_wifi_to_vision_indices)]

        return hard_negative_vision_to_wifi, hard_negative_wifi_to_vision

    # 示例用法
    # vision_feature = torch.randn(batchsize, vision_feature_size)
    # wifi_feature = torch.randn(batchsize, wifi_feature_size)
    # targets = torch.tensor([类别ID列表])
    # hard_vision, hard_wifi = hard_example_mining(vision_feature, wifi_feature, targets)

    def icm_loss(self, video_features, wifi_features, labels):
        """
        Calculate the custom loss based on cosine distance for matched/mismatched video and wifi signals.

        :param video_features: Tensor of video features, shape (batchsize, feature_length)
        :param wifi_features: Tensor of wifi signal features, shape (batchsize, feature_length)
        :param labels: Tensor of labels (0 or 1), shape (batchsize,)
        :return: Loss value as a tensor
        """
        # Calculate cosine similarity (cosine distance is 1 - cosine similarity)
        cos_sim = nn.functional.cosine_similarity(video_features, wifi_features) + 1 # 范围（0,2）

        # Loss for matched pairs (labels == 1) should increase as cosine similarity decreases
        # Loss for mismatched pairs (labels == 0) should increase as cosine similarity increases
        loss = torch.sum(labels * (2 - cos_sim) + (1 - labels) * cos_sim)

        return loss/(labels.shape[0]/3)

    def scale_tensor(self, x):
        # 找到Tensor的最小值和最大值
        min_val = torch.min(x)
        max_val = torch.max(x)
        
        # 将Tensor的数据范围归一化到0至1
        x_normalized = (x - min_val) / (max_val - min_val)
        
        # 将范围调整到-1至1
        x_scaled = x_normalized * 2 - 1
        
        return x_scaled