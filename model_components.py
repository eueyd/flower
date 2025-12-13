# model_components.py
"""
模型组件定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config import NUM_CLASSES, NUM_REGIONS


class RegionProposalNetwork(nn.Module):
    """区域建议网络"""

    def __init__(self, in_channels, hidden_dim=256, num_regions=2):
        super().__init__()
        self.num_regions = num_regions

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, num_regions, 1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        attention_maps = self.softmax(x)
        return attention_maps


class FeaturePyramidFusion(nn.Module):
    """特征金字塔融合模块"""

    def __init__(self, channels_list, output_dim=256):
        super().__init__()
        self.convs = nn.ModuleList()

        for channels in channels_list:
            self.convs.append(nn.Conv2d(channels, output_dim, 1))

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(output_dim * len(channels_list), output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, *feats):
        target_size = feats[0].shape[2:]
        processed_feats = []

        for i, (conv, feat) in enumerate(zip(self.convs, feats)):
            feat = conv(feat)
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size,
                                     mode='bilinear', align_corners=False)
            processed_feats.append(feat)

        fused = torch.cat(processed_feats, dim=1)
        fused = self.fusion_conv(fused)
        return fused


class RegionFeatureEnhancer(nn.Module):
    """区域特征增强模块"""

    def __init__(self, global_dim, region_dim, output_dim):
        super().__init__()
        self.global_proj = nn.Linear(global_dim, region_dim)
        self.region_proj = nn.Linear(region_dim, region_dim)

        self.enhance = nn.Sequential(
            nn.Linear(global_dim + region_dim, output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, global_feat, region_feat):
        global_proj = self.global_proj(global_feat)
        region_proj = self.region_proj(region_feat)

        attention = torch.sigmoid(torch.sum(global_proj * region_proj, dim=1, keepdim=True))
        enhanced = region_feat * attention

        combined = torch.cat([global_feat, enhanced], dim=1)
        output = self.enhance(combined)
        return output


class AdaptiveFusionModule(nn.Module):
    """自适应特征融合模块"""

    def __init__(self, global_dim, region_dim, num_regions, hidden_dim=256):
        super().__init__()
        self.num_regions = num_regions
        self.global_dim = global_dim
        self.region_dim = region_dim

        self.global_proj = nn.Linear(global_dim, region_dim)

        self.attention = nn.Sequential(
            nn.Linear(region_dim * (1 + num_regions), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1 + num_regions),
            nn.Softmax(dim=1)
        )

        self.fusion = nn.Sequential(
            nn.Linear(region_dim * (1 + num_regions), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 128)
        )

    def forward(self, global_feat, region_feats):
        global_feat_proj = self.global_proj(global_feat)

        all_feats = [global_feat_proj] + region_feats
        concatenated = torch.cat(all_feats, dim=1)

        weights = self.attention(concatenated)

        weighted_feats = weights[:, 0:1] * global_feat_proj
        for i in range(self.num_regions):
            weighted_feats = weighted_feats + weights[:, i + 1:i + 2] * region_feats[i]

        fused_feature = self.fusion(concatenated)
        return fused_feature, weights


class ImprovedDualPathModel(nn.Module):
    """改进的双路径模型"""

    def __init__(self, num_classes=NUM_CLASSES, num_regions=NUM_REGIONS,
                 backbone_name='resnet18'):
        super().__init__()
        self.num_regions = num_regions
        self.num_classes = num_classes

        # 骨干网络
        if backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            layer_channels = [64, 128, 256, 512]
        elif backbone_name == 'resnet34':
            backbone = models.resnet34(pretrained=True)
            layer_channels = [64, 128, 256, 512]
        elif backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            layer_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # 区域建议网络
        self.region_proposal = RegionProposalNetwork(
            in_channels=layer_channels[1],
            hidden_dim=128,
            num_regions=num_regions
        )

        # 多尺度特征融合
        self.feature_fusion = FeaturePyramidFusion(
            channels_list=layer_channels[1:],
            output_dim=128
        )

        # 区域特征增强
        self.region_feature_enhancer = nn.ModuleList([
            RegionFeatureEnhancer(
                global_dim=layer_channels[-1],
                region_dim=128,
                output_dim=64
            ) for _ in range(num_regions)
        ])

        # 自适应融合模块
        self.adaptive_fusion = AdaptiveFusionModule(
            global_dim=layer_channels[-1],
            region_dim=64,
            num_regions=num_regions,
            hidden_dim=128
        )

        # 分类器
        self.global_classifier = nn.Sequential(
            nn.Linear(layer_channels[-1], 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        self.final_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

        # 区域分类器（辅助监督）
        self.region_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, num_classes)
            ) for _ in range(num_regions)
        ])

        # 初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, x):
        """提取多尺度特征"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        return feat1, feat2, feat3, feat4

    def forward(self, x, return_attention=False):
        batch_size = x.size(0)

        # 特征提取
        feat1, feat2, feat3, feat4 = self.extract_features(x)

        # 全局特征
        global_pool = nn.AdaptiveAvgPool2d(1)
        global_feat = global_pool(feat4).view(batch_size, -1)

        # 区域建议
        attention_maps = self.region_proposal(feat2)

        attention_maps_full = F.interpolate(
            attention_maps, size=x.shape[2:],
            mode='bilinear', align_corners=False
        )

        # 多尺度特征融合
        fused_feat = self.feature_fusion(feat2, feat3, feat4)

        # 区域特征提取与增强
        region_features = []
        region_logits = []

        for i in range(self.num_regions):
            region_mask = F.interpolate(
                attention_maps[:, i:i + 1],
                size=fused_feat.shape[2:],
                mode='bilinear', align_corners=False
            )

            region_feat_map = fused_feat * region_mask
            region_pool = nn.AdaptiveAvgPool2d(1)
            region_feat = region_pool(region_feat_map).view(batch_size, -1)

            enhanced_region_feat = self.region_feature_enhancer[i](global_feat, region_feat)
            region_features.append(enhanced_region_feat)

            region_logit = self.region_classifiers[i](enhanced_region_feat)
            region_logits.append(region_logit)

        # 自适应特征融合
        final_feature, fusion_weights = self.adaptive_fusion(global_feat, region_features)

        # 分类输出
        global_logits = self.global_classifier(global_feat)
        final_logits = self.final_classifier(final_feature)

        outputs = {
            'final_logits': final_logits,
            'global_logits': global_logits,
            'region_logits': torch.stack(region_logits, dim=1) if region_logits else None,
            'fused_feature': final_feature,
            'fusion_weights': fusion_weights,
            'attention_maps': attention_maps_full,
            'global_feature': global_feat
        }

        if return_attention:
            return outputs, attention_maps_full
        return outputs

    def compute_loss(self, outputs, labels, criterion):
        """计算多任务损失"""
        from config import LAMBDA_GLOBAL, LAMBDA_REGION

        loss_final = criterion(outputs['final_logits'], labels)
        loss_global = criterion(outputs['global_logits'], labels)

        # 区域分类损失
        loss_region = 0
        if outputs['region_logits'] is not None:
            for i in range(self.num_regions):
                loss_region += criterion(outputs['region_logits'][:, i], labels)
            loss_region /= self.num_regions

        total_loss = (loss_final +
                      LAMBDA_GLOBAL * loss_global +
                      LAMBDA_REGION * loss_region)

        loss_details = {
            'loss_final': loss_final.item(),
            'loss_global': loss_global.item(),
            'loss_region': loss_region.item() if outputs['region_logits'] is not None else 0,
            'total_loss': total_loss.item()
        }

        return total_loss, loss_details