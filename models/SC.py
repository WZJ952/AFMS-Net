import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vit_pytorch import vit_small_patch16_224_backbone
from timm.models.swin_transformer import swin_tiny_patch4_window7_224
from torchvision.ops import DeformConv2d



class MultiScaleObjectContextSIM(nn.Module):
    def __init__(self, norm_nc, label_nc, nhidden=64, reduction=4):
        super().__init__()
        self.norm_nc = norm_nc
        self.label_nc = label_nc
        
        # 多尺度语义编码器
        self.semantic_encoder = nn.Sequential(
            nn.Conv2d(label_nc, nhidden//4, 3, padding=1),
            nn.GroupNorm(4, nhidden//4),
            nn.ReLU(True),
            nn.Conv2d(nhidden//4, nhidden//2, 3, padding=1, stride=2),
            nn.GroupNorm(4, nhidden//2),
            nn.ReLU(True),
            nn.Conv2d(nhidden//2, nhidden, 3, padding=1, stride=2),
            nn.GroupNorm(8, nhidden),
            nn.ReLU(True)
        )
        
        # 对象上下文注意力机制[8](@ref)
        self.object_context = ObjectContextAttention(nhidden, norm_nc, reduction)
        
        # 多尺度融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(nhidden * 3, nhidden, 1),
            nn.BatchNorm2d(nhidden),
            nn.ReLU(True)
        )
        
        # 参数生成（保持与原始SIM兼容）
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        
        self.gamma_net = nn.Sequential(
            nn.Conv2d(nhidden, norm_nc, 3, padding=1),
            nn.Sigmoid()
        )
        self.beta_net = nn.Conv2d(nhidden, norm_nc, 3, padding=1)
        
    def forward(self, x, segmap):
        # 多尺度语义特征提取
        B, C, H, W = x.size()
        
        # 调整语义图尺寸匹配输入
        segmap = F.interpolate(segmap, size=(H, W), mode='bilinear')
        
        # 提取多尺度语义特征
        semantic_feat = self.semantic_encoder(segmap)
        
        # 应用对象上下文注意力[8](@ref)
        context_feat = self.object_context(semantic_feat)
        
        # 多尺度特征上采样和融合
        feat_1x = F.interpolate(context_feat, size=(H, W), mode='bilinear')
        feat_2x = F.interpolate(context_feat, size=(H//2, W//2), mode='bilinear')
        feat_4x = F.interpolate(context_feat, size=(H//4, W//4), mode='bilinear')
        
        feat_2x = F.interpolate(feat_2x, size=(H, W), mode='bilinear')
        feat_4x = F.interpolate(feat_4x, size=(H, W), mode='bilinear')
        
        fused_feat = torch.cat([feat_1x, feat_2x, feat_4x], dim=1)
        fused_feat = self.fusion_conv(fused_feat)
        
        # 生成调制参数
        gamma = self.gamma_net(fused_feat)
        beta = self.beta_net(fused_feat)
        
        # 应用调制
        normalized = self.param_free_norm(x)
        out = normalized * (1 + gamma) + beta
        
        return out

class ObjectContextAttention(nn.Module):
    """基于OCRNet的对象上下文注意力机制[8](@ref)"""
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.key_conv = nn.Conv2d(in_channels, in_channels//reduction, 1)
        self.query_conv = nn.Conv2d(in_channels, in_channels//reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        # 计算key, query, value
        key = self.key_conv(x).view(B, -1, H*W)
        query = self.query_conv(x).view(B, -1, H*W)
        value = self.value_conv(x).view(B, -1, H*W)
        
        # 计算注意力权重
        energy = torch.bmm(query.transpose(1, 2), key)
        attention = self.softmax(energy)
        
        # 应用注意力
        out = torch.bmm(value, attention.transpose(1, 2))
        out = out.view(B, -1, H, W)
        
        return out + x  # 残差连接

### 语义分割头（用于生成语义图）
class SemanticSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SemanticSegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.classifier(x)
        return F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)#6好

