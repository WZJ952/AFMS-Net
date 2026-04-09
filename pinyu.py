import torch.fft as fft
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class CrossViewDCTEnhancer(nn.Module):
    def __init__(self, low_freq_ratio=0.1, high_freq_boost=1.8, 
                 spectral_bands=3, adaptive_threshold=True):
        """
        面向无人机-卫星跨视角匹配的DCT频率增强器
        
        参数:
        - low_freq_ratio: 低频部分比例（相对于图像尺寸）
        - high_freq_boost: 高频增强系数
        - spectral_bands: 光谱波段数
        - adaptive_threshold: 是否启用自适应频率阈值
        """
        super().__init__()
        self.low_freq_ratio = low_freq_ratio
        self.adaptive_threshold = adaptive_threshold
        
        # 可学习的频域增强参数
        self.band_weights = nn.Parameter(torch.ones(spectral_bands))
        self.detail_gain = nn.Parameter(torch.tensor(high_freq_boost))
        self.structure_preserve = nn.Parameter(torch.tensor(1.2))
        
        # 多尺度频率注意力权重
        self.scale_attention = nn.Parameter(torch.tensor([0.6, 0.3, 0.1]))  # 低、中、高频注意力，原[0.6, 0.3, 0.1]
        
    def adaptive_frequency_decomposition(self, x_dct, img_size):
        """自适应频率分解 - 根据图像尺寸动态调整频带划分[1](@ref)"""
        if self.adaptive_threshold:
            # 基于图像尺寸的动态阈值[7](@ref)
            min_dim = min(img_size[-2], img_size[-3])
            k = int(min_dim * self.low_freq_ratio)
            k = max(8, min(k, min_dim // 4))  # 确保合理范围
        else:
            k = 30  # 默认阈值
        
        # 创建多频带掩码：低频(结构)、中频(轮廓)、高频(细节)[6](@ref)
        low_mask = torch.zeros_like(x_dct)
        mid_mask = torch.zeros_like(x_dct)
        high_mask = torch.zeros_like(x_dct)
        
        # 低频带 - 基础结构信息
        low_end = k
        low_mask[:, :, :low_end, :low_end] = 1
        
        # 中频带 - 物体轮廓信息
        mid_start = low_end
        mid_end = min(low_end * 2, min_dim // 2)
        mid_mask[:, :, :mid_end, :mid_end] = 1
        mid_mask[:, :, :low_end, :low_end] = 0  # 排除低频部分
        
        # 高频带 - 细节纹理信息
        high_mask = 1 - (low_mask + mid_mask)
        
        return low_mask, mid_mask, high_mask, k
    
    def cross_view_frequency_enhancement(self, x_dct, view_type='drone'):
        """跨视角频率增强 - 针对不同视角特性优化[3](@ref)"""
        B, C, H, W = x_dct.shape
        
        # 生成自适应频率掩码
        low_mask, mid_mask, high_mask, k = self.adaptive_frequency_decomposition(x_dct, x_dct.shape)
        
        # 视角自适应增强策略[2](@ref)
        if view_type == 'drone':
            # 无人机图像：增强细节和局部特征
            low_weight = 1.0    # 保持结构稳定性
            mid_weight = 1.4   # 增强轮廓特征yuan1.4，1.1，2.0不行
            high_weight = self.detail_gain  # 显著增强细节
        else:  # satellite
            # 卫星图像：保持全局结构一致性
            low_weight = self.structure_preserve  # 增强全局结构
            mid_weight = 1.1    # 适度增强轮廓
            high_weight = 0.8   # 抑制噪声和过度细节
        
        # 多频带增强处理
        low_freq = x_dct * low_mask * low_weight
        mid_freq = x_dct * mid_mask * mid_weight
        
        # 细节增强高频处理[6](@ref)
        high_freq = x_dct * high_mask
        high_energy = torch.abs(high_freq).mean(dim=1, keepdim=True)
        
        # 基于能量分布的自适应增强
        energy_weight = 1.0 + (high_weight - 1.0) * torch.sigmoid(high_energy * 3)
        enhanced_high = high_freq * energy_weight
        
        # 应用多尺度注意力权重[1](@ref)
        enhanced_dct = (low_freq * self.scale_attention[0] + 
                      mid_freq * self.scale_attention[1] + 
                      enhanced_high * self.scale_attention[2])
        
        return enhanced_dct
    
    def multi_spectral_fusion(self, x_dct):
        """多光谱频域融合 - 增强跨模态特征一致性[3](@ref)"""
        if x_dct.dim() == 4 and x_dct.shape[1] > 1:
            # 计算跨通道频域注意力
            channel_energy = torch.mean(torch.abs(x_dct), dim=(-2, -1), keepdim=True)
            channel_attn = torch.softmax(channel_energy, dim=1)
            
            # 频域特征融合
            fused_dct = x_dct + x_dct * channel_attn * 0.5   #原来0.3,0.7效果不好
            return fused_dct
        return x_dct

    def forward(self, x, view_type='drone'):
        """
        前向传播 - 针对跨视角匹配优化
        
        参数:
        - x: 输入图像 [B, C, H, W]
        - view_type: 视角类型 ('drone' 或 'satellite')
        """
        # 多光谱频域处理
        if x.dim() == 4 and x.shape[1] > 1:
            spectral_components = []
            for i in range(x.shape[1]):
                channel_dct = dct_2d(x[:, i:i+1])
                # 应用可学习的光谱权重
                weighted_dct = channel_dct * self.band_weights[i]
                spectral_components.append(weighted_dct)
            x_dct = torch.cat(spectral_components, dim=1)
        else:
            x_dct = dct_2d(x)
        
        # 跨模态频域融合
        x_dct = self.multi_spectral_fusion(x_dct)
        
        # 视角自适应频率增强
        enhanced_dct = self.cross_view_frequency_enhancement(x_dct, view_type)
        
        return idct_2d(enhanced_dct)



def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc.real * W_r - Vc.imag * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
    v = torch.fft.ifft(tmp)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape).real


def dct_2d(x, norm=None):
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)







