"""
VAE解码器实现（将潜变量解码为图像）
关键结构：
1. 残差块（保持特征稳定性）
2. 注意力块（捕捉全局依赖）
3. 上采样层（逐步恢复分辨率）
"""

import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    """空间注意力块（用于低分辨率特征图）"""
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)  # 单头注意力
    
    def forward(self, x):
        # 输入形状：[B, C, H, W]
        residue = x
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        
        # 空间注意力（将HxW视为序列长度）
        x = x.view(n, c, h*w).transpose(-1, -2)  # [B, H*W, C]
        x = self.attention(x)  # 无掩码
        x = x.transpose(-1, -2).view(n, c, h, w)
        return x + residue

class VAE_ResidualBlock(nn.Module):
    """残差块（包含两个卷积层和跳跃连接）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 第一个卷积块
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # 第二个卷积块
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # 跳跃连接（通道数变化时使用1x1卷积）
        self.residual_layer = nn.Identity() if in_channels == out_channels else \
                             nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residue = x
        x = F.silu(self.groupnorm_1(x))  # Swish激活
        x = self.conv_1(x)
        x = F.silu(self.groupnorm_2(x))
        x = self.conv_2(x)
        return x + self.residual_layer(residue)

class VAE_Decoder(nn.Sequential):
    """VAE解码器（将潜变量z恢复为图像）
    
    设计要点：
    1. 渐进式上采样：通过3次2倍上采样，将空间维度从H/8恢复到H
    2. 残差块堆叠：每个分辨率阶段使用3个残差块增强特征表达能力
    3. 注意力机制：在高分辨率阶段引入自注意力捕捉全局依赖
    
    面试考点：
    Q: 为什么每个卷积后都使用GroupNorm+SiLU组合？
    A: (1) GroupNorm对小批量数据更稳定，适合生成任务；(2) SiLU（Swish）激活函数
       在深度网络中表现优于ReLU，且更平滑有利于梯度流动
    
    Q: *self._build_blocks()的语法含义？
    A: 这是Python的迭代器解包语法，将列表展开为位置参数。等效于：
       self.add_module(block1), self.add_module(block2), ...
       优势：动态生成模块列表，提高代码可维护性
    """
    def __init__(self):
        super().__init__(
            # 初始投影：将4通道潜变量映射到512通道
            # 使用1x1卷积调整通道数而不改变空间维度
            nn.Conv2d(4, 4, kernel_size=1),  # [B,4,H/8,W/8] -> [B,4,H/8,W/8]
            nn.Conv2d(4, 512, kernel_size=3, padding=1),  # 增加通道数
            
            # 解码器主体：通过多个上采样阶段逐步恢复分辨率
            # 阶段1：1/8 -> 1/4 (H/8 → H/4)
            *self._build_blocks(512, 512, upsample=True),  # 解包模块列表
            
            # 阶段2：1/4 -> 1/2 (H/4 → H/2)
            *self._build_blocks(512, 512, upsample=True),
            
            # 阶段3：1/2 -> 1 (H/2 → H)
            *self._build_blocks(512, 256, upsample=True),
            
            # 最终处理阶段：保持分辨率，通道数降到128
            *self._build_blocks(256, 128, upsample=False),
            
            # 输出层：128通道 -> 3通道RGB
            nn.GroupNorm(32, 128),
            nn.SiLU(),  # 最终激活保证输出在(0, ∞)
            nn.Conv2d(128, 3, kernel_size=3, padding=1)  # 无激活，输出线性值
        )

    def _build_blocks(self, in_c, out_c, upsample=False):
        """构建解码阶段的模块序列
        参数：
            upsample: 是否在该阶段进行2倍上采样
        设计逻辑：
            (1) 上采样使用最近邻插值+3x3卷积，避免棋盘伪影
            (2) 每个阶段包含3个残差块，增强特征转换能力
            (3) 通道数变化仅在阶段过渡时发生
        """
        blocks = []
        if upsample:
            # 上采样层：最近邻插值保留高频信息，配合3x3卷积平滑特征
            blocks += [
                nn.Upsample(scale_factor=2, mode='nearest'),  # 2倍上采样
                nn.Conv2d(in_c, in_c, kernel_size=3, padding=1)  # 平滑上采样结果
            ]
        # 残差块堆叠：3个连续残差块增强非线性表达能力
        blocks += [
            VAE_ResidualBlock(in_c, out_c),
            VAE_ResidualBlock(out_c, out_c),
            VAE_ResidualBlock(out_c, out_c)
        ]
        return blocks

    def forward(self, x):
        # 输入潜变量（已标准化）
        x /= 0.18215
        for module in self:
            x = module(x)
        return x  # [B, 3, H, W]