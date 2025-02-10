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
    """VAE解码器（将潜变量z恢复为图像）"""
    def __init__(self):
        super().__init__(
            # 初始投影层
            nn.Conv2d(4, 4, kernel_size=1),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            # 多级残差块与上采样
            *self._build_blocks(512, 512, upsample=True),  # 1/8 -> 1/4
            *self._build_blocks(512, 512, upsample=True),  # 1/4 -> 1/2
            *self._build_blocks(512, 256, upsample=True),  # 1/2 -> 1
            *self._build_blocks(256, 128, upsample=False),  # 保持分辨率
            
            # 最终输出层
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def _build_blocks(self, in_c, out_c, upsample=False):
        """构建解码器块序列"""
        blocks = []
        if upsample:
            blocks.append(nn.Upsample(scale_factor=2))
            blocks.append(nn.Conv2d(in_c, in_c, kernel_size=3, padding=1))
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