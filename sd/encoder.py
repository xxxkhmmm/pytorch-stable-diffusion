import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # 输入层：3通道RGB图像 -> 128通道特征图
            # 卷积核3x3，padding=1保持空间维度不变
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            # 第一组残差块（保持通道数）
            # 残差连接帮助梯度流动，防止深层网络退化
            VAE_ResidualBlock(128, 128),  # in_ch, out_ch
            VAE_ResidualBlock(128, 128),
            
            # 下采样层：stride=2实现空间维度减半
            # 使用3x3卷积，无padding导致尺寸缩小2倍（需要后续填充）
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # 第二组残差块（通道数翻倍：128->256）
            # 每个残差块包含两个卷积层和跳跃连接
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),
            
            # 第二次下采样（256->256通道，空间维度再减半）
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            
            # 第三组残差块（通道数翻倍：256->512）
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),
            
            # 第三次下采样（保持512通道，空间维度减至1/8）
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            
            # 深层特征处理块（保持通道数）
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            # 自注意力机制层：捕捉全局依赖关系
            # 适用于低分辨率特征图（H/8 x W/8）
            VAE_AttentionBlock(512),
            
            # 最终残差块
            VAE_ResidualBlock(512, 512),
            
            # 输出归一化和激活
            nn.GroupNorm(32, 512),  # 将通道分为32组进行归一化
            nn.SiLU(),  # Swish激活函数，比ReLU更平滑
            
            # 最终输出卷积：512通道 -> 8通道（均值和对数方差各4通道）
            # 使用3x3卷积保持空间维度，padding=1补偿卷积核尺寸
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            
            # 1x1卷积进一步调整通道（可选层，实际SD中用于对齐维度）
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        前向传播过程
        参数：
            x: 输入图像张量 [B, 3, H, W]
            noise: 随机噪声张量 [B, 4, H/8, W/8]
        返回：
            重参数化后的潜变量 [B, 4, H/8, W/8]
        """
        # 遍历所有子模块
        for module in self:
            # 处理下采样层的非对称填充
            if getattr(module, 'stride', None) == (2, 2):
                # 只在右下侧填充1像素（解决偶数尺寸问题）
                # 输入尺寸：H x W → 输出尺寸：(H+1)//2 x (W+1)//2
                x = F.pad(x, (0, 1, 0, 1))  # (左, 右, 上, 下)
            
            x = module(x)

        # 将输出拆分为均值和对数方差（各4通道）
        # 原始论文使用对角高斯分布假设
        mean, log_variance = torch.chunk(x, 2, dim=1)
        
        # 对数方差截断（保证数值稳定性）
        # 限制范围相当于方差在[1e-14, e^20]之间
        log_variance = torch.clamp(log_variance, -30, 20)
        
        # 计算标准差（避免除零错误）
        std = torch.exp(log_variance * 0.5)  # 方差=exp(log_variance)
        
        # 重参数化技巧：从标准正态分布采样
        # x = μ + σ·ε，其中ε ~ N(0, I)
        latent = mean + std * noise
        
        # 缩放因子（来自Stable Diffusion官方配置）
        # 用于将潜变量约束到接近标准正态分布
        latent *= 0.18215
        
        return latent

"""
VAE编码器关键设计解析：
1. 渐进式下采样：通过3次stride=2的卷积，将空间维度缩小到1/8
   - 输入：H x W → H/8 x W/8
   - 平衡计算效率和特征保留

2. 残差块结构：
   ┌───────────────┐
   │ GroupNorm      │
   │ SiLU激活       │
   │ 3x3卷积        │
   └─────┬───────────┘
         │
   ┌─────▼──────────┐
   │ GroupNorm      │
   │ SiLU激活       │
   │ 3x3卷积        │
   └─────┬──────────┘
         │
   ┌─────▼──────────┐
   │ 残差连接 (+)   │
   └───────────────┘

3. 注意力机制：在低分辨率特征图上应用自注意力
   - 计算全局依赖关系
   - 增强对图像整体结构的编码能力

4. 输出处理：
   - 分离均值和对数方差：实现变分推断
   - 重参数化：解耦随机性与梯度传播
   - 缩放因子：适配扩散模型需求
"""

