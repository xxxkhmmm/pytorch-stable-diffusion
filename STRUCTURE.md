# Stable Diffusion 代码结构解析

## 核心模块与代码对应

### 1. 变分自编码器 (VAE)
- **文件**：`SD.py` 中的 `StableDiffusionVAE` 类
- **功能**：将图像编码到潜空间（Latent Space）及解码还原
- **关键代码**：
  ```python
  class StableDiffusionVAE(nn.Module):
      def __init__(self, in_channels=3, latent_channels=4, base_channels=128):
          # 编码器
          self.encoder = nn.Sequential(
              nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
              ResBlock(base_channels), 
              nn.Conv2d(base_channels, 2*base_channels, 3, stride=2, padding=1),
              ResBlock(2*base_channels),
              # ... 更多下采样层
          )
          
          # 解码器
          self.decoder = nn.Sequential(
              ResBlock(4*base_channels),
              nn.ConvTranspose2d(4*base_channels, 2*base_channels, 3, stride=2, padding=1),
              ResBlock(2*base_channels),
              # ... 更多上采样层
          )
  ```

### 2. 扩散模型核心 (UNet)
- **文件**：`SD.py` 中的 `UNetModel` 类
- **结构特点**：包含下采样和上采样路径，集成时间嵌入与注意力机制
- **代码分解**：
  ```python
  class UNetModel(nn.Module):
      def __init__(self, in_channels, out_channels, base_channels=128, 
                  time_emb_dim=512, context_dim=768):
          # 时间嵌入处理
          self.time_embedding = TimeEmbedding(time_emb_dim)
          
          # 下采样路径
          self.down_block1 = nn.ModuleList([
              ResBlock(in_channels, base_channels, time_emb_dim),
              SpatialTransformer(base_channels, context_dim)
          ])
          
          # 上采样路径 
          self.up_block3 = nn.ModuleList([
              ResBlock(4*base_channels, 2*base_channels, time_emb_dim),
              SpatialTransformer(2*base_channels, context_dim)
          ])
          
          # 输出层
          self.out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
  ```

### 3. 时间步嵌入
- **文件**：`SD.py` 中的 `TimeEmbedding` 类
- **数学原理**：将离散时间步映射为连续向量
  ```python
  class TimeEmbedding(nn.Module):
      def __init__(self, emb_dim):
          self.fc = nn.Sequential(
              nn.Linear(emb_dim, 4*emb_dim),
              nn.SiLU(),
              nn.Linear(4*emb_dim, emb_dim)
          )
          
      def forward(self, t):
          # 正弦位置编码
          half_dim = self.emb_dim // 2
          emb = math.log(10000) / (half_dim - 1)
          emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
          # ... 完整编码计算
  ```

### 4. 扩散过程调度
- **文件**：`SD.py` 中的 `cosine_beta_schedule` 函数
- **关键算法**：余弦噪声调度策略
  ```python
  def cosine_beta_schedule(T, s=0.008):
      steps = T + 1
      x = torch.linspace(0, T, steps)
      alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
      alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
      betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
      return torch.clip(betas, 0.0001, 0.9999)
  ```

### 5. 训练流程
- **文件**：`SD.py` 中的 `train_diffusion` 函数
- **核心步骤**：
  1. 加载预训练VAE
  2. 初始化UNet和文本编码器
  3. 前向扩散过程
  4. 噪声预测与损失计算

  ```python
  def train_diffusion():
      # 加载VAE
      vae = StableDiffusionVAE(...).to(device)
      vae.load_state_dict(torch.load("stable_diffusion_vae.pth"))
      
      # 初始化UNet
      diffusion_model = UNetModel(...).to(device)
      optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-4)
      
      # 训练循环
      for epoch in range(epochs):
          for images, _ in dataloader:
              # 图像编码到潜空间
              z, _, _ = vae.encode(images)
              
              # 随机时间步
              t = torch.randint(0, T, (images.size(0),), device=device)
              
              # 添加噪声
              noise = torch.randn_like(z)
              z_noisy = sqrt_alpha_cumprod[t] * z + sqrt_one_minus_alpha_cumprod[t] * noise
              
              # 预测噪声
              pred_noise = diffusion_model(z_noisy, t, text_emb)
              
              # 计算损失
              loss = F.mse_loss(pred_noise, noise)
  ```

## 代码学习路径建议

1. **从VAE开始**：
   - 理解`StableDiffusionVAE`的编码器/解码器结构
   - 运行`train_vae()`观察潜空间特征学习过程

2. **研究扩散过程**：
   - 在Jupyter Notebook中可视化`cosine_beta_schedule`生成的噪声曲线
   - 调试`forward_diffusion`函数观察潜变量加噪过程

3. **剖析UNet结构**：
   - 重点关注`SpatialTransformer`模块的交叉注意力机制
   - 使用TensorBoard可视化特征图变化

4. **完整训练流程**：
   - 从`train_diffusion()`入口跟踪整个训练过程
   - 添加日志记录和指标监控

建议结合论文《High-Resolution Image Synthesis with Latent Diffusion Models》阅读代码，重点关注：
- 第3章：潜扩散模型公式推导
- 第4.1节：UNet架构改进细节
- 附录B：训练参数设置 