"""
去噪扩散概率模型（DDPM）采样器实现
核心功能：
1. 管理噪声调度（Noise Schedule）
2. 实现前向扩散过程
3. 执行反向去噪过程
参考论文：Denoising Diffusion Probabilistic Models (Ho et al. 2020)
在Stable Diffusion中的应用：在潜在空间（latent space）执行扩散过程，而非像素空间
"""

import torch
import numpy as np

class DDPMSampler:
    def __init__(self, generator: torch.Generator, num_training_steps=1000, 
                beta_start: float = 0.00085, beta_end: float = 0.0120):
        """
        参数解析：
        - generator: 控制随机性的随机数生成器（确保结果可复现）
        - num_training_steps: 训练总步数（默认1000）
        - beta_start/beta_end: 噪声调度参数（来自Stable Diffusion官方配置）
        
        关键变量：
        - betas: 噪声率序列，控制每个时间步添加的噪声量
        - alphas: 1 - betas，表示保留原始数据的比例
        - alphas_cumprod: α的累积乘积，ᾱ_t = Π(1-β_s) from s=1 to t
        
        数学基础：
        前向扩散过程：q(x_t|x_{t-1}) = N(x_t; sqrt(1-β_t)x_{t-1}, β_tI)
        逆向去噪过程：p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))
        """
        # 使用线性调度生成beta（原始DDPM使用余弦调度，SD采用改进版本）
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # ᾱ_t = Π(1-β_s)
        self.one = torch.tensor(1.0)

        self.generator = generator  # 随机数生成器，在某些特定操作中使用独立的随机数生成器，就需要 torch.Generator

        # 训练相关参数
        self.num_train_timesteps = num_training_steps
        # 时间步倒序排列（扩散过程从t=0到t=T，去噪从t=T到t=0）
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
        '''
        在 NumPy 中，.copy() 是用于 创建数组的副本（deep copy） 的方法，它确保新数组不会与原数组共享内存。
        pytorch中.copy_() 是原地操作，会直接修改原张量
        clone() 用于创建张量的 深拷贝，新张量和原张量 不共享存储与原始张量无关。
        新的 b 张量 不会影响 a，它们的 存储空间是独立的。
        如果 a 是 计算图中的叶子节点（requires_grad=True），直接 clone() 会 复制计算图的梯度
        如果你想要复制但不保留梯度信息，可以用 detach().clone()
        '''

    def set_inference_timesteps(self, num_inference_steps=50):
        # Stable Diffusion在潜空间操作，噪声扰动更平滑，适合这种近似
        #原始DDPM理论确实需要逐步计算。
        # Stable Diffusion中的加速是通过工程近似实现的，与DDIM的理论突破有本质区别。
        # 这种实践优化在潜空间扩散中效果较好，但若追求更高效率，仍需采用DDIM等改进算法。

        """
        设置推理时间步（加速采样过程）
        实现策略：
        - 从1000个训练步中均匀采样50个时间点（步长=20）
        - 例如：当num_inference_steps=50时，使用[999, 979, 959,...]等时间步
        
        数学原理：
        通过q(x_t|x_{t-Δ})近似原过程，其中Δ=20。当Δ足够小时，这种近似是有效的
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        """计算前一时间步（用于反向过程）"""
        return timestep - self.num_train_timesteps // self.num_inference_steps
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        """
        计算反向过程的方差σ_t^2（公式7）
        推导过程：
        σ_t^2 = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t
        当t=0时，σ_0^2 = β_0（无随机性）
        """
        prev_t = self._get_previous_timestep(timestep)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)  # 防止数值不稳定
        return variance
    
    def set_strength(self, strength=1):
        """
        设置控制强度（用于图像到图像生成）
        参数说明：
        - strength ∈ [0,1]: 控制添加的噪声量
        - strength=1: 完全重绘
        - strength=0.5: 保留部分原始图像信息
        
        实现原理：
        通过跳过前N个噪声步，保留部分原始图像信息
        """
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        """
        执行单步去噪过程（算法2步骤4-6）
        输入：
        - timestep: 当前时间步
        - latents: 当前噪声潜变量x_t
        - model_output: 模型预测的噪声ε_θ(x_t,t)
        
        数学推导：
        1. 根据公式(15)计算预测的原始样本x_0：
           x_0 = (x_t - sqrt(1-ᾱ_t)ε_θ) / sqrt(ᾱ_t)
        2. 根据公式(7)计算均值μ_t：
           μ_t = sqrt(ᾱ_{t-1})β_t/(1-ᾱ_t) x_0 + sqrt(α_t)(1-ᾱ_{t-1})/(1-ᾱ_t) x_t
        3. 采样x_{t-1} ~ N(μ_t, σ_t^2I)
        """
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # 计算α和β的相关值
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 预测原始样本x0（公式15）
        pred_x0 = (latents - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5

        # 计算均值μ_t（公式7）
        pred_prev_sample = (alpha_prod_t_prev**0.5 * current_beta_t / beta_prod_t) * pred_x0 + \
                          (current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t) * latents

        # 添加噪声（当t>0时）
        variance = 0
        if t > 0:
            noise = torch.randn(model_output.shape, generator=self.generator,
                              device=model_output.device, dtype=model_output.dtype)
            variance = (self._get_variance(t)**0.5) * noise

        return pred_prev_sample + variance

    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor):
        """
        前向扩散过程（公式4）：q(x_t|x_0)
        数学形式：
        x_t = sqrt(ᾱ_t)x_0 + sqrt(1-ᾱ_t)ε, ε ~ N(0,I)
        
        实现细节：
        - 使用预先计算的ᾱ_t进行线性组合
        - 保持广播维度一致性（处理不同维度的输入）
        """
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 维度调整确保广播正确（例如处理4D图像输入）
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # 根据重参数化技巧生成噪声样本
        noise = torch.randn(original_samples.shape, generator=self.generator,
                           device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

        

    