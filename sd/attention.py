"""
Transformer 注意力机制实现
包含自注意力(Self-Attention)和交叉注意力(Cross-Attention)模块
核心公式：Attention(Q,K,V) = softmax(QK^T/√d_k)V
"""

import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    """自注意力机制（用于捕捉序列内部依赖关系）"""
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        """
        参数：
            n_heads: 注意力头数（将特征分割为多头的数量）
            d_embed: 输入特征维度（必须能被n_heads整除）
            in_proj_bias: 是否在QKV投影中添加偏置
            out_proj_bias: 是否在输出投影中添加偏置
        """
        super().__init__()
        # 合并Q、K、V的投影矩阵（对应公式中的Wq, Wk, Wv）
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # 输出投影矩阵（对应公式中的Wo）
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads  # 每个头的维度

    def forward(self, x, causal_mask=False):
        """
        前向传播过程（时间复杂度O(n^2), 空间复杂度O(n^2)）
        输入：
            x: 输入序列 [batch_size, seq_len, d_embed]
            causal_mask: 是否使用因果掩码（用于生成任务）
        输出：
            注意力加权后的序列 [batch_size, seq_len, d_embed]
        """
        # 输入形状
        batch_size, seq_len, d_embed = x.shape
        
        # 生成Q、K、V（对应公式中的Q,K,V）
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # 多头分割与转置（为了并行计算各头的注意力）
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # 计算注意力分数（缩放点积）
        # [batch, heads, seq_len, seq_len]
        attn_scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        
        # 因果掩码（防止关注未来信息）
        if causal_mask:
            mask = torch.ones_like(attn_scores, dtype=torch.bool).triu(1)
            attn_scores.masked_fill_(mask, -torch.inf)
        
        # 注意力权重归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 注意力加权求和
        output = attn_weights @ v
        
        # 合并多头结果
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_embed)
        
        # 输出投影
        return self.out_proj(output)

class CrossAttention(nn.Module):
    """交叉注意力（用于处理多模态交互，如文本-图像）"""
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        """
        参数：
            d_cross: 交叉模态的输入维度（如文本编码维度）
        """
        super().__init__()
        # Q来自主模态，K、V来自交叉模态
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        """
        前向传播（Query来自x，Key和Value来自y）
        输入：
            x: 主模态输入 [batch, seq_len_q, d_embed]
            y: 交叉模态输入 [batch, seq_len_kv, d_cross]
        输出：
            跨模态注意力结果 [batch, seq_len_q, d_embed]
        """
        batch_size, seq_len_q, _ = x.shape
        
        # 生成Q、K、V
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        
        # 多头分割
        q = q.view(batch_size, seq_len_q, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        
        # 计算注意力
        attn_scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = attn_weights @ v
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
        
        return self.out_proj(output)