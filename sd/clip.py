"""
CLIP模型实现（对比语言-图像预训练）
关键结构：
1. 文本编码器（类似Transformer）
2. 图像编码器（ViT结构，本文件未包含）
3. 对比学习目标函数（本文件未包含）
"""

import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    """CLIP文本嵌入层（包含词嵌入和位置嵌入）"""
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        # 词嵌入矩阵（49408是CLIP的BPE词表大小）
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # 可学习的位置嵌入（最大序列长度77）
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))
    
    def forward(self, tokens):
        # 词嵌入 + 位置嵌入
        x = self.token_embedding(tokens)  # [B, 77] -> [B, 77, 768]
        x += self.position_embedding  # 广播相加
        return x

class CLIPLayer(nn.Module):
    """CLIP Transformer层（与标准Transformer不同点：使用QuickGELU）"""
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        # 预层归一化架构
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        # 前馈网络（扩大4倍维度）
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        # 自注意力子层
        residue = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)  # 使用因果掩码
        x += residue

        # 前馈子层
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)  # QuickGELU（比标准GELU更快）
        x = self.linear_2(x)
        x += residue
        return x

class CLIP(nn.Module):
    """CLIP文本编码器（12层Transformer）"""
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.ModuleList([CLIPLayer(12, 768) for _ in range(12)])
        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # 文本编码流程
        state = self.embedding(tokens)
        for layer in self.layers:
            state = layer(state)
        output = self.layernorm(state)
        return output  # [B, 77, 768]