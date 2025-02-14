"""
对比语言-图像预训练模型（CLIP）文本编码器实现
核心组件：
1. 文本编码器（本文件实现）
2. 图像编码器（ViT/ResNet结构，未包含）
3. 对比损失函数（未包含）
参考论文：Learning Transferable Visual Models From Natural Language Supervision

同一个batchsize中，自动产生负样本
可以从0训练也可以从预训练模型开始训练，比如图片先VIT，文本先BERT
学习到的是语义和图片的匹配度

最大的贡献是给文字和图片拉到同一个空间
"""

import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    """CLIP文本嵌入层（词嵌入 + 位置嵌入）
    关键设计：
    - 使用可学习的位置嵌入（非正弦编码）
    - 最大序列长度固定为77（BPE编码限制）
    
    参数说明：
    n_vocab: 49408 （CLIP的BPE词表大小）
    n_embd: 768 （嵌入维度，与Transformer隐藏层一致）
    n_token: 77 （最大文本标记数）
    """
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embd)  # 词嵌入矩阵
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))  # 可学习位置编码
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # 输入形状：[batch_size, 77]
        # 词嵌入 + 位置嵌入（广播相加）
        x = self.token_embedding(tokens)  # [B, 77, 768]
        x += self.position_embedding  # [77, 768] → 广播到[B, 77, 768]
        return x

class CLIPLayer(nn.Module):
    """CLIP Transformer层（与标准Transformer的差异点）：
    1. 使用预层归一化（Pre-LN）结构
    2. 前馈网络使用QuickGELU激活
    3. 注意力层无偏置项
    
    结构组成：
    - 自注意力子层
    - 前馈神经网络子层
    - 残差连接
    """
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd, in_proj_bias=False)  # 无偏置投影
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)  # 扩展4倍维度
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力子层
        residue = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)  # 因果掩码防止信息泄漏
        x += residue

        # 前馈子层
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)  # QuickGELU（比标准GELU计算更快）
        x = self.linear_2(x)
        x += residue
        return x

class CLIP(nn.Module):
    """CLIP文本编码器完整架构
    结构特点：
    - 12层Transformer
    - 输出层归一化
    - 输出取EOS标记的嵌入作为文本表示
    
    未实现但重要的部分：
    1. 图像编码器（通常为ViT或ResNet-50x4）
    2. 对比损失计算（相似度矩阵 + 对称交叉熵）
    3. 可学习的温度参数τ
    """
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.ModuleList([CLIPLayer(12, 768) for _ in range(12)])  # 12层Transformer
        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # 输入形状：[batch_size, 77]
        state = self.embedding(tokens)
        for layer in self.layers:
            state = layer(state)
        output = self.layernorm(state)
        return output[:, 0, :]  # 取EOS标记作为文本全局表示 [B, 768]

"""
CLIP模型完整流程补充说明（代码中未实现部分）：

图像编码器示例（ViT结构）：
class ViTEncoder:
    def __init__(self):
        self.patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16)  # 分块嵌入
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))  # 分类标记
        self.position_embed = nn.Parameter(torch.randn(1, 50, 768))  # 位置编码
        self.transformer = TransformerEncoder(...)  # 与文本编码器类似的结构

def contrastive_loss(logits_per_text, logits_per_image, temperature=0.07):
    
    CLIP对比损失计算，通过矩阵相似度对齐图文表示
    
    参数说明：
        logits_per_text: 文本-图像相似度矩阵 [B, B]，logits_text[i,j] 表示第i个文本与第j个图像的匹配度
        logits_per_image: 图像-文本相似度矩阵 [B, B]，logits_image[i,j] 表示第i个图像与第j个文本的匹配度
        temperature: 温度系数，控制softmax分布的平滑程度
        
    损失构建过程：
        1. 对角线对齐：假设batch内第i个文本与第i个图像是正样本对
        2. 双路监督：同时计算文本->图像和图像->文本两个方向的交叉熵损失
        3. 温度缩放：用可学习的温度参数调整logits分布，改善训练稳定性
        4. 对称优化：最终损失取两个方向损失的平均，确保图文表示空间的双向对齐
    
    B = logits_per_text.size(0)  # 获取batch size
    
    # 创建目标标签：对角线索引为正样本（第i个文本匹配第i个图像）
    labels = torch.arange(B, device=logits_per_text.device)  # [B]
    
    # 文本到图像方向的对比损失（每个文本与对应图像作为正样本）
    loss_text = F.cross_entropy(
        logits_per_text / temperature,  # 温度缩放调整logits量纲
        labels,  # 目标标签为对角线索引
    )
    
    # 图像到文本方向的对比损失（每个图像与对应文本作为正样本）
    loss_image = F.cross_entropy(
        logits_per_image / temperature,  # 保持两个方向温度系数一致
        labels,
    )
    
    # 对称损失求平均，确保文本和图像编码器的协同优化
    return (loss_text + loss_image) / 2

训练关键技巧：
- 超大batch size（可达32768？） （但容易出现伪负样本）
- 梯度裁剪（防止梯度爆炸）
- 混合精度训练
- 图像/文本增强（随机裁剪、文本dropout）

零样本分类流程：
1. 编码所有类别文本（"a photo of a {label}"）
2. 编码待分类图像
3. 计算图像嵌入与所有文本嵌入的余弦相似度
4. 选择相似度最高的类别

# 以下是完整的对比损失计算流程（伪代码实现）
class CLIPModel(nn.Module):
    def __init__(self):
        # 文本编码器（本文件已实现）
        self.text_encoder = CLIP()
        
        # 图像编码器（示例：ViT-B/32）
        self.image_encoder = VisionTransformer(
            image_size=224,
            patch_size=32,
            width=768,
            layers=12,
            heads=12
        )
        
        # 投影层（将文本/图像嵌入映射到同一空间）
        self.text_proj = nn.Linear(768, 512)  # 文本投影头
        self.image_proj = nn.Linear(768, 512) # 图像投影头
        
        # 可学习的温度参数（初始值对应论文中的0.07）
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))

    def encode_text(self, text):
        # [B, 77] → [B, 768]
        text_features = self.text_encoder(text)
        # [B, 768] → [B, 512]
        return self.text_proj(text_features)

    def encode_image(self, image):
        # [B, 3, 224, 224] → [B, 768]
        image_features = self.image_encoder(image)
        # [B, 768] → [B, 512]
        return self.image_proj(image_features)

    def forward(self, text, image):
        # 获取归一化后的特征
        text_emb = F.normalize(self.encode_text(text), dim=-1)
        image_emb = F.normalize(self.encode_image(image), dim=-1)
        
        # 计算相似度矩阵（余弦相似度缩放后）
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, image_emb.t()) * logit_scale # 一个序列默认是竖着排列
        logits_per_image = logits_per_text.t()                                # logit_scale温度系数，避免损失函数进入饱和区（梯度消失）
        
        return logits_per_text, logits_per_image

对比损失计算流程详解：
1. 特征提取：
   - 文本输入：[B, 77] → 文本编码器 → [B, 768]
   - 图像输入：[B, 3, 224, 224] → 图像编码器 → [B, 768]

2. 特征投影：
   - 文本特征 → 线性层(768→512) → L2归一化 → [B, 512]
   - 图像特征 → 线性层(768→512) → L2归一化 → [B, 512]

3. 相似度计算：
   - 文本到图像：text_emb @ image_emb.T → [B, B]   因为softmax是按照行计算的
   - 图像到文本：image_emb @ text_emb.T → [B, B]

4. 温度缩放：
   - logit_scale = exp(可学习参数)
   - 实际实现：scale = logit_scale.exp().clamp(max=100)

关键数学公式：
logits_per_text[i,j] = (text_emb[i] · image_emb[j]) * exp(logit_scale)
loss = (CE(softmax(logits_text), labels) + CE(softmax(logits_image), labels)) / 2
"""