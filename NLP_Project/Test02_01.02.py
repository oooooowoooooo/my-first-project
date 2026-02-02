import torch
import torch.nn as nn
import torch.nn.functional as F
zero_tensor=torch.zeros(1,3,224,224)
Conv_layer=nn.Conv2d(3,768,16,16)
patches=Conv_layer(zero_tensor)
patches=patches.flatten(2)
patches=patches.transpose(1,2)
print(patches.shape)
qkv_layer = nn.Linear(768, 768 * 3)
Q,K,V=qkv_layer(patches).chunk(3, dim=-1)
print(Q.shape,K.shape,V.shape)
Q=Q.reshape(1,196,12,64)
Q=Q.permute(0,2,1,3)
K=K.reshape(1,196,12,64)
K=K.permute(0,2,1,3)
V=V.reshape(1,196,12,64)
V=V.permute(0,2,1,3)
scores = Q @ K.transpose(-2, -1)
scores = scores / (Q.size(-1) ** 0.5)

probs = F.softmax(scores, dim=-1)
attn_output = probs @ V
attn_output = attn_output.permute(0, 2, 1, 3).contiguous().reshape(1, 196, 768)
layer=nn.Linear(768,768)
output=layer(attn_output)


class LinearWithLoRA(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=1.0):
        super().__init__()
        self.linear = original_linear  # 1. 吃掉原来的 Linear (冻结它)

        # 冻结原参数 (不更新权重)
        for param in self.linear.parameters():
            param.requires_grad = False

        # 2. 定义 LoRA 旁路 (可训练)
        self.lora_a = nn.Parameter(torch.randn(original_linear.in_features, rank) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(rank, original_linear.out_features))

        self.scaling = alpha / rank  # 缩放系数

    def forward(self, x):
        # 3. 原路 (Base)
        base_out = self.linear(x)

        # 4. 旁路 (LoRA)
        lora_out = (x @ self.lora_a @ self.lora_b) * self.scaling


        # 5. 合流
        return base_out + lora_out
fc = nn.Linear(768, 768)
fc_lora = LinearWithLoRA(fc, rank=4)
print(fc_lora)
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(), # 激活函数，比 ReLU 更平滑
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


