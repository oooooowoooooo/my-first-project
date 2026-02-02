import torch
import torch.nn as nn
import math


class SimpleViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 1. 真实的 Patchify (用卷积实现)
        # kernel_size=16, stride=16 就能完美切出 16x16 的块
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)

        # 2. QKV 投影层 (一次性算出 QKV，比定义三个 Linear 更快)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)

        # 3. 输出层
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        pass  # 等会填
