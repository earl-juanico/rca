# --------------------------------------------------------
# Ristretto
# Copyright (c) 2025 LiAutoAD
# Licensed under The MIT License
# --------------------------------------------------------

import numpy as np
from torch import nn
import torch.nn.functional as F



class FFN(nn.Module):
    def __init__(self, dim, out_dim, mlp_ratio=3):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.f1 = nn.Linear(dim, mlp_ratio * dim)
        self.f2 = nn.Linear(dim, mlp_ratio * dim)
        self.g = nn.Linear(mlp_ratio * dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.layernorm(x)
        input = x
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.g(x)
        return x


class TokenAdaptiveProjector(nn.Module):
    def __init__(self, vit_hidden_size, llm_hidden_size, num_image_token):
        super().__init__()
        self.num_image_token = num_image_token
        self.mlp = FFN(vit_hidden_size, llm_hidden_size)

    def find_resize_hw(self, H, W, num_image_token):
        target_h = target_w = int(num_image_token ** 0.5)
        resize_h = int(np.ceil(H / target_h)) * target_h
        resize_w = int(np.ceil(W / target_w)) * target_h
        return resize_h, resize_w, target_h, target_w

    def forward(self, x, num_image_token=None):
        bs, L, C = x.shape

        if num_image_token is None:
            num_image_token = self.num_image_token

        H = W = int(L ** 0.5)
        assert L == H * W, "L should equal H * W"

        resize_h, resize_w, target_h, target_w = self.find_resize_hw(
            H, W, num_image_token
        )

        x = x.view(bs, H, W, C).permute(0, 3, 1, 2)  # [bs, C, H, W]
        if resize_h != H or resize_w != W:
            x = F.interpolate(
                x, size=(resize_h, resize_w), mode="bilinear", align_corners=True
            )
            _, _, H, W = x.shape

        n = target_h
        patch_h = patch_w = H // n

        x = (
            F.avg_pool2d(x, (patch_h, patch_w)).permute(0, 2, 3, 1).reshape(bs, -1, C)
        )
        x = self.mlp(x)
        return x
