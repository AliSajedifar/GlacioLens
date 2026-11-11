"""P22 reference block only; no TerraTorch wrapper, pretrained weights or trainer.

The context tensor must be the LIVE, differentiable fused UPerNet feature
before the original classifier. The caller preserves the original B head,
upsampling and all input normalization. All original B modules train jointly.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class JointSpatialPath(nn.Module):
    """Context-conditioned additive logit residual (57,249 parameters).

    Construct in a separate torch.random.fork_rng context so branch
    initialization does not consume the historical training/data RNG stream.
    Only the final prediction convolution is zero-initialized.
    """

    def __init__(self, input_channels: int = 9, context_channels: int = 256) -> None:
        super().__init__()
        if input_channels != 9 or context_channels != 256:
            raise ValueError("P22 fixes nine normalized inputs and 256 context channels.")
        self.local = nn.Sequential(
            nn.Conv2d(9, 32, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.predict = nn.Conv2d(64, 1, 1, bias=True)
        nn.init.zeros_(self.predict.weight)
        nn.init.zeros_(self.predict.bias)

    def residual(self, inputs: Tensor, context: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 9:
            raise ValueError("Expected [N,9,H,W] already-normalized B inputs.")
        if context.ndim != 4 or context.shape[1] != 256:
            raise ValueError("Expected live UPerNet feature [N,256,h,w].")
        if context.shape[0] != inputs.shape[0]:
            raise ValueError("Batch mismatch between image and decoder feature.")
        local = self.local(inputs)
        semantic = F.interpolate(
            self.context(context), size=local.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        residual = self.predict(self.fuse(torch.cat((local, semantic), dim=1)))
        return F.interpolate(
            residual, size=inputs.shape[-2:], mode="bilinear", align_corners=False,
        )

    def forward(self, inputs: Tensor, context: Tensor, base_logits: Tensor) -> Tensor:
        if base_logits.ndim != 4 or base_logits.shape[1] != 1:
            raise ValueError("Expected original full-resolution logits [N,1,H,W].")
        if base_logits.shape[0] != inputs.shape[0] or base_logits.shape[-2:] != inputs.shape[-2:]:
            raise ValueError("Original logits must retain the original B output grid.")
        return base_logits + self.residual(inputs, context)
