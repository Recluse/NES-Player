"""Pixel state encoder for Decision Transformer.

Replaces 6 hand-crafted features with learned conv features from raw frames.
This gives DT much richer state information.

    from nes_player.policy.pixel_encoder import PixelStateEncoder
    encoder = PixelStateEncoder(d_model=384)
    # encoder(frame_tensor) -> (B, T, d_model)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PixelStateEncoder(nn.Module):
    """CNN encoder: (B, T, H, W, 3) -> (B, T, d_model).

    Processes each frame independently through a small CNN,
    then projects to d_model.
    """

    def __init__(self, d_model: int = 384):
        super().__init__()
        # Small CNN for NES frames (224x240x3)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),  # -> 32x56x60
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 64x28x30
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # -> 64x14x15
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> 64x4x4
        )
        self.proj = nn.Linear(64 * 4 * 4, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, W, 3) -> (B, T, d_model)"""
        B, T = x.shape[:2]
        # Reshape to process each frame independently
        x = x.reshape(B * T, *x.shape[2:])  # (B*T, H, W, 3)
        x = x.permute(0, 3, 1, 2)  # (B*T, 3, H, W)
        x = self.conv(x)  # (B*T, 64, 4, 4)
        x = x.reshape(B * T, -1)  # (B*T, 1024)
        x = self.proj(x)  # (B*T, d_model)
        return x.reshape(B, T, -1)  # (B, T, d_model)


class SmallPixelEncoder(nn.Module):
    """Smaller CNN encoder for faster training."""

    def __init__(self, d_model: int = 384):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=8),  # -> 16x28x30
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=4),  # -> 32x7x8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),  # -> 32x2x2
        )
        self.proj = nn.Linear(32 * 2 * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(B * T, -1)
        x = self.proj(x)
        return x.reshape(B, T, -1)
