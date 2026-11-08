"""DINOv2 state encoder: frozen pretrained visual features for BC/DT.

Replaces 6 hand-crafted features (score band, brightness, RGB means) with
DINOv2 pretrained visual features (384-dim). The encoder is frozen — only
the downstream policy is trained.

From "Visual Encoders for Data-Efficient Imitation Learning" (arXiv:2312.02312):
pretrained encoders significantly improve BC over end-to-end training,
especially with low-res images and limited demonstrations.

    from nes_player.policy.dinov2_encoder import DINOv2Encoder
    encoder = DINOv2Encoder()  # frozen
    feats = encoder(frames_tensor)  # (B, T, 384)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class DINOv2Encoder(nn.Module):
    """Frozen DINOv2 ViT-S encoder for NES frames.

    Input:  (B, T, H, W, 3) float in [0, 1]
    Output: (B, T, 384) projected to d_model if specified.
    """

    def __init__(self, d_model: int = 384, image_size: int = 224):
        super().__init__()
        self.image_size = image_size
        self.dino_dim = 384  # ViT-S/14

        # Load frozen DINOv2
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.eval()

        # Projection if d_model != 384
        if d_model != self.dino_dim:
            self.proj = nn.Linear(self.dino_dim, d_model)
        else:
            self.proj = nn.Identity()

    def _prepare_frames(self, x: torch.Tensor) -> torch.Tensor:
        """Resize NES frames (H, W, 3) to DINOv2 input (224, 224, 3).

        NES frames are 224x240 — resize to 224x224 (square crop).
        """
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])  # (B*T, H, W, 3)
        x = x.permute(0, 3, 1, 2)  # (B*T, 3, H, W)

        # Resize to 224x224
        x = torch.nn.functional.interpolate(
            x, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )

        # DINOv2 normalization (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device)
        x = (x - mean[None, :, None, None]) / std[None, :, None, None]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, W, 3) -> (B, T, d_model)"""
        B, T = x.shape[:2]
        frames = self._prepare_frames(x)  # (B*T, 3, 224, 224)

        with torch.no_grad():
            feats = self.dino(frames)  # (B*T, 384)

        feats = self.proj(feats)  # (B*T, d_model)
        return feats.reshape(B, T, -1)  # (B, T, d_model)


class DINOv2Features:
    """Precompute DINOv2 features for an episode (offline).

    For training, it's faster to precompute features once than to run
    DINOv2 on every batch.
    """

    def __init__(self, device: str = "cpu"):
        self.encoder = DINOv2Encoder().to(device)
        self.device = device

    @torch.no_grad()
    def extract(self, frames: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """Extract features from (T, H, W, 3) frames -> (T, 384)."""
        T = len(frames)
        feats = []
        for i in range(0, T, batch_size):
            batch = torch.tensor(
                frames[i : i + batch_size], dtype=torch.float32, device=self.device
            ) / 255.0
            # (B, H, W, 3) -> (1, B, H, W, 3) for the encoder
            batch = batch.unsqueeze(0)
            f = self.encoder(batch)  # (1, B, 384)
            feats.append(f.squeeze(0).cpu().numpy())
        return np.concatenate(feats, axis=0)
