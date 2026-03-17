"""Vision, proprioception, and joint encoders for the JEPA backbone."""
from __future__ import annotations

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """4-layer CNN: 64x64 RGB -> feature_dim vector."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ELU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ELU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProprioEncoder(nn.Module):
    """MLP: proprio_dim -> feature_dim."""

    def __init__(self, input_dim: int = 47, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ELU(),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JointEncoder(nn.Module):
    """Fuses vision + proprio into a single latent vector."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.vis_enc = VisionEncoder(128)
        self.prop_enc = ProprioEncoder(47, 128)
        self.fusion = nn.Sequential(
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, vision: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.vis_enc(vision), self.prop_enc(proprio)], dim=-1)
        return self.fusion(z)
