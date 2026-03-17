"""Action-conditioned recurrent latent predictor."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class LatentPredictor(nn.Module):
    """GRU-based dynamics model: (z_t, cmd_t, h_t) -> (z_{t+1}, h_{t+1}).

    The predictor maps from the online encoder's representation space to the
    target encoder's representation space. During training the loss is
    MSE(z_pred, stop_grad(z_target)).
    """

    def __init__(self, latent_dim: int = 256, cmd_dim: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + cmd_dim, latent_dim),
            nn.ELU(),
        )
        self.rnn = nn.GRUCell(latent_dim, latent_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self, z_t: torch.Tensor, c_t: torch.Tensor, h_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(torch.cat([z_t, c_t], dim=-1))
        h_next = self.rnn(x, h_t)
        z_next = self.output_proj(h_next)
        return z_next, h_next
