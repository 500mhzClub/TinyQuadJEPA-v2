"""Canonical JEPA with student-teacher EMA architecture."""
from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import JointEncoder
from .predictor import LatentPredictor


class CanonicalJEPA(nn.Module):
    """Student-teacher JEPA.

    - **online_encoder** (student): receives gradient updates.
    - **target_encoder** (teacher): exponential moving average of the online
      encoder — receives NO gradients.
    - **predictor**: maps ``online_encoder(state_t) + cmd_t`` to the target
      encoder's representation of ``state_{t+1}``.

    The asymmetry between the two encoders prevents representation collapse
    without needing VICReg's variance / covariance terms.
    """

    def __init__(self, latent_dim: int = 256, ema_tau: float = 0.996):
        super().__init__()
        self.latent_dim = latent_dim
        self.ema_tau = ema_tau

        # Online encoder — gets gradients.
        self.online_encoder = JointEncoder(latent_dim=latent_dim)

        # Target encoder — EMA copy, frozen.
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # Predictor: online space -> target space.
        self.predictor = LatentPredictor(latent_dim=latent_dim, cmd_dim=3)

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """target = tau * target + (1 - tau) * online."""
        for p_online, p_target in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            p_target.data.mul_(self.ema_tau).add_(
                p_online.data, alpha=1.0 - self.ema_tau,
            )

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def forward_step(
        self,
        vis_t: torch.Tensor,
        prop_t: torch.Tensor,
        cmd_t: torch.Tensor,
        vis_next: torch.Tensor,
        prop_next: torch.Tensor,
        h_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-step prediction with loss.

        Returns (loss, h_next, z_pred, z_target).
        """
        z_t = self.online_encoder(vis_t, prop_t)
        z_pred, h_next = self.predictor(z_t, cmd_t, h_t)

        with torch.no_grad():
            z_target = self.target_encoder(vis_next, prop_next)

        loss = F.mse_loss(z_pred, z_target.detach(), reduction="none").mean(dim=-1)
        return loss, h_next, z_pred, z_target

    def encode_online(self, vis: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        """Encode with the online (student) encoder."""
        return self.online_encoder(vis, prop)

    def encode_target(self, vis: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        """Encode with the target (teacher) encoder."""
        return self.target_encoder(vis, prop)
