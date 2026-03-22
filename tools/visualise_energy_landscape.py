#!/usr/bin/env python3
"""Visualise the energy function over a 2D (vx, wz) command grid."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
import torch

from tqjepa.models import CanonicalJEPA, GoalEnergyHead
from tqjepa.checkpoint_utils import load_jepa_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt", required=True)
    parser.add_argument("--head_ckpt", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--grid_n", type=int, default=40)
    parser.add_argument("--out", default="jepa_logs/energy_landscape.png")
    args = parser.parse_args()

    dev = torch.device(args.device)

    model = CanonicalJEPA().to(dev)
    sd, _ = load_jepa_checkpoint(args.jepa_ckpt, dev)
    model.load_state_dict(sd, strict=False)
    model.eval()

    head = GoalEnergyHead().to(dev)
    head_sd = torch.load(args.head_ckpt, map_location=dev)
    head.load_state_dict(head_sd.get("energy_head_state_dict", head_sd.get("head_state_dict", head_sd)))
    head.eval()

    # Synthetic start / goal latents.
    z_start = torch.randn(1, model.latent_dim, device=dev) * 0.3
    z_goal = torch.randn(1, model.latent_dim, device=dev) * 0.3

    vx_range = np.linspace(-0.40, 0.40, args.grid_n)
    wz_range = np.linspace(-0.60, 0.60, args.grid_n)
    energy_grid = np.zeros((args.grid_n, args.grid_n), dtype=np.float32)

    with torch.no_grad():
        for i, vx in enumerate(vx_range):
            for j, wz in enumerate(wz_range):
                cmd = torch.tensor([[vx, 0.0, wz]], device=dev, dtype=torch.float32)
                z_roll = z_start.clone()
                h_t = torch.zeros(1, model.latent_dim, device=dev)
                for _ in range(args.horizon):
                    z_roll, h_t = model.predictor(z_roll, cmd, h_t)
                e = head(z_roll, z_goal.expand_as(z_roll))
                energy_grid[j, i] = float(e.item())

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        energy_grid, origin="lower", aspect="auto",
        extent=[vx_range[0], vx_range[-1], wz_range[0], wz_range[-1]],
        cmap="viridis",
    )
    ax.set_xlabel("vx (m/s)")
    ax.set_ylabel("wz (rad/s)")
    ax.set_title("Energy Landscape (lower = more compatible with goal)")
    fig.colorbar(im, ax=ax, label="Energy")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
