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
from matplotlib import cm

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
    sd, cfg = load_jepa_checkpoint(args.jepa_ckpt, dev)
    model.load_state_dict(sd, strict=False)
    model.eval()

    head = GoalEnergyHead().to(dev)
    head_sd = torch.load(args.head_ckpt, map_location=dev)
    head.load_state_dict(head_sd.get("energy_head_state_dict", head_sd.get("head_state_dict", head_sd)))
    head.eval()

    epoch = head_sd.get("epoch", "?")
    global_step = head_sd.get("global_step", "?")

    # Synthetic start / goal latents.
    z_start = torch.randn(1, model.latent_dim, device=dev) * 0.3
    z_goal = torch.randn(1, model.latent_dim, device=dev) * 0.3

    vx_vals = np.linspace(-0.40, 0.40, args.grid_n)
    wz_vals = np.linspace(-0.60, 0.60, args.grid_n)
    VX, WZ = np.meshgrid(vx_vals, wz_vals)
    ENERGY = np.zeros_like(VX, dtype=np.float32)

    with torch.no_grad():
        for i in range(args.grid_n):
            for j in range(args.grid_n):
                cmd = torch.tensor([[VX[i, j], 0.0, WZ[i, j]]], device=dev, dtype=torch.float32)
                z_roll = z_start.clone()
                h_t = torch.zeros(1, model.latent_dim, device=dev)
                for _ in range(args.horizon):
                    z_roll, h_t = model.predictor(z_roll, cmd, h_t)
                e = head(z_roll, z_goal.expand_as(z_roll))
                ENERGY[i, j] = float(e.item())

    best_flat = int(np.argmin(ENERGY))
    bi, bj = np.unravel_index(best_flat, ENERGY.shape)
    best_vx = float(VX[bi, bj])
    best_wz = float(WZ[bi, bj])
    best_e = float(ENERGY[bi, bj])

    fig = plt.figure(figsize=(16, 7))

    # Left: heatmap
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(
        ENERGY, origin="lower", aspect="auto",
        extent=[vx_vals[0], vx_vals[-1], wz_vals[0], wz_vals[-1]],
        interpolation="bicubic", cmap="viridis",
    )
    ax1.scatter([best_vx], [best_wz], marker="x", s=120, color="white", linewidths=2, zorder=5)
    ax1.set_xlabel("vx (m/s)")
    ax1.set_ylabel("wz (rad/s)")
    ax1.set_title("Energy heatmap (lower = more goal-compatible)")
    cbar1 = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Energy")
    ax1.text(
        0.02, 0.02,
        f"best: vx={best_vx:+.2f}, wz={best_wz:+.2f}\nenergy={best_e:.4f}",
        transform=ax1.transAxes, va="bottom", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    # Right: 3-D surface
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf = ax2.plot_surface(VX, WZ, ENERGY, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.95)
    ax2.scatter([best_vx], [best_wz], [best_e], s=60, color="red", zorder=5)
    ax2.set_xlabel("vx (m/s)")
    ax2.set_ylabel("wz (rad/s)")
    ax2.set_zlabel("Energy")
    ax2.set_title("Energy surface")
    fig.colorbar(surf, ax=ax2, shrink=0.6, aspect=18, pad=0.08)

    fig.suptitle(
        f"JEPA energy landscape | epoch={epoch} step={global_step} | horizon={args.horizon}",
        fontsize=14,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"Saved to {args.out}")
    print(f"Best command: vx={best_vx:+.3f}, wz={best_wz:+.3f}, energy={best_e:.4f}")


if __name__ == "__main__":
    main()
