#!/usr/bin/env python3
"""Inspect CNN activations from the VisionEncoder."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
import torch

from tqjepa.models import CanonicalJEPA
from tqjepa.checkpoint_utils import load_jepa_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="jepa_logs/feature_maps.png")
    args = parser.parse_args()

    dev = torch.device(args.device)
    model = CanonicalJEPA().to(dev)
    sd, _ = load_jepa_checkpoint(args.jepa_ckpt, dev)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Synthetic input.
    img = torch.randn(1, 3, 64, 64, device=dev)

    # Hook into each conv layer.
    activations = []
    hooks = []
    for module in model.online_encoder.vis_enc.net:
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(
                lambda m, inp, out: activations.append(out.detach().cpu().numpy()[0])
            ))

    with torch.no_grad():
        model.online_encoder.vis_enc(img)

    for h in hooks:
        h.remove()

    fig, axes = plt.subplots(1, len(activations), figsize=(4 * len(activations), 4))
    if len(activations) == 1:
        axes = [axes]

    for i, act in enumerate(activations):
        # Show mean activation across channels.
        mean_act = np.mean(act, axis=0)
        axes[i].imshow(mean_act, cmap="viridis")
        axes[i].set_title(f"Conv {i+1}: {act.shape[0]}ch, {act.shape[1]}x{act.shape[2]}")
        axes[i].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
