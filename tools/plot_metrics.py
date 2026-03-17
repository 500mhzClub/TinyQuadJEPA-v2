#!/usr/bin/env python3
"""Plot JEPA training metrics from CSV log."""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def smooth(y: np.ndarray, k: int = 50) -> np.ndarray:
    if len(y) < k:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="jepa_logs/training_metrics.csv")
    parser.add_argument("--out", default="jepa_logs/training_curve.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # MSE loss
    axes[0].plot(df["step"], smooth(df["mse_loss"].values), color="tab:blue")
    axes[0].set_title("MSE Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")

    # EMA tau
    axes[1].plot(df["step"], df["ema_tau"], color="tab:orange")
    axes[1].set_title("EMA Tau")
    axes[1].set_xlabel("Step")

    # Collapse monitor
    axes[2].plot(df["step"], smooth(df["z_target_std"].values), color="tab:green")
    axes[2].axhline(0.1, color="red", linestyle="--", alpha=0.5, label="collapse threshold")
    axes[2].set_title("z_target std (collapse monitor)")
    axes[2].set_xlabel("Step")
    axes[2].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
