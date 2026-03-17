#!/usr/bin/env python3
"""Spot-check HDF5 dataset files by exporting random trajectory clips as GIFs."""
from __future__ import annotations

import argparse
import glob
import os

import h5py
import imageio
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="jepa_final_dataset")
    parser.add_argument("--out_dir", default="verification_videos")
    parser.add_argument("--n_clips", type=int, default=5)
    parser.add_argument("--clip_len", type=int, default=60)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.data_dir, "*_rgb.h5")))
    if not files:
        print(f"No *_rgb.h5 files found in {args.data_dir}")
        return

    rng = np.random.RandomState(42)

    for ci in range(args.n_clips):
        fpath = files[rng.randint(len(files))]
        with h5py.File(fpath, "r") as h5f:
            N, T = h5f["vision"].shape[:2]
            e = rng.randint(N)
            t0 = rng.randint(max(1, T - args.clip_len))
            t1 = min(t0 + args.clip_len, T)

            vis = h5f["vision"][e, t0:t1]  # (T, 3, 64, 64)
            cmds = h5f["cmds"][e, t0:t1]

            has_col = "collisions" in h5f
            col = h5f["collisions"][e, t0:t1] if has_col else np.zeros(t1 - t0, dtype=bool)

        frames = np.transpose(vis, (0, 2, 3, 1))  # -> (T, 64, 64, 3)
        out_path = os.path.join(args.out_dir, f"clip_{ci:02d}.gif")
        imageio.mimsave(out_path, list(frames), duration=0.1)

        mean_vx = float(np.mean(cmds[:, 0]))
        n_col = int(np.sum(col))
        print(f"  clip {ci}: {os.path.basename(fpath)} env={e} t={t0}-{t1} "
              f"mean_vx={mean_vx:.2f} collisions={n_col} -> {out_path}")

    print(f"Saved {args.n_clips} clips to {args.out_dir}/")


if __name__ == "__main__":
    main()
