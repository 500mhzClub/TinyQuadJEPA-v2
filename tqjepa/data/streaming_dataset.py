"""Streaming HDF5 dataset for JEPA and energy-head training."""
from __future__ import annotations

import glob
import os
from typing import Iterator, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset


class StreamingJEPADataset(IterableDataset):
    """Yields (vision, proprio, cmds, dones, collisions) sequence tuples.

    Streams directly from HDF5 files — no full-dataset RAM copy.  Each worker
    gets a disjoint file shard so there is no duplication.

    Args:
        data_dir: directory containing ``*_rgb.h5`` files.
        seq_len: number of timesteps per returned sequence.
        batch_size: worker-side micro-batch size.
        require_no_done: skip sequences that contain a ``done`` flag.
        require_no_collision: skip sequences that contain a ``collision`` flag.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 16,
        batch_size: int = 256,
        require_no_done: bool = True,
        require_no_collision: bool = True,
    ):
        super().__init__()
        self.files: List[str] = sorted(glob.glob(os.path.join(data_dir, "*_rgb.h5")))
        if not self.files:
            raise FileNotFoundError(f"No *_rgb.h5 files in {data_dir}")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.require_no_done = require_no_done
        self.require_no_collision = require_no_collision

    def _shard_files(self) -> List[str]:
        info = torch.utils.data.get_worker_info()
        if info is None:
            return self.files
        per_worker = max(1, len(self.files) // info.num_workers)
        lo = info.id * per_worker
        hi = lo + per_worker if info.id < info.num_workers - 1 else len(self.files)
        return self.files[lo:hi]

    def _build_indices(
        self, h5f: h5py.File,
    ) -> List[Tuple[int, int]]:
        """Return (env_idx, start_t) pairs for valid sequences."""
        n_envs = h5f["vision"].shape[0]
        T = h5f["vision"].shape[1]
        dones = h5f["dones"][:] if "dones" in h5f else None
        collisions = h5f["collisions"][:] if "collisions" in h5f else None
        indices = []

        for e in range(n_envs):
            for t0 in range(0, T - self.seq_len + 1, self.seq_len):
                t1 = t0 + self.seq_len
                if self.require_no_done and dones is not None:
                    if np.any(dones[e, t0:t1]):
                        continue
                if self.require_no_collision and collisions is not None:
                    if np.any(collisions[e, t0:t1]):
                        continue
                indices.append((e, t0))

        return indices

    def __iter__(self) -> Iterator:
        files = self._shard_files()
        rng = np.random.RandomState()

        for fpath in files:
            with h5py.File(fpath, "r") as h5f:
                indices = self._build_indices(h5f)
                if not indices:
                    continue
                rng.shuffle(indices)

                vis_ds = h5f["vision"]
                prop_ds = h5f["proprio"]
                cmd_ds = h5f["cmds"]
                done_ds = h5f["dones"] if "dones" in h5f else None
                col_ds = h5f["collisions"] if "collisions" in h5f else None

                for b0 in range(0, len(indices), self.batch_size):
                    batch_idx = indices[b0: b0 + self.batch_size]
                    B = len(batch_idx)
                    if B == 0:
                        continue

                    vis = np.empty((B, self.seq_len, 3, 64, 64), dtype=np.uint8)
                    prop = np.empty((B, self.seq_len, 47), dtype=np.float32)
                    cmds = np.empty((B, self.seq_len, 3), dtype=np.float32)
                    dones = np.zeros((B, self.seq_len), dtype=np.bool_)
                    collisions = np.zeros((B, self.seq_len), dtype=np.bool_)

                    for i, (e, t0) in enumerate(batch_idx):
                        t1 = t0 + self.seq_len
                        vis[i] = vis_ds[e, t0:t1]
                        prop[i] = prop_ds[e, t0:t1]
                        cmds[i] = cmd_ds[e, t0:t1]
                        if done_ds is not None:
                            dones[i] = done_ds[e, t0:t1]
                        if col_ds is not None:
                            collisions[i] = col_ds[e, t0:t1]

                    yield (
                        torch.from_numpy(vis),
                        torch.from_numpy(prop),
                        torch.from_numpy(cmds),
                        torch.from_numpy(dones),
                        torch.from_numpy(collisions),
                    )
