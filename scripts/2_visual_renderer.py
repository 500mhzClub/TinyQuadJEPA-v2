#!/usr/bin/env python3
"""
Visual renderer: renders egocentric 64x64 RGB from recorded physics rollouts.

Reads .npz chunk files produced by 1_physics_rollout.py, replays the recorded
trajectories in isolated Genesis render scenes (one per worker process), applies
visual domain randomization, and writes the final dataset to HDF5.

Key v2 changes over v1:
  - Imports from the tqjepa package (math_utils, genesis_utils, texture_utils,
    obstacle_utils) instead of inlining helpers.
  - Recreates obstacles in the render scene from the layout JSON stored in each
    .npz chunk, with randomized obstacle colors per worker.
  - Random ground texture per worker (selected from a procedurally generated
    texture set via texture_utils.generate_texture_set).
  - Stores collision flags in the HDF5 alongside vision/proprio/cmds/dones.
  - Expanded visual domain randomization: brightness, contrast, Gaussian noise,
    and a slight per-frame hue shift.

Usage:
    python scripts/2_visual_renderer.py --raw_dir jepa_raw_data --out_dir jepa_final_dataset --workers 4
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
from typing import List, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from tqjepa.math_utils import forward_up_from_quat
from tqjepa.genesis_utils import init_genesis_once, to_numpy
from tqjepa.texture_utils import generate_texture_set
from tqjepa.obstacle_utils import ObstacleLayout, ObstacleSpec

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

URDF_PATH = "assets/mini_pupper/mini_pupper.urdf"

JOINTS_ACTUATED = [
    "lf_hip_joint", "lh_hip_joint", "rf_hip_joint", "rh_hip_joint",
    "lf_thigh_joint", "lh_thigh_joint", "rf_thigh_joint", "rh_thigh_joint",
    "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint",
]

HIP_SPLAY = 0.06
THIGH0 = 0.85
CALF0 = -1.75

Q0_LIST = [
    HIP_SPLAY, HIP_SPLAY, -HIP_SPLAY, -HIP_SPLAY,
    THIGH0, THIGH0, THIGH0, THIGH0,
    CALF0, CALF0, CALF0, CALF0,
]

IMG_RES = 64
CAM_FOV = 58
CAM_FORWARD_OFFSET = 0.10  # metres
CAM_UP_OFFSET = 0.05       # metres
CAM_LOOKAT_DIST = 1.0      # metres ahead of camera position


# --------------------------------------------------------------------------- #
# Visual domain randomization
# --------------------------------------------------------------------------- #

def apply_visual_domain_randomization(rgb: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Apply per-frame visual domain randomization to an (H, W, 3) uint8 image.

    Augmentations:
      - brightness: uniform(-0.4, 0.4)
      - contrast:   uniform(0.5, 1.5)
      - gaussian noise: sigma in uniform(0.02, 0.08)
      - slight hue shift (applied in float space)
    """
    img = rgb.astype(np.float32) / 255.0

    # Brightness
    brightness = rng.uniform(-0.4, 0.4)
    img = img + brightness

    # Contrast (around per-channel mean)
    contrast = rng.uniform(0.5, 1.5)
    mean = img.mean(axis=(0, 1), keepdims=True)
    img = (img - mean) * contrast + mean

    # Gaussian noise
    sigma = rng.uniform(0.02, 0.08)
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    img = img + noise

    # Slight hue shift: rotate R/G/B channels by a small angle in colour space.
    # This is an approximate hue rotation using a Rodrigues-style 3x3 matrix
    # around the (1,1,1) axis.
    hue_angle = rng.uniform(-0.08, 0.08)  # radians
    cos_a = math.cos(hue_angle)
    sin_a = math.sin(hue_angle)
    one_third = 1.0 / 3.0
    sqrt_third = math.sqrt(one_third)
    # Rotation matrix around (1,1,1)/sqrt(3)
    hue_mat = np.array([
        [cos_a + one_third * (1 - cos_a),
         one_third * (1 - cos_a) - sqrt_third * sin_a,
         one_third * (1 - cos_a) + sqrt_third * sin_a],
        [one_third * (1 - cos_a) + sqrt_third * sin_a,
         cos_a + one_third * (1 - cos_a),
         one_third * (1 - cos_a) - sqrt_third * sin_a],
        [one_third * (1 - cos_a) - sqrt_third * sin_a,
         one_third * (1 - cos_a) + sqrt_third * sin_a,
         cos_a + one_third * (1 - cos_a)],
    ], dtype=np.float32)
    img = img @ hue_mat.T

    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


# --------------------------------------------------------------------------- #
# Render worker (runs in its own process for Vulkan isolation)
# --------------------------------------------------------------------------- #

def render_worker(args_tuple):
    """Each worker renders a subset of environments from one .npz chunk.

    Args (packed tuple):
        worker_id:        int    - worker index (for progress bars)
        chunk_file:       str    - path to the .npz recording
        start_env:        int    - first env index (inclusive)
        end_env:          int    - last env index (exclusive)
        tmp_file:         str    - path to write the worker's partial HDF5
        sim_backend:      str    - Genesis backend string
        texture_dir:      str    - directory containing generated textures
        obstacle_json:    str    - JSON string describing the obstacle layout
    """
    (worker_id, chunk_file, start_env, end_env,
     tmp_file, sim_backend, texture_dir, obstacle_json) = args_tuple

    N_subset = end_env - start_env

    # Skip if already fully rendered (fault tolerance)
    if os.path.exists(tmp_file):
        try:
            with h5py.File(tmp_file, "r") as f:
                if "vision" in f and f["vision"].shape[0] == N_subset:
                    return tmp_file
        except Exception:
            pass

    # ---- Genesis init (isolated Vulkan instance per process) ---- #
    import genesis as gs
    import torch

    init_genesis_once(sim_backend)

    scene = gs.Scene(
        show_viewer=False,
        vis_options=gs.options.VisOptions(
            plane_reflection=False,
            show_world_frame=False,
            show_link_frame=False,
            show_cameras=False,
        ),
        renderer=gs.renderers.Rasterizer(),
    )

    # ---- Ground plane with random texture ---- #
    rng = np.random.RandomState(worker_id + int.from_bytes(os.urandom(4), "little"))

    textures = generate_texture_set(texture_dir, count=10)
    tex_path = textures[rng.randint(0, len(textures))]

    scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(image_path=tex_path),
        ),
    )

    # ---- Recreate obstacles with random colours ---- #
    layout = ObstacleLayout.from_json(obstacle_json)
    for obs in layout.obstacles:
        # Randomize colour per worker so each render pass looks different
        r_col = float(np.clip(rng.uniform(0.3, 0.7) + rng.uniform(-0.1, 0.1), 0.1, 0.9))
        g_col = float(np.clip(rng.uniform(0.3, 0.7) + rng.uniform(-0.1, 0.1), 0.1, 0.9))
        b_col = float(np.clip(rng.uniform(0.3, 0.7) + rng.uniform(-0.1, 0.1), 0.1, 0.9))
        scene.add_entity(
            gs.morphs.Box(pos=obs.pos, size=obs.size, fixed=True),
            surface=gs.surfaces.Rough(color=(r_col, g_col, b_col)),
        )

    # ---- Robot ---- #
    robot = scene.add_entity(
        gs.morphs.URDF(file=URDF_PATH, fixed=False, merge_fixed_links=False),
    )

    # ---- Brain camera ---- #
    cam = scene.add_camera(res=(IMG_RES, IMG_RES), fov=CAM_FOV, GUI=False)

    scene.build(n_envs=1)

    # ---- Resolve actuated DOFs ---- #
    name_to_joint = {j.name: j for j in robot.joints}
    dof_idx = [list(name_to_joint[jn].dofs_idx_local)[0] for jn in JOINTS_ACTUATED]
    act_dofs = torch.tensor(dof_idx, device=gs.device, dtype=torch.int64)

    q0 = torch.tensor(Q0_LIST, device=gs.device, dtype=torch.float32)

    # ---- Load chunk data ---- #
    data = np.load(chunk_file, allow_pickle=True)
    T = data["base_pos"].shape[1]

    # ---- Render loop ---- #
    with h5py.File(tmp_file, "w") as f:
        h5_vision = f.create_dataset(
            "vision", (N_subset, T, 3, IMG_RES, IMG_RES), dtype="uint8", compression="gzip",
        )

        pbar = tqdm(
            range(start_env, end_env),
            desc=f"Worker {worker_id}",
            position=worker_id,
            leave=False,
        )

        for local_idx, env_idx in enumerate(pbar):
            base_pos_seq = torch.tensor(
                data["base_pos"][env_idx], device=gs.device, dtype=torch.float32,
            )
            base_quat_seq = torch.tensor(
                data["base_quat"][env_idx], device=gs.device, dtype=torch.float32,
            )
            joint_pos_seq = torch.tensor(
                data["joint_pos"][env_idx], device=gs.device, dtype=torch.float32,
            )

            env_video = np.zeros((T, 3, IMG_RES, IMG_RES), dtype=np.uint8)

            for step in range(T):
                base_pos = base_pos_seq[step].unsqueeze(0)
                base_quat = base_quat_seq[step].unsqueeze(0)

                # Set robot state
                robot.set_pos(base_pos)
                robot.set_quat(base_quat)
                robot.set_dofs_position(joint_pos_seq[step].unsqueeze(0), act_dofs)

                scene.step(update_visualizer=False)

                # ---- Camera placement ---- #
                # forward_up_from_quat expects a 1-D (4,) wxyz quaternion
                q_np = to_numpy(base_quat_seq[step])
                fw, up = forward_up_from_quat(q_np)

                pos_np = to_numpy(base_pos_seq[step])
                cam_pos = pos_np + CAM_FORWARD_OFFSET * fw + CAM_UP_OFFSET * up
                cam_lookat = cam_pos + CAM_LOOKAT_DIST * fw

                cam.set_pose(
                    pos=cam_pos,
                    lookat=cam_lookat,
                    up=up,
                )

                # ---- Render ---- #
                render_out = cam.render(rgb=True)
                rgb = render_out[0]
                if hasattr(rgb, "cpu"):
                    rgb = rgb.cpu().numpy()
                rgb = np.asarray(rgb, dtype=np.uint8)

                # ---- Visual domain randomization ---- #
                rgb = apply_visual_domain_randomization(rgb, rng)

                # (H, W, 3) -> (3, H, W)
                env_video[step] = np.transpose(rgb, (2, 0, 1))

            h5_vision[local_idx] = env_video

    return tmp_file


# --------------------------------------------------------------------------- #
# Stitch worker outputs into one HDF5
# --------------------------------------------------------------------------- #

def stitch_hdf5(
    out_path: str,
    tmp_files: List[str],
    tasks: list,
    data: dict,
    N: int,
    T: int,
) -> None:
    """Merge per-worker HDF5 shards and raw data into one final HDF5 file."""
    with h5py.File(out_path, "w") as h5f:
        # Vision: (N, T, 3, 64, 64) uint8
        h5_vision = h5f.create_dataset(
            "vision", (N, T, 3, IMG_RES, IMG_RES),
            dtype="uint8",
            chunks=(1, T, 3, IMG_RES, IMG_RES),
            compression="gzip",
        )

        # Proprio: (N, T, 47) float32
        h5f.create_dataset("proprio", data=data["proprio"], compression="gzip")

        # Cmds: (N, T, 3) float32
        h5f.create_dataset("cmds", data=data["cmds"], compression="gzip")

        # Dones: (N, T) bool
        h5f.create_dataset("dones", data=data["dones"], compression="gzip")

        # Collisions: (N, T) bool
        if "collisions" in data:
            h5f.create_dataset("collisions", data=data["collisions"], compression="gzip")
        else:
            # Fallback: store all-False if missing from raw data
            h5f.create_dataset(
                "collisions",
                data=np.zeros((N, T), dtype=bool),
                compression="gzip",
            )

        # Copy rendered vision from each worker shard
        for tmp_file, task in zip(tmp_files, tasks):
            start, end = task[2], task[3]
            try:
                with h5py.File(tmp_file, "r") as tmp_in:
                    h5_vision[start:end] = tmp_in["vision"][:]
                os.remove(tmp_file)
            except Exception as e:
                print(f"[stitch] Failed to merge {tmp_file}: {e}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Render egocentric 64x64 RGB from recorded physics rollouts.",
    )
    parser.add_argument(
        "--raw_dir", type=str, default="jepa_raw_data",
        help="Directory containing .npz chunk files from 1_physics_rollout.py",
    )
    parser.add_argument(
        "--out_dir", type=str, default="jepa_final_dataset",
        help="Output directory for final HDF5 dataset files",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel Genesis render processes (each gets its own Vulkan instance)",
    )
    parser.add_argument(
        "--sim_backend", type=str, default="auto",
        help="Genesis simulation backend: auto, gpu, cuda, cpu",
    )
    args = parser.parse_args()

    raw_files = sorted(glob.glob(os.path.join(args.raw_dir, "chunk_*.npz")))
    if not raw_files:
        print(f"No raw data found in {args.raw_dir}/. Run 1_physics_rollout.py first.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # Pre-generate the texture set once (shared across all workers)
    texture_dir = os.path.join(args.out_dir, "_textures")
    print(f"Generating ground texture set in {texture_dir} ...")
    generate_texture_set(texture_dir, count=10)

    # ---- Process each chunk ---- #
    for file_path in raw_files:
        chunk_name = os.path.basename(file_path).split(".")[0]
        out_path = os.path.join(args.out_dir, f"{chunk_name}.h5")

        # Skip already-complete chunks
        if os.path.exists(out_path):
            try:
                with h5py.File(out_path, "r") as h5f:
                    if "vision" in h5f and "collisions" in h5f:
                        print(f"Skipping {chunk_name} (already complete at {out_path})")
                        continue
            except Exception:
                print(f"Warning: {out_path} is corrupted, will overwrite.")

        print(f"\nRendering {chunk_name} -> {out_path}")

        data = np.load(file_path, allow_pickle=True)
        N, T = data["base_pos"].shape[:2]

        # Extract obstacle layout JSON from the chunk
        if "obstacle_layout" in data:
            obs_layout_raw = data["obstacle_layout"]
            # numpy stores strings as 0-d arrays sometimes
            if hasattr(obs_layout_raw, "item"):
                obstacle_json = str(obs_layout_raw.item())
            else:
                obstacle_json = str(obs_layout_raw)
        else:
            # No obstacles recorded -- use an empty layout
            obstacle_json = "[]"

        # Split environments across workers
        envs_per_worker = math.ceil(N / args.workers)
        tasks = []
        tmp_files = []

        for i in range(args.workers):
            start = i * envs_per_worker
            end = min(start + envs_per_worker, N)
            if start >= end:
                break

            tmp = os.path.join(args.out_dir, f"{chunk_name}_tmp_{i}.h5")
            tmp_files.append(tmp)
            tasks.append((
                i,
                file_path,
                start,
                end,
                tmp,
                args.sim_backend,
                texture_dir,
                obstacle_json,
            ))

        print(f"Spawning {len(tasks)} isolated Vulkan render processes ...")

        with mp.Pool(len(tasks)) as pool:
            list(tqdm(
                pool.imap_unordered(render_worker, tasks),
                total=len(tasks),
                desc="Workers",
                position=args.workers + 1,
            ))

        # ---- Stitch worker outputs into final HDF5 ---- #
        print(f"Stitching {len(tmp_files)} worker shards into {out_path} ...")
        stitch_hdf5(out_path, tmp_files, tasks, data, N, T)
        print(f"Chunk {chunk_name} complete ({N} envs, {T} steps).")

    print("\nAll chunks rendered successfully.")


if __name__ == "__main__":
    main()
