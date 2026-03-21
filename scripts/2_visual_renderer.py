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
  - Random ground textures from an expanded procedural texture bank, with
    multiple textured scene variants reused across environments in each worker.
  - Stores collision flags in the HDF5 alongside vision/proprio/cmds/dones.
  - Expanded visual domain randomization: brightness, contrast, Gaussian noise,
    and a slight per-frame hue shift.

Usage:
    python scripts/2_visual_renderer.py --raw_dir jepa_raw_data --out_dir jepa_final_dataset --workers 4
"""
from __future__ import annotations

import argparse
import glob
import math
import multiprocessing as mp
import queue
import os
import sys
from typing import List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import h5py
import numpy as np
from tqdm import tqdm

from tqjepa.math_utils import forward_up_from_quat
from tqjepa.genesis_utils import init_genesis_once, to_numpy
from tqjepa.texture_utils import generate_texture_set
from tqjepa.obstacle_utils import ObstacleLayout

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

URDF_PATH = "assets/mini_pupper/mini_pupper.urdf"

JOINTS_ACTUATED = [
    "lf_hip_joint", "lh_hip_joint", "rf_hip_joint", "rh_hip_joint",
    "lf_thigh_joint", "lh_thigh_joint", "rf_thigh_joint", "rh_thigh_joint",
    "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint",
]

IMG_RES = 64
CAM_FOV = 58
CAM_FORWARD_OFFSET = 0.10  # metres
CAM_UP_OFFSET = 0.05       # metres
CAM_LOOKAT_DIST = 1.0      # metres ahead of camera position
DEFAULT_TEXTURE_COUNT = 27
DEFAULT_TEXTURE_VARIANTS_PER_WORKER = 4
VULKAN_SAFE_WORKER_LIMIT = 4
VULKAN_SAFE_TEXTURE_VARIANT_LIMIT = 1


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


def sample_obstacle_color(rng: np.random.RandomState) -> Tuple[float, float, float]:
    base = rng.uniform(0.3, 0.7)
    tint = rng.uniform(-0.1, 0.1, size=3)
    color = np.clip(base + tint, 0.1, 0.9)
    return (float(color[0]), float(color[1]), float(color[2]))


def build_render_bundle(gs, torch, texture_path: str, layout: ObstacleLayout, rng: np.random.RandomState):
    """Construct one textured render scene variant."""
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

    scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(image_path=texture_path),
        ),
    )

    for obs in layout.obstacles:
        scene.add_entity(
            gs.morphs.Box(pos=obs.pos, size=obs.size, fixed=True),
            surface=gs.surfaces.Rough(color=sample_obstacle_color(rng)),
        )

    robot = scene.add_entity(
        gs.morphs.URDF(file=URDF_PATH, fixed=False, merge_fixed_links=False),
    )
    cam = scene.add_camera(res=(IMG_RES, IMG_RES), fov=CAM_FOV, GUI=False)
    scene.build(n_envs=1)

    name_to_joint = {j.name: j for j in robot.joints}
    dof_idx = [list(name_to_joint[jn].dofs_idx_local)[0] for jn in JOINTS_ACTUATED]
    act_dofs = torch.tensor(dof_idx, device=gs.device, dtype=torch.int64)

    return {
        "scene": scene,
        "robot": robot,
        "cam": cam,
        "act_dofs": act_dofs,
        "texture_path": texture_path,
    }


# --------------------------------------------------------------------------- #
# Render worker (runs in its own process for Vulkan isolation)
# --------------------------------------------------------------------------- #

def render_worker(args_tuple):
    """Each worker renders a subset of environments from one .npz chunk.

    Args (packed tuple):
        worker_id:        int        - worker index
        chunk_file:       str        - path to the .npz recording
        start_env:        int        - first env index (inclusive)
        end_env:          int        - last env index (exclusive)
        tmp_file:         str        - path to write the worker's partial HDF5
        sim_backend:      str        - Genesis backend string
        texture_dir:      str        - directory containing generated textures
        obstacle_json:    str        - JSON string describing the obstacle layout
        texture_count:    int        - number of generated textures to draw from
        texture_variants: int        - number of textured scene variants per worker
        progress_queue:   mp.Queue   - receives 1 per completed env for the main-process bar
    """
    (worker_id, chunk_file, start_env, end_env,
     tmp_file, sim_backend, texture_dir, obstacle_json,
     texture_count, texture_variants, progress_queue) = args_tuple

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
    import logging as _logging

    init_genesis_once(sim_backend, logging_level=_logging.ERROR)
    texture_count = max(1, int(texture_count))
    texture_variants = max(1, int(texture_variants))
    worker_seed = worker_id + int.from_bytes(os.urandom(4), "little")
    worker_rng = np.random.RandomState(worker_seed)

    texture_paths = sorted(glob.glob(os.path.join(texture_dir, "*.png")))
    if len(texture_paths) < texture_count:
        texture_paths = generate_texture_set(texture_dir, count=texture_count)
    else:
        texture_paths = texture_paths[:texture_count]
    if not texture_paths:
        raise RuntimeError(f"No textures available in {texture_dir}")

    layout = ObstacleLayout.from_json(obstacle_json)
    variant_count = max(1, min(texture_variants, len(texture_paths)))
    variant_ids = worker_rng.choice(len(texture_paths), size=variant_count, replace=False)
    bundles = [
        build_render_bundle(
            gs,
            torch,
            texture_paths[int(texture_idx)],
            layout,
            np.random.RandomState(worker_seed + 1009 * (variant_offset + 1)),
        )
        for variant_offset, texture_idx in enumerate(variant_ids)
    ]
    env_bundle_ids = worker_rng.randint(0, len(bundles), size=N_subset)

    # ---- Load chunk data ---- #
    data = np.load(chunk_file, allow_pickle=True)
    T = data["base_pos"].shape[1]

    # ---- Render loop ---- #
    with h5py.File(tmp_file, "w") as f:
        h5_vision = f.create_dataset(
            "vision", (N_subset, T, 3, IMG_RES, IMG_RES), dtype="uint8", compression="gzip",
        )

        for local_idx, env_idx in enumerate(range(start_env, end_env)):
            bundle = bundles[int(env_bundle_ids[local_idx])]
            scene = bundle["scene"]
            robot = bundle["robot"]
            cam = bundle["cam"]
            act_dofs = bundle["act_dofs"]
            env_rng = np.random.RandomState(worker_seed + env_idx * 7919 + 17)

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
                rgb = apply_visual_domain_randomization(rgb, env_rng)

                # (H, W, 3) -> (3, H, W)
                env_video[step] = np.transpose(rgb, (2, 0, 1))

            h5_vision[local_idx] = env_video
            progress_queue.put(1)

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
    """Merge per-worker HDF5 shards and raw data into one final HDF5 file.

    Writes to a temporary file first, then renames atomically so a crash
    during stitching never leaves a corrupt output file.
    """
    tmp_out = out_path + ".stitching"
    try:
        with h5py.File(tmp_out, "w") as h5f:
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
                except Exception as e:
                    print(f"[stitch] Failed to merge {tmp_file}: {e}")
                    raise

        # Atomic rename: only replace out_path once fully written
        os.replace(tmp_out, out_path)

        # Clean up worker shards only after successful rename
        for tmp_file in tmp_files:
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except Exception as e:
                print(f"[stitch] Warning: could not remove {tmp_file}: {e}")

    except Exception:
        # Leave tmp_out in place so the user can inspect it; remove if empty/corrupt
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except Exception:
                pass
        raise


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
        help="Genesis simulation backend: auto, gpu, cuda, vulkan, metal, cpu",
    )
    parser.add_argument(
        "--texture_count", type=int, default=DEFAULT_TEXTURE_COUNT,
        help="How many procedurally generated ground textures to create and sample from.",
    )
    parser.add_argument(
        "--texture_variants_per_worker", type=int, default=DEFAULT_TEXTURE_VARIANTS_PER_WORKER,
        help="How many textured scene variants each worker keeps alive for per-environment reuse.",
    )
    parser.add_argument(
        "--unsafe_vulkan_parallelism",
        action="store_true",
        help=(
            "Disable the conservative Vulkan worker/scene caps. "
            "Useful only if you have already validated higher process counts."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render chunks that already have a complete output file.",
    )
    args = parser.parse_args()

    effective_workers = max(1, int(args.workers))
    effective_texture_variants = max(1, int(args.texture_variants_per_worker))

    if args.sim_backend.lower().strip() == "vulkan" and not args.unsafe_vulkan_parallelism:
        capped_workers = min(effective_workers, VULKAN_SAFE_WORKER_LIMIT)
        capped_variants = min(
            effective_texture_variants,
            VULKAN_SAFE_TEXTURE_VARIANT_LIMIT,
        )
        if (capped_workers, capped_variants) != (effective_workers, effective_texture_variants):
            tqdm.write(
                "Vulkan safety caps applied: "
                f"workers {effective_workers}->{capped_workers}, "
                f"texture_variants {effective_texture_variants}->{capped_variants}."
            )
        effective_workers = capped_workers
        effective_texture_variants = capped_variants

    raw_files = sorted(glob.glob(os.path.join(args.raw_dir, "chunk_*.npz")))
    if not raw_files:
        tqdm.write(f"No raw data found in {args.raw_dir}/. Run 1_physics_rollout.py first.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # Pre-generate the texture set once (shared across all workers)
    texture_dir = os.path.join(args.out_dir, "_textures")
    tqdm.write(f"Generating ground texture set ({args.texture_count} textures) ...")
    generate_texture_set(texture_dir, count=args.texture_count)

    n_chunks = len(raw_files)

    # ---- Count total envs across all chunks for the top-level bar ---- #
    chunk_meta = []  # list of (file_path, chunk_name, N, T)
    for file_path in raw_files:
        chunk_name = os.path.basename(file_path).split(".")[0]
        d = np.load(file_path, allow_pickle=True)
        chunk_meta.append((file_path, chunk_name, int(d["base_pos"].shape[0]), int(d["base_pos"].shape[1])))

    # Determine which chunks need rendering (respects --force and skip logic)
    pending = []
    skipped = 0
    for file_path, chunk_name, N, T in chunk_meta:
        out_path = os.path.join(args.out_dir, f"{chunk_name}_rgb.h5")
        if not args.force and os.path.exists(out_path):
            try:
                with h5py.File(out_path, "r") as h5f:
                    if "vision" in h5f and "collisions" in h5f:
                        skipped += 1
                        continue
            except Exception:
                pass  # corrupted — will overwrite
        pending.append((file_path, chunk_name, N, T))

    if skipped:
        tqdm.write(f"Skipping {skipped}/{n_chunks} already-complete chunk(s).  Pass --force to re-render.")

    if not pending:
        tqdm.write("Nothing to render.")
        return

    total_envs = sum(N for _, _, N, _ in pending)

    # Single progress bar across all pending chunks
    with tqdm(
        total=total_envs,
        unit="env",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} envs  [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:

        for chunk_idx, (file_path, chunk_name, N, T) in enumerate(pending):
            out_path = os.path.join(args.out_dir, f"{chunk_name}_rgb.h5")

            # ---- Cleanup stale tmp files from a previous crashed run ---- #
            stale_tmps = sorted(glob.glob(os.path.join(args.out_dir, f"{chunk_name}_tmp_*.h5")))
            for stale in stale_tmps:
                try:
                    os.remove(stale)
                except Exception as e:
                    tqdm.write(f"Warning: could not remove stale {os.path.basename(stale)}: {e}")
            if stale_tmps:
                tqdm.write(f"Removed {len(stale_tmps)} stale shard(s) from previous run.")

            # Remove stale .stitching temp file if present
            stitching_tmp = out_path + ".stitching"
            if os.path.exists(stitching_tmp):
                try:
                    os.remove(stitching_tmp)
                except Exception:
                    pass

            chunk_label = f"{chunk_name}  [{chunk_idx + 1}/{len(pending)}]"
            pbar.set_description(chunk_label)

            data = np.load(file_path, allow_pickle=True)

            if "obstacle_layout" in data:
                obs_layout_raw = data["obstacle_layout"]
                obstacle_json = str(obs_layout_raw.item() if hasattr(obs_layout_raw, "item") else obs_layout_raw)
            else:
                obstacle_json = "[]"

            # Split environments across workers
            envs_per_worker = math.ceil(N / effective_workers)
            tasks = []
            tmp_files = []
            for i in range(effective_workers):
                start = i * envs_per_worker
                end = min(start + envs_per_worker, N)
                if start >= end:
                    break
                tmp = os.path.join(args.out_dir, f"{chunk_name}_tmp_{i}.h5")
                tmp_files.append(tmp)
                tasks.append((
                    i, file_path, start, end, tmp,
                    args.sim_backend, texture_dir, obstacle_json,
                    args.texture_count, effective_texture_variants,
                ))

            # Per-env progress queue: each worker sends 1 after every env it finishes
            progress_queue = mp.Queue()
            tasks_with_q = [(*t, progress_queue) for t in tasks]

            processes = [mp.Process(target=render_worker, args=(t,)) for t in tasks_with_q]
            for p in processes:
                p.start()

            # Drain queue until all N envs are accounted for, with a liveness check
            envs_received = 0
            while envs_received < N:
                try:
                    progress_queue.get(timeout=5)
                    pbar.update(1)
                    envs_received += 1
                except queue.Empty:
                    if not any(p.is_alive() for p in processes):
                        break  # all workers exited (possibly with error)

            for p in processes:
                p.join()

            # ---- Stitch worker outputs into final HDF5 ---- #
            pbar.set_description(f"{chunk_label}  stitching...")
            stitch_hdf5(out_path, tmp_files, tasks, data, N, T)
            tqdm.write(f"  {chunk_name} done  ({N} envs, {T} steps)  ->  {out_path}")

    tqdm.write("All chunks rendered.")


if __name__ == "__main__":
    main()
