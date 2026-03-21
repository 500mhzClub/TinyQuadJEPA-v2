#!/usr/bin/env python3
"""Collect trajectory data from a frozen PPO walking policy in Genesis.

This script runs N parallel environments for multiple chunks, each chunk with
a different random obstacle layout.  Collisions are detected via AABB overlap
(with margin) and colliding environments are reset to safe positions.

Output: one .npz per chunk in --out_dir, each containing:
    proprio   (n_envs, steps, 47)   noisy proprioceptive observation
    cmds      (n_envs, steps, 3)    velocity commands (vx, vy, wz)
    dones     (n_envs, steps)       episode termination flags
    base_pos  (n_envs, steps, 3)    world-frame base position
    base_quat (n_envs, steps, 4)    world-frame base orientation (wxyz)
    joint_pos (n_envs, steps, 12)   actuated joint positions
    collisions(n_envs, steps)       per-step collision flags
    obstacle_layout                 JSON string describing the obstacle layout
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import genesis as gs

from tqjepa.models.ppo import ActorCritic
from tqjepa.math_utils import world_to_body_vec, yaw_to_quat
from tqjepa.genesis_utils import init_genesis_once, to_genesis_target
from tqjepa.checkpoint_utils import load_ppo_checkpoint
from tqjepa.obstacle_utils import (
    generate_random_layout,
    add_obstacles_to_scene,
    detect_collisions,
    ObstacleLayout,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

JOINTS_ACTUATED = [
    "lf_hip_joint",  "lh_hip_joint",  "rf_hip_joint",  "rh_hip_joint",
    "lf_thigh_joint","lh_thigh_joint","rf_thigh_joint","rh_thigh_joint",
    "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint",
]

Q0_VALUES = [
    0.06,  0.06, -0.06, -0.06,   # hips (LF, LH, RF, RH)
    0.85,  0.85,  0.85,  0.85,   # thighs
   -1.75, -1.75, -1.75, -1.75,   # calves
]

URDF_PATH = "assets/mini_pupper/mini_pupper.urdf"
ROBOT_SPAWN = (0.0, 0.0, 0.12)
DEFAULT_SCENE_BATCH_ENVS = 32768

# --------------------------------------------------------------------------- #
# Simulation config
# --------------------------------------------------------------------------- #

@dataclass
class SimConfig:
    n_envs: int = 2048
    dt: float = 0.01
    substeps: int = 4
    decimation: int = 4
    kp: float = 5.0
    kv: float = 0.5
    action_scale: float = 0.30
    min_z: float = 0.05
    max_tilt: float = 1.0
    collision_margin: float = 0.15
    safe_clearance: float = 0.40

# --------------------------------------------------------------------------- #
# Ornstein-Uhlenbeck noise
# --------------------------------------------------------------------------- #

class OUNoiseBatched:
    """Batched Ornstein-Uhlenbeck process for correlated command exploration."""

    def __init__(
        self,
        n_envs: int,
        dim: int,
        device: str,
        theta: float = 0.15,
        sigma: float = 0.2,
    ):
        self.n_envs = n_envs
        self.dim = dim
        self.device = device
        self.theta = theta
        self.sigma = sigma
        self.state = torch.zeros((n_envs, dim), device=device)

    def step(self) -> torch.Tensor:
        noise = torch.randn((self.n_envs, self.dim), device=self.device)
        self.state = self.state - self.theta * self.state + self.sigma * noise
        return self.state

    def reset(self, env_ids: torch.Tensor) -> None:
        self.state[env_ids] = 0.0


# --------------------------------------------------------------------------- #
# Safe respawn
# --------------------------------------------------------------------------- #

def sample_safe_positions(
    n: int,
    layout: ObstacleLayout,
    clearance: float,
    spawn_range: float = 2.0,
    max_attempts: int = 200,
    device: str = "cpu",
) -> torch.Tensor:
    """Sample n (x, y) positions that are >= clearance from every obstacle.

    Falls back to the origin if too many attempts fail (should be rare given
    sensible obstacle densities).
    """
    positions = torch.zeros((n, 2), device=device)
    filled = 0
    attempts = 0

    while filled < n and attempts < max_attempts * n:
        batch_size = min(n - filled, 256)
        candidates = (torch.rand((batch_size, 2), device=device) * 2 - 1) * spawn_range
        colliding = detect_collisions(candidates, layout, margin=clearance)
        safe_mask = ~colliding
        safe_pts = candidates[safe_mask]
        take = min(safe_pts.shape[0], n - filled)
        if take > 0:
            positions[filled : filled + take] = safe_pts[:take]
            filled += take
        attempts += batch_size

    if filled < n:
        # Fallback: place remaining at origin (always cleared during layout gen).
        print(f"  [WARN] Could only find {filled}/{n} safe spawn points; "
              f"placing remainder at origin.")

    return positions


def load_frozen_policy(ckpt_path: str) -> ActorCritic:
    """Load the PPO actor-critic onto the current Genesis torch device."""
    model = ActorCritic(obs_dim=50, act_dim=12).to(gs.device)
    ppo_sd = load_ppo_checkpoint(ckpt_path, device=gs.device)
    model.load_state_dict(ppo_sd, strict=False)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect trajectory data from frozen PPO policy with obstacles."
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to PPO checkpoint.")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Timesteps per chunk.")
    parser.add_argument("--chunks", type=int, default=5,
                        help="Number of data chunks to collect.")
    parser.add_argument("--n_envs", type=int, default=2048,
                        help="Number of parallel environments.")
    parser.add_argument("--out_dir", type=str, default="jepa_raw_data",
                        help="Output directory for .npz chunks.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed.")
    parser.add_argument("--sim_backend", type=str, default="auto",
                        help="Genesis backend: auto | gpu | cuda | vulkan | metal | cpu.")
    parser.add_argument(
        "--scene_batch_envs",
        type=int,
        default=DEFAULT_SCENE_BATCH_ENVS,
        help=(
            "Maximum envs per Genesis scene build. Larger logical chunks are "
            "split across multiple scene batches to avoid solver size limits. "
            "Set to 0 to disable batching."
        ),
    )
    args = parser.parse_args()

    # Validate checkpoint exists.
    if not os.path.isfile(args.ckpt):
        print(f"ERROR: checkpoint not found: {args.ckpt}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SimConfig(n_envs=args.n_envs)
    scene_batch_envs = cfg.n_envs if args.scene_batch_envs <= 0 else min(args.scene_batch_envs, cfg.n_envs)
    n_scene_batches = math.ceil(cfg.n_envs / scene_batch_envs)

    # ---- reproducibility -------------------------------------------------- #
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\nPhysics rollout configuration:")
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  Envs       : {cfg.n_envs}")
    print(f"  Scene batch: {scene_batch_envs} ({n_scene_batches} scene(s)/chunk)")
    print(f"  Steps/chunk: {args.steps}")
    print(f"  Chunks     : {args.chunks}")
    print(f"  Output     : {out_dir.resolve()}")
    print(f"  Seed       : {args.seed}")
    print()

    # -------------------------------------------------------------------- #
    # Chunk loop: each chunk gets a fresh scene with different obstacles
    # -------------------------------------------------------------------- #
    for chunk_idx in range(args.chunks):
        chunk_seed = args.seed + chunk_idx
        print(f"--- Chunk {chunk_idx + 1}/{args.chunks} (seed={chunk_seed}) ---")

        # Generate a new random obstacle layout for this chunk.
        layout = generate_random_layout(seed=chunk_seed)
        print(f"  Obstacles: {len(layout.obstacles)}")

        # ---- Per-chunk state ---------------------------------------------- #
        d_proprio   = torch.zeros((cfg.n_envs, args.steps, 47), dtype=torch.float32, device="cpu")
        d_cmds      = torch.zeros((cfg.n_envs, args.steps, 3),  dtype=torch.float32, device="cpu")
        d_dones     = torch.zeros((cfg.n_envs, args.steps),      dtype=torch.bool,    device="cpu")
        d_base_pos  = torch.zeros((cfg.n_envs, args.steps, 3),  dtype=torch.float32, device="cpu")
        d_base_quat = torch.zeros((cfg.n_envs, args.steps, 4),  dtype=torch.float32, device="cpu")
        d_joint_pos = torch.zeros((cfg.n_envs, args.steps, 12), dtype=torch.float32, device="cpu")
        d_collisions = torch.zeros((cfg.n_envs, args.steps),     dtype=torch.bool,    device="cpu")

        t0 = time.time()
        total_resets = 0
        model = None
        try:
            # Genesis/Vulkan can hold onto large solver allocations even after a
            # scene is destroyed, so use a fresh runtime for every chunk.
            init_genesis_once(args.sim_backend)
            model = load_frozen_policy(args.ckpt)

            for batch_idx, env_start in enumerate(range(0, cfg.n_envs, scene_batch_envs)):
                env_end = min(env_start + scene_batch_envs, cfg.n_envs)
                batch_n_envs = env_end - env_start
                batch_slice = slice(env_start, env_end)

                print(
                    f"  Scene batch {batch_idx + 1}/{n_scene_batches}: "
                    f"envs {env_start}-{env_end - 1} ({batch_n_envs})"
                )

                scene = None
                try:
                    # Build a manageable Genesis scene for this env slice.
                    scene = gs.Scene(show_viewer=False)
                    scene.add_entity(gs.morphs.Plane())
                    robot = scene.add_entity(
                        gs.morphs.URDF(file=URDF_PATH, pos=ROBOT_SPAWN, fixed=False)
                    )
                    add_obstacles_to_scene(scene, layout)
                    scene.build(n_envs=batch_n_envs)

                    # ---- Joint indexing ----------------------------------- #
                    name_to_joint = {j.name: j for j in robot.joints}
                    missing = [jn for jn in JOINTS_ACTUATED if jn not in name_to_joint]
                    if missing:
                        print(f"ERROR: missing joints in URDF: {missing}")
                        sys.exit(1)

                    dof_idx = [list(name_to_joint[jn].dofs_idx_local)[0]
                               for jn in JOINTS_ACTUATED]
                    act_dofs = torch.tensor(dof_idx, device=gs.device, dtype=torch.int64)

                    q0 = torch.tensor(Q0_VALUES, device=gs.device, dtype=torch.float32)

                    robot.set_dofs_kp(torch.ones(12, device=gs.device) * cfg.kp, act_dofs)
                    robot.set_dofs_kv(torch.ones(12, device=gs.device) * cfg.kv, act_dofs)

                    # ---- Per-scene state --------------------------------- #
                    ou_noise = OUNoiseBatched(batch_n_envs, 3, gs.device)
                    latency_buffer = torch.zeros((2, batch_n_envs, 3), device=gs.device)
                    prev_a = torch.zeros((batch_n_envs, 12), device=gs.device)

                    for step in range(args.steps):
                        # -- Command generation (OU noise -> tanh -> scale) - #
                        raw_cmds = torch.tanh(ou_noise.step())
                        scaled_cmds = raw_cmds.clone()
                        scaled_cmds[:, 0] *= 0.40   # vx
                        scaled_cmds[:, 1] *= 0.25   # vy
                        scaled_cmds[:, 2] *= 0.60   # wz

                        # Two-step command latency buffer.
                        latency_buffer = torch.roll(latency_buffer, shifts=-1, dims=0)
                        latency_buffer[-1] = scaled_cmds
                        active_cmds = latency_buffer[0]

                        # -- Read robot state ------------------------------- #
                        pos = robot.get_pos()
                        quat = robot.get_quat()
                        vel_b = world_to_body_vec(quat, robot.get_vel())
                        ang_b = world_to_body_vec(quat, robot.get_ang())
                        q = robot.get_dofs_position(act_dofs)
                        dq = robot.get_dofs_velocity(act_dofs)
                        q_rel = q - q0.unsqueeze(0)

                        # 47-dim proprio: [height(1), quat(4), vel_body(3),
                        # ang_body(3), q_rel(12), dq(12), prev_action(12)]
                        proprio = torch.cat(
                            [pos[:, 2:3], quat, vel_b, ang_b, q_rel, dq, prev_a], dim=1
                        )

                        # -- Synthetic sensor noise ------------------------- #
                        noise = torch.randn_like(proprio) * 0.01
                        noise[:, 1:5] *= 2.0
                        noise[:, 5:11] *= 5.0
                        proprio_noisy = proprio + noise

                        # -- Forward pass through frozen policy ------------- #
                        obs = torch.cat([proprio_noisy, active_cmds], dim=1)
                        actions = model.act_deterministic(obs)
                        prev_a = actions.clone()

                        # -- Compute joint targets and step physics --------- #
                        q_tgt = q0.unsqueeze(0) + cfg.action_scale * actions
                        q_tgt[:, 0:4] = torch.clamp(q_tgt[:, 0:4], -0.8, 0.8)
                        q_tgt[:, 4:8] = torch.clamp(q_tgt[:, 4:8], -1.5, 1.5)
                        q_tgt[:, 8:12] = torch.clamp(q_tgt[:, 8:12], -2.5, -0.5)

                        robot.control_dofs_position(q_tgt, act_dofs)
                        for _ in range(cfg.decimation):
                            scene.step()

                        # -- Termination: falls and collisions -------------- #
                        fallen = pos[:, 2] < cfg.min_z
                        colliding = detect_collisions(
                            pos[:, :2], layout, margin=cfg.collision_margin
                        )
                        done = fallen | colliding

                        # -- Reset environments that are done --------------- #
                        done_ids = torch.nonzero(done).squeeze(-1)
                        if done_ids.numel() > 0:
                            total_resets += done_ids.numel()
                            ou_noise.reset(done_ids)
                            prev_a[done_ids] = 0.0

                            n_reset = done_ids.numel()
                            robot.set_dofs_position(
                                q0.unsqueeze(0).expand(n_reset, -1), act_dofs, envs_idx=done_ids
                            )
                            robot.set_dofs_velocity(
                                torch.zeros((n_reset, 12), device=gs.device), act_dofs, envs_idx=done_ids
                            )

                            safe_xy = sample_safe_positions(
                                n_reset,
                                layout,
                                clearance=cfg.safe_clearance,
                                device=gs.device,
                            )

                            respawn_pos = torch.zeros((n_reset, 3), device=gs.device)
                            respawn_pos[:, 0] = safe_xy[:, 0]
                            respawn_pos[:, 1] = safe_xy[:, 1]
                            respawn_pos[:, 2] = ROBOT_SPAWN[2]

                            yaw_angles = torch.rand(n_reset, device=gs.device) * 2 * math.pi
                            respawn_quat = torch.zeros((n_reset, 4), device=gs.device)
                            respawn_quat[:, 0] = torch.cos(yaw_angles * 0.5)
                            respawn_quat[:, 3] = torch.sin(yaw_angles * 0.5)

                            robot.set_pos(respawn_pos, envs_idx=done_ids, zero_velocity=True)
                            robot.set_quat(respawn_quat, envs_idx=done_ids, zero_velocity=False)

                        # -- Record data (CPU to stay in RAM) --------------- #
                        d_proprio[batch_slice, step] = proprio_noisy.cpu()
                        d_cmds[batch_slice, step] = scaled_cmds.cpu()
                        d_dones[batch_slice, step] = done.cpu()
                        d_base_pos[batch_slice, step] = pos.cpu()
                        d_base_quat[batch_slice, step] = quat.cpu()
                        d_joint_pos[batch_slice, step] = q.cpu()
                        d_collisions[batch_slice, step] = colliding.cpu()

                        # Progress reporting every 10%.
                        if (step + 1) % max(1, args.steps // 10) == 0:
                            elapsed = time.time() - t0
                            env_steps_done = env_start * args.steps + batch_n_envs * (step + 1)
                            fps = env_steps_done / elapsed
                            print(
                                f"  Batch {batch_idx + 1}/{n_scene_batches} | "
                                f"Step {step + 1:>6d}/{args.steps}  |  "
                                f"FPS: {fps:,.0f}  |  resets so far: {total_resets}"
                            )
                finally:
                    if scene is not None:
                        scene.destroy()
                        del scene
                        gc.collect()
        finally:
            if model is not None:
                del model
            if getattr(gs, "_initialized", False):
                gs.destroy()
            gc.collect()

        # -- Save chunk ----------------------------------------------------- #
        chunk_path = out_dir / f"chunk_{chunk_idx:04d}.npz"
        np.savez_compressed(
            str(chunk_path),
            proprio=d_proprio.numpy(),
            cmds=d_cmds.numpy(),
            dones=d_dones.numpy(),
            base_pos=d_base_pos.numpy(),
            base_quat=d_base_quat.numpy(),
            joint_pos=d_joint_pos.numpy(),
            collisions=d_collisions.numpy(),
            obstacle_layout=np.array(layout.to_json()),
        )

        chunk_elapsed = time.time() - t0
        chunk_fps = (cfg.n_envs * args.steps) / chunk_elapsed
        size_mb = chunk_path.stat().st_size / (1024 * 1024)
        print(f"  Saved {chunk_path} ({size_mb:.1f} MB)  |  "
              f"Chunk FPS: {chunk_fps:,.0f}  |  Total resets: {total_resets}\n")

    print("Physics rollout complete.")


if __name__ == "__main__":
    main()
