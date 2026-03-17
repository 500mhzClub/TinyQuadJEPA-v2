#!/usr/bin/env python3
"""Closed-loop MPC waypoint navigation eval for CanonicalJEPA.

Adapted from v1's 6_genesis_eval.py for the canonical student-teacher JEPA
architecture.  Key differences from v1:

- ``model.encode_online()`` for current state (predictor expects online space).
- ``model.encode_target()`` for goal states (energy head expects target space).
- Predictor output is already in target space — no additional projection needed.
- Optional obstacles via ``--with_obstacles``.
- All maths, checkpoint, genesis, texture, and obstacle helpers come from the
  ``tqjepa`` package instead of being inlined.

Route (default): W1 -> W2 -> W3 -> W2 -> W1 (repeating).
Beacon positions:
    RED   (2.0, 0.5, 0)   approach (-0.3, 0)
    GREEN (1.0, 2.5, 0)   approach ( 0, -0.3)
    BLUE  (3.0, 2.0, 0)   approach (-0.3, 0)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

import genesis as gs

from tqjepa.models import CanonicalJEPA, GoalEnergyHead, ActorCritic
from tqjepa.math_utils import (
    clamp,
    wrap_to_pi,
    yaw_to_quat,
    quat_to_yaw,
    body_to_world_xy,
    world_to_body_xy,
    forward_up_from_quat,
    world_to_body_vec,
)
from tqjepa.genesis_utils import init_genesis_once, to_genesis_target, to_numpy
from tqjepa.checkpoint_utils import load_jepa_checkpoint, load_ppo_checkpoint
from tqjepa.obstacle_utils import generate_random_layout, add_obstacles_to_scene
from tqjepa.texture_utils import make_checkerboard

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

WORLD_MIN = np.array([-1.0, -1.0], dtype=np.float32)
WORLD_MAX = np.array([4.5, 4.0], dtype=np.float32)

JOINTS_ACTUATED = [
    "lf_hip_joint",  "lh_hip_joint",  "rf_hip_joint",  "rh_hip_joint",
    "lf_thigh_joint","lh_thigh_joint","rf_thigh_joint","rh_thigh_joint",
    "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint",
]

Q0_VALUES = np.array([
    0.06,  0.06, -0.06, -0.06,
    0.85,  0.85,  0.85,  0.85,
   -1.75, -1.75, -1.75, -1.75,
], dtype=np.float32)

URDF_PATH = "assets/mini_pupper/mini_pupper.urdf"

# --------------------------------------------------------------------------- #
# Waypoint specification
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class WaypointSpec:
    pos: np.ndarray
    approach_dir_xy: np.ndarray
    name: str
    color_rgb: Tuple[float, float, float]
    panel_pos: Tuple[float, float, float]
    panel_size: Tuple[float, float, float]


def make_waypoints() -> List[WaypointSpec]:
    return [
        WaypointSpec(
            pos=np.array([2.0, 0.5, 0.0], dtype=np.float32),
            approach_dir_xy=np.array([-0.3, 0.0], dtype=np.float32),
            name="RED beacon",
            color_rgb=(0.94, 0.16, 0.16),
            panel_pos=(2.8, 0.5, 0.60),
            panel_size=(0.05, 1.20, 1.20),
        ),
        WaypointSpec(
            pos=np.array([1.0, 2.5, 0.0], dtype=np.float32),
            approach_dir_xy=np.array([0.0, -0.3], dtype=np.float32),
            name="GREEN beacon",
            color_rgb=(0.16, 0.86, 0.16),
            panel_pos=(1.0, 3.3, 0.60),
            panel_size=(1.20, 0.05, 1.20),
        ),
        WaypointSpec(
            pos=np.array([3.0, 2.0, 0.0], dtype=np.float32),
            approach_dir_xy=np.array([-0.3, 0.0], dtype=np.float32),
            name="BLUE beacon",
            color_rgb=(0.16, 0.31, 0.94),
            panel_pos=(3.8, 2.0, 0.60),
            panel_size=(0.05, 1.20, 1.20),
        ),
    ]


DEFAULT_ROUTE = [0, 1, 2, 1, 0]   # W1 -> W2 -> W3 -> W2 -> W1

# --------------------------------------------------------------------------- #
# Scene construction
# --------------------------------------------------------------------------- #

def build_scene(
    waypoints: List[WaypointSpec],
    with_obstacles: bool = False,
    obstacle_seed: int = 42,
):
    """Build Genesis scene with plane, robot, beacons, optional obstacles, cameras."""

    scene = gs.Scene(show_viewer=False)

    # Textured ground plane.
    checker_path = make_checkerboard(grid=16, path="eval_checker.png")
    scene.add_entity(
        gs.morphs.Plane(),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(image_path=checker_path),
        ),
    )

    # Robot.
    robot = scene.add_entity(
        gs.morphs.URDF(file=URDF_PATH, pos=(0.0, 0.0, 0.12), fixed=False),
    )

    # Beacon panels.
    for wp in waypoints:
        scene.add_entity(
            gs.morphs.Box(pos=wp.panel_pos, size=wp.panel_size, fixed=True),
            surface=gs.surfaces.Rough(color=wp.color_rgb),
        )

    # Optional obstacles.
    obstacle_layout = None
    if with_obstacles:
        obstacle_layout = generate_random_layout(
            n_range=(4, 8),
            spawn_radius=(0.8, 2.8),
            robot_clearance=0.50,
            seed=obstacle_seed,
        )
        add_obstacles_to_scene(scene, obstacle_layout)

    # Cameras.
    cam_brain = scene.add_camera(res=(64, 64), fov=50)
    cam_eye = scene.add_camera(res=(384, 384), fov=50)
    cam_overhead = scene.add_camera(res=(512, 512), fov=50)

    scene.build()

    # Resolve joint DOF indices.
    dofs = [robot.get_joint(jn).dofs_idx_local[0] for jn in JOINTS_ACTUATED]

    q0 = torch.tensor(Q0_VALUES, device=gs.device, dtype=torch.float32)

    # Initial robot pose.
    robot.set_pos(np.array([0.0, 0.0, 0.12], dtype=np.float32))
    robot.set_quat(yaw_to_quat(0.0))
    robot.set_dofs_position(Q0_VALUES, dofs)
    robot.set_dofs_kp(torch.ones(12, device=gs.device) * 5.0, dofs)
    robot.set_dofs_kv(torch.ones(12, device=gs.device) * 0.5, dofs)

    return scene, robot, cam_brain, cam_eye, cam_overhead, dofs, q0, obstacle_layout


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #

def get_sys1_obs(
    robot,
    q0: torch.Tensor,
    prev_action: torch.Tensor,
    cmd: torch.Tensor,
    dofs,
    dev: torch.device,
) -> torch.Tensor:
    """Build the 50-dim proprioceptive observation for the PPO policy.

    Layout (50):
        height(1), quat(4), vel_body(3), ang_body(3), q_rel(12), dq(12),
        prev_action(12), cmd(3)
    """
    pos = robot.get_pos().to(dev)
    quat = robot.get_quat().to(dev)
    vel = robot.get_vel().to(dev)
    ang = robot.get_ang().to(dev)

    # Ensure batch dimension.
    pos, quat, vel, ang = [
        x.unsqueeze(0) if x.dim() == 1 else x for x in (pos, quat, vel, ang)
    ]

    vel_b = world_to_body_vec(quat, vel)
    ang_b = world_to_body_vec(quat, ang)

    q = robot.get_dofs_position(dofs).to(dev).unsqueeze(0) if robot.get_dofs_position(dofs).dim() == 1 else robot.get_dofs_position(dofs).to(dev)
    dq = robot.get_dofs_velocity(dofs).to(dev).unsqueeze(0) if robot.get_dofs_velocity(dofs).dim() == 1 else robot.get_dofs_velocity(dofs).to(dev)

    obs = torch.cat([
        pos[:, 2:3],        # height (1)
        quat,               # orientation (4)
        vel_b,              # body-frame linear velocity (3)
        ang_b,              # body-frame angular velocity (3)
        q - q0.unsqueeze(0),  # joint position error (12)
        dq,                 # joint velocities (12)
        prev_action,        # previous action (12)
        cmd,                # velocity command (3)
    ], dim=1)
    return obs


@torch.no_grad()
def get_jepa_state(
    robot,
    cam_brain,
    q0: torch.Tensor,
    prev_action: torch.Tensor,
    dofs,
    dev: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render brain camera and build (vision, proprio) pair for the JEPA encoder."""
    img = cam_brain.render()[0]
    img = to_numpy(img)
    rgb = np.transpose(img[:, :, :3], (2, 0, 1)).copy()
    vision = torch.from_numpy(rgb).float().to(dev) / 255.0
    vision = vision.unsqueeze(0)

    # Proprio is the first 47 dims of sys1 obs (before prev_action + cmd).
    obs = get_sys1_obs(robot, q0, prev_action, torch.zeros((1, 3), device=dev), dofs, dev)
    proprio = obs[:, :47]
    return vision, proprio


# --------------------------------------------------------------------------- #
# Camera placement
# --------------------------------------------------------------------------- #

def update_cameras(robot, cam_brain, cam_eye, cam_overhead):
    """Position cameras relative to robot pose.  Returns pose tuples for HUD projection."""
    pos = to_numpy(robot.get_pos())
    quat = to_numpy(robot.get_quat())
    if pos.ndim > 1:
        pos = pos[0]
    if quat.ndim > 1:
        quat = quat[0]

    fw, up = forward_up_from_quat(quat)

    # Brain + eye: first-person.
    cp = pos + fw * 0.10 + up * 0.05
    lk = cp + fw * 1.0
    for c in (cam_brain, cam_eye):
        c.set_pose(pos=cp, lookat=lk, up=up)

    # Overhead: chase-cam style.
    c3p = pos - fw * 1.8 + np.array([0.0, 0.0, 0.8])
    c3l = pos + fw * 0.5
    c3u = np.array([0.0, 0.0, 1.0])
    cam_overhead.set_pose(pos=c3p, lookat=c3l, up=c3u)

    return cp, lk, up, c3p, c3l, c3u


# --------------------------------------------------------------------------- #
# Kinematic rollout (for geometric cost component)
# --------------------------------------------------------------------------- #

def rollout_cmd_kinematic(
    start_xy: np.ndarray,
    start_yaw: float,
    cmd_xyw: np.ndarray,
    horizon: int,
    dt: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Forward-simulate a constant body-frame command kinematically."""
    pos = np.array(start_xy, dtype=np.float32).copy()
    yaw = float(start_yaw)
    path = []
    for _ in range(horizon):
        path.append(pos.copy())
        world_v = body_to_world_xy(yaw, cmd_xyw[:2])
        pos += dt * world_v
        yaw = wrap_to_pi(yaw + dt * float(cmd_xyw[2]))
    return np.stack(path, axis=0), pos, yaw


# --------------------------------------------------------------------------- #
# World -> pixel projection (for HUD overlays)
# --------------------------------------------------------------------------- #

def project_world_to_pixel(
    wp: np.ndarray,
    cam_pos: np.ndarray,
    cam_lookat: np.ndarray,
    cam_up: np.ndarray,
    fov: float,
    w: int,
    h: int,
) -> Optional[Tuple[int, int]]:
    """Project a world point to pixel coordinates in a pinhole camera."""
    dist = np.linalg.norm(cam_lookat - cam_pos)
    f = (cam_lookat - cam_pos) / max(dist, 1e-8)
    s = np.cross(f, cam_up / max(np.linalg.norm(cam_up), 1e-8))
    sn = np.linalg.norm(s)
    if sn < 1e-5:
        return None
    s /= sn
    u = np.cross(s, f)

    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = s, u, -f
    view[0, 3] = -np.dot(s, cam_pos)
    view[1, 3] = -np.dot(u, cam_pos)
    view[2, 3] = np.dot(f, cam_pos)

    asp = w / h
    fy = 1.0 / np.tan(np.radians(fov) / 2.0)
    proj = np.zeros((4, 4))
    proj[0, 0] = fy / asp
    proj[1, 1] = fy
    proj[2, 2] = -1.0
    proj[2, 3] = -0.02
    proj[3, 2] = -1.0

    pt = np.array([wp[0], wp[1], wp[2], 1.0], dtype=np.float32)
    clip = proj @ view @ pt
    if clip[3] <= 0:
        return None
    ndc = clip[:3] / clip[3]
    return int((ndc[0] + 1.0) * 0.5 * w), int((1.0 - ndc[1]) * 0.5 * h)


# --------------------------------------------------------------------------- #
# MPC planner
# --------------------------------------------------------------------------- #

@torch.no_grad()
def plan_best_cmd(
    jepa: CanonicalJEPA,
    head: GoalEnergyHead,
    z_current: torch.Tensor,
    z_goal: torch.Tensor,
    robot_xy: np.ndarray,
    robot_yaw: float,
    goal_xy: np.ndarray,
    goal_body_xy: np.ndarray,
    dist_to_goal: float,
    heading_error: float,
    n_candidates: int,
    horizon: int,
    dev: torch.device,
    prev_cmd: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """CEM-style random-shooting planner.

    Scores candidate command sequences by:
        cost = energy(predictor_rollout, z_goal) + geometric_cost + smoothness_penalty
    """
    far = dist_to_goal > 0.9
    transl_scale = 0.30 if far else 0.18
    if abs(heading_error) > 0.9:
        transl_scale *= 0.45

    mean = torch.tensor([
        clamp(float(goal_body_xy[0]) * transl_scale, -0.35, 0.35),
        clamp(float(goal_body_xy[1]) * transl_scale, -0.20, 0.20),
        clamp(0.40 * heading_error, -0.45, 0.45),
    ], device=dev, dtype=torch.float32)

    std = torch.tensor([
        0.12 if far else 0.09,
        0.10 if far else 0.08,
        0.22 if far else 0.18,
    ], device=dev, dtype=torch.float32)

    best_cmd = mean.view(1, 3)
    best_path: Optional[np.ndarray] = None
    best_cost: Optional[float] = None
    best_energy: Optional[float] = None

    # CEM iterations.
    for _ in range(5):
        cmds = mean + std * torch.randn((n_candidates, 3), device=dev)
        cmds[:, 0].clamp_(-0.40, 0.40)
        cmds[:, 1].clamp_(-0.25, 0.25)
        cmds[:, 2].clamp_(-0.60, 0.60)

        # Predictor rollout: start from z_current (online space) and roll forward.
        # The predictor maps online -> target space, so its output is directly
        # comparable to z_goal (encoded with target_encoder).
        z_roll = z_current.expand(n_candidates, -1)
        h_t = torch.zeros((n_candidates, jepa.latent_dim), device=dev, dtype=z_roll.dtype)
        for _t in range(horizon):
            z_roll, h_t = jepa.predictor(z_roll, cmds, h_t)

        # Energy cost: lower = closer to goal in latent space.
        eng = head(z_roll, z_goal.expand_as(z_roll))

        # Geometric cost: kinematic rollout for spatial awareness.
        geo_cost = torch.empty((n_candidates,), device=dev, dtype=torch.float32)
        path_cache: List[np.ndarray] = []
        for i in range(n_candidates):
            cmd_np = cmds[i].detach().cpu().numpy()
            path_xy, end_xy, end_yaw = rollout_cmd_kinematic(
                robot_xy, robot_yaw, cmd_np, horizon,
            )
            path_cache.append(path_xy)
            end_dist = float(np.linalg.norm(end_xy - goal_xy))
            end_goal_angle = math.atan2(
                float(goal_xy[1] - end_xy[1]),
                float(goal_xy[0] - end_xy[0]),
            )
            end_heading_err = abs(wrap_to_pi(end_goal_angle - end_yaw))
            geo_cost[i] = 0.85 * end_dist + 0.10 * end_heading_err

        cost = eng + geo_cost
        if prev_cmd is not None:
            cost = cost + 0.10 * (cmds - prev_cmd).pow(2).sum(dim=-1)

        # CEM elite selection.
        k = max(n_candidates // 10, 8)
        elite_idx = torch.topk(cost, k=k, largest=False).indices
        elite_cmds = cmds[elite_idx]
        mean = elite_cmds.mean(dim=0)
        std = elite_cmds.std(dim=0) + 1e-4

        iter_best = int(torch.argmin(cost).item())
        iter_best_cost = float(cost[iter_best].item())
        if best_cost is None or iter_best_cost < best_cost:
            best_cost = iter_best_cost
            best_cmd = cmds[iter_best].view(1, 3).detach().clone()
            best_energy = float(eng[iter_best].item())
            best_path = path_cache[iter_best]

    assert best_path is not None and best_energy is not None and best_cost is not None
    return best_cmd, {"energy": best_energy, "path": best_path, "cost": best_cost}


# --------------------------------------------------------------------------- #
# Latent breadcrumb harvesting
# --------------------------------------------------------------------------- #

@torch.no_grad()
def harvest_breadcrumb(
    robot,
    cam_brain,
    q0: torch.Tensor,
    jepa: CanonicalJEPA,
    ppo: ActorCritic,
    dofs,
    dev: torch.device,
    scene,
    wp: WaypointSpec,
    n_avg: int = 5,
    warmup: int = 10,
    speed: float = 0.25,
    start_offset: float = 0.45,
) -> torch.Tensor:
    """Drive the robot near a beacon and encode the approach with target_encoder.

    The target encoder is used because the energy head compares predicted latents
    (which live in target space) against goal latents.
    """
    approach = wp.approach_dir_xy.astype(np.float32)
    norm = float(np.linalg.norm(approach))
    if norm > 1e-8:
        approach = approach / norm
    start_xy = wp.pos[:2] - start_offset * approach
    start_yaw = math.atan2(float(approach[1]), float(approach[0]))

    robot.set_pos(np.array([start_xy[0], start_xy[1], 0.12], dtype=np.float32))
    robot.set_quat(yaw_to_quat(start_yaw))
    robot.set_dofs_position(q0.detach().cpu().numpy(), dofs)
    for _ in range(8):
        scene.step()

    pa = torch.zeros((1, 12), device=dev)
    cmd = torch.tensor([[speed, 0.0, 0.0]], device=dev)

    # Walk forward for warmup steps to reach approach pose.
    for _ in range(warmup):
        obs = get_sys1_obs(robot, q0, pa, cmd, dofs, dev)
        pa = ppo.act_deterministic(obs)
        target = to_genesis_target(q0 + 0.3 * pa[0])
        robot.control_dofs_position(target, dofs)
        for _ in range(4):
            scene.step()

    # Collect and average latents at the approach pose.
    latents = []
    for _ in range(n_avg):
        obs = get_sys1_obs(robot, q0, pa, cmd, dofs, dev)
        pa = ppo.act_deterministic(obs)
        target = to_genesis_target(q0 + 0.3 * pa[0])
        robot.control_dofs_position(target, dofs)
        for _ in range(4):
            scene.step()
        v, p = get_jepa_state(robot, cam_brain, q0, pa, dofs, dev)
        # Use target_encoder: goal latents must be in target space.
        latents.append(jepa.encode_target(v, p).detach())

    return torch.stack(latents, dim=0).mean(dim=0)


# --------------------------------------------------------------------------- #
# HUD drawing helpers
# --------------------------------------------------------------------------- #

def _world_to_map_px(
    xy: np.ndarray,
    map_x0: int, map_y0: int, map_w: int, map_h: int,
) -> Tuple[int, int]:
    nx = (float(xy[0]) - float(WORLD_MIN[0])) / max(float(WORLD_MAX[0] - WORLD_MIN[0]), 1e-8)
    ny = (float(xy[1]) - float(WORLD_MIN[1])) / max(float(WORLD_MAX[1] - WORLD_MIN[1]), 1e-8)
    px = map_x0 + int(np.clip(nx, 0.0, 1.0) * map_w)
    py = map_y0 + map_h - int(np.clip(ny, 0.0, 1.0) * map_h)
    return px, py


def draw_energy_bar(
    draw: ImageDraw.ImageDraw,
    x: int, y: int, w: int, h: int,
    energy: float,
    label: str,
    color: Tuple[int, int, int] = (0, 200, 100),
):
    draw.rectangle([x, y, x + w, y + h], outline=(80, 80, 80), fill=(30, 30, 30))
    frac = max(0.0, min(1.0, energy / 4.0))
    bar_w = int(w * frac)
    if bar_w > 0:
        draw.rectangle([x, y, x + bar_w, y + h], fill=color)
    draw.text((x + w + 4, y), f"{label}: {energy:.2f}", fill=(200, 200, 200))


def draw_progress_bar(
    draw: ImageDraw.ImageDraw,
    x: int, y: int, w: int, h: int,
    frac: float,
    label: str,
    color: Tuple[int, int, int] = (0, 160, 255),
):
    frac = max(0.0, min(1.0, frac))
    draw.rectangle([x, y, x + w, y + h], outline=(90, 90, 90), fill=(30, 30, 30))
    if frac > 0:
        draw.rectangle([x, y, x + int(w * frac), y + h], fill=color)
    draw.text((x, y - 16), label, fill=(200, 200, 200))


def draw_minimap(
    draw: ImageDraw.ImageDraw,
    map_x0: int, map_y0: int, map_w: int, map_h: int,
    waypoints: List[WaypointSpec],
    route: List[int],
    route_ptr: int,
    robot_xy: np.ndarray,
    robot_yaw: float,
    trail: List[np.ndarray],
    plan_path: Optional[np.ndarray],
    visit_counts: Dict[int, int],
    dist_thresh: float,
):
    """Draw a minimap showing the world, waypoints, trail, and planned path."""
    draw.rectangle(
        [map_x0, map_y0, map_x0 + map_w, map_y0 + map_h],
        fill=(18, 18, 18), outline=(95, 95, 95),
    )

    # Grid lines.
    for gx in np.linspace(WORLD_MIN[0], WORLD_MAX[0], 7):
        p0 = _world_to_map_px(np.array([gx, WORLD_MIN[1]], dtype=np.float32), map_x0, map_y0, map_w, map_h)
        p1 = _world_to_map_px(np.array([gx, WORLD_MAX[1]], dtype=np.float32), map_x0, map_y0, map_w, map_h)
        draw.line([p0, p1], fill=(35, 35, 35), width=1)
    for gy in np.linspace(WORLD_MIN[1], WORLD_MAX[1], 6):
        p0 = _world_to_map_px(np.array([WORLD_MIN[0], gy], dtype=np.float32), map_x0, map_y0, map_w, map_h)
        p1 = _world_to_map_px(np.array([WORLD_MAX[0], gy], dtype=np.float32), map_x0, map_y0, map_w, map_h)
        draw.line([p0, p1], fill=(35, 35, 35), width=1)

    route_colors = [(255, 80, 80), (80, 255, 80), (80, 140, 255)]

    # Pixel-per-metre for goal radius ring.
    px_per_m_x = map_w / max(float(WORLD_MAX[0] - WORLD_MIN[0]), 1e-8)
    px_per_m_y = map_h / max(float(WORLD_MAX[1] - WORLD_MIN[1]), 1e-8)
    goal_r_px = max(3, int(dist_thresh * 0.5 * (px_per_m_x + px_per_m_y)))

    # Route backbone.
    if len(route) > 1:
        pts = [
            _world_to_map_px(waypoints[idx].pos[:2], map_x0, map_y0, map_w, map_h)
            for idx in route
        ]
        draw.line(pts, fill=(120, 120, 120), width=2)

    # Trail.
    if len(trail) > 1:
        trail_pts = [
            _world_to_map_px(t, map_x0, map_y0, map_w, map_h)
            for t in trail[-240:]
        ]
        if len(trail_pts) > 1:
            draw.line(trail_pts, fill=(255, 214, 10), width=2)

    # Planned path.
    if plan_path is not None and len(plan_path) > 1:
        plan_pts = [
            _world_to_map_px(pt, map_x0, map_y0, map_w, map_h)
            for pt in plan_path
        ]
        draw.line(plan_pts, fill=(0, 170, 255), width=3)

    # Waypoint markers.
    active_idx = route[route_ptr]
    for wi, wp in enumerate(waypoints):
        px, py = _world_to_map_px(wp.pos[:2], map_x0, map_y0, map_w, map_h)
        col = route_colors[wi % len(route_colors)]
        ring_col = (255, 255, 255) if wi == active_idx else (90, 90, 90)
        draw.ellipse(
            [px - goal_r_px, py - goal_r_px, px + goal_r_px, py + goal_r_px],
            outline=ring_col, width=2,
        )
        r = 8 if wi == active_idx else 6
        draw.ellipse([px - r, py - r, px + r, py + r], fill=col, outline=(230, 230, 230), width=1)
        draw.text((px + goal_r_px + 6, py - 8), f"W{wi + 1} x{visit_counts.get(wi, 0)}", fill=col)

    # Robot triangle.
    rx, ry = _world_to_map_px(robot_xy, map_x0, map_y0, map_w, map_h)
    head_dir = np.array([math.cos(robot_yaw), math.sin(robot_yaw)], dtype=np.float32)
    left_dir = np.array([math.cos(robot_yaw + 2.5), math.sin(robot_yaw + 2.5)], dtype=np.float32)
    right_dir = np.array([math.cos(robot_yaw - 2.5), math.sin(robot_yaw - 2.5)], dtype=np.float32)
    scale = 12.0
    tri = [
        (rx + int(head_dir[0] * scale), ry - int(head_dir[1] * scale)),
        (rx + int(left_dir[0] * scale * 0.8), ry - int(left_dir[1] * scale * 0.8)),
        (rx + int(right_dir[0] * scale * 0.8), ry - int(right_dir[1] * scale * 0.8)),
    ]
    draw.polygon(tri, fill=(255, 255, 255), outline=(10, 10, 10))
    draw.text((map_x0, map_y0 - 16), "World map", fill=(200, 200, 200))


def compose_hud_frame(
    overhead_img: np.ndarray,
    eye_img: np.ndarray,
    waypoints: List[WaypointSpec],
    route: List[int],
    route_ptr: int,
    target_wp_idx: int,
    step: int,
    dist: float,
    heading_error: float,
    raw_energy: float,
    ema_energy: float,
    plan_info: dict,
    cmd: torch.Tensor,
    robot_xy: np.ndarray,
    robot_yaw: float,
    trail: List[np.ndarray],
    visit_counts: Dict[int, int],
    energy_history: List[float],
    dist_thresh: float,
    c3p: np.ndarray,
    c3l: np.ndarray,
    c3u: np.ndarray,
    with_obstacles: bool,
) -> np.ndarray:
    """Compose the full HUD frame with overhead view, eye view, minimap, and stats."""

    # Overhead image with planner rollout overlay.
    p3 = Image.fromarray(overhead_img[:, :, :3].astype(np.uint8))
    d3 = ImageDraw.Draw(p3)

    # Overlay planned path on overhead view.
    h_px = [
        project_world_to_pixel(
            np.array([pt[0], pt[1], 0.05], dtype=np.float32),
            c3p, c3l, c3u, 50, 512, 512,
        )
        for pt in plan_info["path"]
    ]
    valid_px = [px for px in h_px if px is not None]
    if len(valid_px) > 1:
        d3.line(valid_px, fill=(0, 150, 255), width=4)

    # Overlay waypoint markers on overhead view.
    lm_colors = [(255, 80, 80), (80, 255, 80), (80, 130, 255)]
    completed = set(route[:route_ptr])
    for wi, wp in enumerate(waypoints):
        px = project_world_to_pixel(wp.pos, c3p, c3l, c3u, 50, 512, 512)
        if px is None:
            continue
        col = (100, 100, 100) if wi in completed else lm_colors[wi % len(lm_colors)]
        if wi == target_wp_idx:
            col = (255, 255, 255)
        d3.ellipse([px[0] - 12, px[1] - 12, px[0] + 12, px[1] + 12], outline=col, width=3)
        d3.text((px[0] + 16, px[1] - 6), f"W{wi + 1} x{visit_counts[wi]}", fill=col)

    # Eye image.
    pe = Image.fromarray(eye_img[:, :, :3].astype(np.uint8))

    # Canvas: header + overhead (512) + eye (384 side panel).
    header_h = 176
    canvas = Image.new("RGB", (896, header_h + 512), (20, 20, 20))
    canvas.paste(p3, (0, header_h))
    canvas.paste(pe, (512, header_h))

    drw = ImageDraw.Draw(canvas)

    # Header background.
    drw.rectangle([0, 0, 895, header_h - 1], fill=(10, 10, 10), outline=(55, 55, 55))
    drw.line([(0, header_h - 1), (895, header_h - 1)], fill=(90, 90, 90), width=2)

    # Panel labels.
    drw.rectangle([0, header_h, 511, header_h + 511], outline=(70, 70, 70), width=2)
    drw.rectangle([512, header_h, 895, header_h + 383], outline=(70, 70, 70), width=2)
    drw.text((12, header_h - 22), "World view + planner rollout", fill=(190, 190, 190))
    drw.text((524, header_h - 22), "Robot eye view", fill=(190, 190, 190))

    # Route string.
    route_str = " -> ".join([f"W{i + 1}" for i in route])

    # Text overlays.
    title = "CanonicalJEPA | MPC Navigation Eval"
    if with_obstacles:
        title += " [obstacles]"
    drw.text((20, 16), title, fill=(0, 255, 100))
    drw.text(
        (20, 36),
        f"Step: {step:04d} | Route target: {route_ptr + 1}/{len(route)} "
        f"| Active: W{target_wp_idx + 1} ({waypoints[target_wp_idx].name})",
        fill=(200, 200, 200),
    )
    drw.text((20, 56), f"Route: {route_str}", fill=(160, 160, 160))
    drw.text(
        (20, 76),
        f"Dist: {dist:.2f}m | Threshold: {dist_thresh:.2f}m "
        f"| Heading err: {np.degrees(heading_error):+.0f}deg",
        fill=(200, 200, 200),
    )
    drw.text(
        (20, 96),
        f"Raw E: {raw_energy:.2f} | EMA E: {ema_energy:.2f} "
        f"| Plan cost: {plan_info['cost']:.2f}",
        fill=(200, 200, 200),
    )
    drw.text(
        (20, 116),
        f"Cmd: vx={float(cmd[0, 0]):+.2f} vy={float(cmd[0, 1]):+.2f} "
        f"wz={float(cmd[0, 2]):+.2f}",
        fill=(200, 200, 200),
    )

    # Energy bars.
    draw_energy_bar(drw, 20, 140, 160, 12, raw_energy, "raw", color=(200, 80, 40))
    draw_energy_bar(drw, 20, 158, 160, 12, ema_energy, "ema", color=(0, 180, 120))
    draw_progress_bar(
        drw, 220, 146, 220, 12,
        route_ptr / max(len(route) - 1, 1),
        "Route completion", color=(120, 180, 255),
    )

    # Energy trace.
    if len(energy_history) > 2:
        hist = energy_history[-80:]
        hist_arr = np.asarray(hist, dtype=np.float32)
        h_min = float(hist_arr.min())
        h_max = max(float(hist_arr.max()), h_min + 0.1)
        pts = []
        for i, val in enumerate(hist_arr):
            x_px = 20 + int(i / len(hist_arr) * 420)
            y_px = 170 - int((float(val) - h_min) / (h_max - h_min) * 34)
            pts.append((x_px, y_px))
        if len(pts) > 1:
            drw.line(pts, fill=(0, 200, 255), width=2)
        drw.text((20, 124), "EMA energy history", fill=(180, 180, 180))

    # Minimap.
    draw_minimap(
        drw,
        map_x0=540, map_y0=22, map_w=322, map_h=134,
        waypoints=waypoints,
        route=route,
        route_ptr=route_ptr,
        robot_xy=robot_xy,
        robot_yaw=robot_yaw,
        trail=trail,
        plan_path=plan_info["path"],
        visit_counts=visit_counts,
        dist_thresh=dist_thresh,
    )

    # Footer legend.
    footer = "Blue = planned rollout | Yellow = actual trail | White ring = active goal"
    drw.text((540, 160), footer, fill=(120, 120, 120))

    return np.array(canvas)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Closed-loop MPC waypoint navigation eval for CanonicalJEPA.",
    )
    parser.add_argument("--jepa_ckpt", type=str, required=True,
                        help="Path to CanonicalJEPA checkpoint.")
    parser.add_argument("--head_ckpt", type=str, required=True,
                        help="Path to GoalEnergyHead checkpoint.")
    parser.add_argument("--ppo_ckpt", type=str, required=True,
                        help="Path to PPO ActorCritic checkpoint.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device: auto | cuda | cpu.")
    parser.add_argument("--sim_backend", type=str, default="auto",
                        help="Genesis backend: auto | gpu | cuda | cpu.")
    parser.add_argument("--n_steps", type=int, default=600,
                        help="Maximum number of navigation steps.")
    parser.add_argument("--n_candidates", type=int, default=10000,
                        help="Number of random candidate commands per planning step.")
    parser.add_argument("--horizon", type=int, default=15,
                        help="Predictor rollout horizon for planning.")
    parser.add_argument("--out", type=str, default="jepa_logs/eval_output.mp4",
                        help="Output video path.")
    parser.add_argument("--no_video", action="store_true",
                        help="Disable video recording.")
    parser.add_argument("--with_obstacles", action="store_true",
                        help="Add random obstacles to the scene.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    # ---- Reproducibility -------------------------------------------------- #
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Device ----------------------------------------------------------- #
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
    print(f"Torch device: {dev}")

    # ---- Validate checkpoints --------------------------------------------- #
    for label, path in [
        ("JEPA", args.jepa_ckpt),
        ("Energy head", args.head_ckpt),
        ("PPO", args.ppo_ckpt),
    ]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} checkpoint not found: {path}")
            sys.exit(1)

    # ---- Load models ------------------------------------------------------ #
    jepa = CanonicalJEPA().to(dev)
    jepa_sd, jepa_meta = load_jepa_checkpoint(args.jepa_ckpt, device=dev)
    jepa.load_state_dict(jepa_sd)
    jepa.eval()
    print(f"Loaded CanonicalJEPA from {args.jepa_ckpt}")

    head = GoalEnergyHead().to(dev)
    head_ckpt = torch.load(args.head_ckpt, map_location=dev)
    head_sd_key = "energy_head_state_dict" if "energy_head_state_dict" in head_ckpt else "model_state_dict"
    from tqjepa.checkpoint_utils import clean_state_dict
    head.load_state_dict(clean_state_dict(head_ckpt[head_sd_key]))
    head.eval()
    print(f"Loaded GoalEnergyHead from {args.head_ckpt}")

    ppo = ActorCritic().to(dev)
    ppo_sd = load_ppo_checkpoint(args.ppo_ckpt, device=dev)
    ppo.load_state_dict(ppo_sd, strict=False)
    ppo.eval()
    print(f"Loaded ActorCritic from {args.ppo_ckpt}")

    # ---- Build scene ------------------------------------------------------ #
    waypoints = make_waypoints()
    route = list(DEFAULT_ROUTE)
    dist_thresh = 0.3

    init_genesis_once(args.sim_backend)
    scene, robot, cam_brain, cam_eye, cam_overhead, dofs, q0, obstacle_layout = build_scene(
        waypoints,
        with_obstacles=args.with_obstacles,
        obstacle_seed=args.seed,
    )

    print(f"\nScene built: {len(waypoints)} beacons", end="")
    if obstacle_layout is not None:
        print(f", {len(obstacle_layout.obstacles)} obstacles", end="")
    print()

    # ---- Harvest latent breadcrumbs --------------------------------------- #
    print("\nHarvesting latent breadcrumbs (approach-aligned, averaged) ...")
    latent_breadcrumbs: List[torch.Tensor] = []
    for i, wp in enumerate(waypoints):
        print(f"  W{i + 1}: pos={wp.pos[:2]}  approach={wp.approach_dir_xy}")
        z_goal = harvest_breadcrumb(
            robot, cam_brain, q0, jepa, ppo, dofs, dev, scene, wp,
            n_avg=5, warmup=10,
        )
        latent_breadcrumbs.append(z_goal)
    print("  Done.\n")

    # ---- Reset robot to start -------------------------------------------- #
    robot.set_pos(np.array([0.0, 0.0, 0.12], dtype=np.float32))
    robot.set_quat(yaw_to_quat(0.0))
    robot.set_dofs_position(q0.detach().cpu().numpy(), dofs)
    for _ in range(20):
        scene.step()

    # ---- Video writer ----------------------------------------------------- #
    writer = None
    if not args.no_video:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        writer = imageio.get_writer(args.out, fps=30)

    # ---- Navigation state ------------------------------------------------- #
    prev_action = torch.zeros((1, 12), device=dev)
    prev_cmd: Optional[torch.Tensor] = None
    route_ptr = 0
    target_wp_idx = route[route_ptr]
    goal_xy = waypoints[target_wp_idx].pos[:2].copy()

    ema_energy = 0.0
    ema_alpha = 0.30
    energy_history: List[float] = []
    trail: List[np.ndarray] = []
    visit_counts: Dict[int, int] = {i: 0 for i in range(len(waypoints))}
    waypoint_arrival_steps: List[dict] = []

    route_str = " -> ".join([f"W{i + 1}" for i in route])
    print(f"Running MPC navigation ({args.n_steps} steps)")
    print(f"  Route:      {route_str}")
    print(f"  Candidates: {args.n_candidates}")
    print(f"  Horizon:    {args.horizon}")
    print(f"  Threshold:  {dist_thresh}m")
    print()

    t0 = time.time()

    # ---- Main navigation loop --------------------------------------------- #
    for step in range(args.n_steps):

        # Update cameras.
        cp, lk, up, c3p, c3l, c3u = update_cameras(
            robot, cam_brain, cam_eye, cam_overhead,
        )

        # Encode current state with online_encoder (predictor expects online space).
        vis, prop = get_jepa_state(robot, cam_brain, q0, prev_action, dofs, dev)
        with torch.no_grad():
            z_current = jepa.encode_online(vis, prop).detach()
            raw_energy = float(
                head(z_current, latent_breadcrumbs[target_wp_idx]).item()
            )

        ema_energy = ema_alpha * raw_energy + (1.0 - ema_alpha) * ema_energy
        energy_history.append(ema_energy)

        # Robot world state.
        rp = to_numpy(robot.get_pos())
        rq = to_numpy(robot.get_quat())
        if rp.ndim > 1:
            rp = rp[0]
        if rq.ndim > 1:
            rq = rq[0]

        trail.append(rp[:2].copy())
        yaw = quat_to_yaw(rq)
        dist = float(np.linalg.norm(rp[:2] - goal_xy))

        # ---- Check waypoint arrival -------------------------------------- #
        if dist < dist_thresh:
            visit_counts[target_wp_idx] += 1
            waypoint_arrival_steps.append({
                "route_idx": route_ptr,
                "waypoint": target_wp_idx,
                "step": step,
                "dist": dist,
                "ema_energy": ema_energy,
            })
            print(
                f"\n  [REACHED] Route {route_ptr + 1}/{len(route)} = "
                f"W{target_wp_idx + 1} at step {step}  "
                f"dist={dist:.2f}  ema_E={ema_energy:.2f}"
            )

            if route_ptr == len(route) - 1:
                print(f"\n  Full route complete at step {step}!")
                # Record one final video frame before exiting.
                if writer is not None:
                    overhead_raw = to_numpy(cam_overhead.render()[0])
                    eye_raw = to_numpy(cam_eye.render()[0])
                    frame = compose_hud_frame(
                        overhead_img=overhead_raw, eye_img=eye_raw,
                        waypoints=waypoints, route=route,
                        route_ptr=route_ptr, target_wp_idx=target_wp_idx,
                        step=step, dist=dist, heading_error=0.0,
                        raw_energy=raw_energy, ema_energy=ema_energy,
                        plan_info={"energy": raw_energy, "path": trail[-2:], "cost": 0.0},
                        cmd=torch.zeros((1, 3), device=dev),
                        robot_xy=rp[:2], robot_yaw=yaw,
                        trail=trail, visit_counts=visit_counts,
                        energy_history=energy_history,
                        dist_thresh=dist_thresh,
                        c3p=c3p, c3l=c3l, c3u=c3u,
                        with_obstacles=args.with_obstacles,
                    )
                    writer.append_data(frame)
                break

            route_ptr += 1
            target_wp_idx = route[route_ptr]
            goal_xy = waypoints[target_wp_idx].pos[:2].copy()
            ema_energy = 4.0
            prev_cmd = None
            dist = float(np.linalg.norm(rp[:2] - goal_xy))

        # ---- Plan -------------------------------------------------------- #
        goal_vec = goal_xy - rp[:2]
        goal_angle = math.atan2(float(goal_vec[1]), float(goal_vec[0]))
        heading_error = wrap_to_pi(goal_angle - yaw)
        goal_dir_world = goal_vec / max(float(np.linalg.norm(goal_vec)), 1e-8)
        goal_body = world_to_body_xy(yaw, goal_dir_world)

        cmd, plan_info = plan_best_cmd(
            jepa=jepa,
            head=head,
            z_current=z_current,
            z_goal=latent_breadcrumbs[target_wp_idx],
            robot_xy=rp[:2].copy(),
            robot_yaw=yaw,
            goal_xy=goal_xy.copy(),
            goal_body_xy=goal_body.copy(),
            dist_to_goal=dist,
            heading_error=heading_error,
            n_candidates=args.n_candidates,
            horizon=args.horizon,
            dev=dev,
            prev_cmd=prev_cmd,
        )
        prev_cmd = cmd.clone()

        # ---- Execute via PPO --------------------------------------------- #
        with torch.no_grad():
            obs = get_sys1_obs(robot, q0, prev_action, cmd, dofs, dev)
            prev_action = ppo.act_deterministic(obs).detach()
        target = to_genesis_target(q0 + 0.3 * prev_action[0])
        robot.control_dofs_position(target, dofs)
        for _ in range(4):
            scene.step()

        # ---- Record video frame ------------------------------------------ #
        if writer is not None:
            overhead_raw = to_numpy(cam_overhead.render()[0])
            eye_raw = to_numpy(cam_eye.render()[0])
            frame = compose_hud_frame(
                overhead_img=overhead_raw, eye_img=eye_raw,
                waypoints=waypoints, route=route,
                route_ptr=route_ptr, target_wp_idx=target_wp_idx,
                step=step, dist=dist, heading_error=heading_error,
                raw_energy=raw_energy, ema_energy=ema_energy,
                plan_info=plan_info, cmd=cmd,
                robot_xy=rp[:2], robot_yaw=yaw,
                trail=trail, visit_counts=visit_counts,
                energy_history=energy_history,
                dist_thresh=dist_thresh,
                c3p=c3p, c3l=c3l, c3u=c3u,
                with_obstacles=args.with_obstacles,
            )
            writer.append_data(frame)

        # ---- Console progress -------------------------------------------- #
        elapsed = time.time() - t0
        fps = (step + 1) / max(elapsed, 1e-8)
        print(
            f"\r  step={step:04d}/{args.n_steps} | "
            f"route={route_ptr + 1}/{len(route)} | "
            f"W{target_wp_idx + 1} | "
            f"dist={dist:.2f} | "
            f"ema_E={ema_energy:.2f} | "
            f"hdg={np.degrees(heading_error):+.0f}deg | "
            f"{fps:.1f} fps",
            end="",
        )

    # ---- Cleanup ---------------------------------------------------------- #
    if writer is not None:
        writer.close()
        print(f"\n\nVideo saved to {args.out}")

    elapsed = time.time() - t0
    route_complete = route_ptr == len(route) - 1 and dist < dist_thresh
    final_step = step  # noqa: F821 — guaranteed by loop

    # ---- Summary JSON ----------------------------------------------------- #
    summary = {
        "route": [i + 1 for i in route],
        "route_complete": route_complete,
        "route_ptr": route_ptr,
        "total_steps": final_step + 1,
        "max_steps": args.n_steps,
        "n_candidates": args.n_candidates,
        "horizon": args.horizon,
        "dist_thresh": dist_thresh,
        "with_obstacles": args.with_obstacles,
        "elapsed_sec": round(elapsed, 2),
        "visit_counts": {f"W{k + 1}": v for k, v in visit_counts.items()},
        "arrivals": waypoint_arrival_steps,
        "final_ema_energy": round(ema_energy, 4),
        "checkpoints": {
            "jepa": args.jepa_ckpt,
            "head": args.head_ckpt,
            "ppo": args.ppo_ckpt,
        },
    }

    summary_path = os.path.join(
        os.path.dirname(args.out) or ".", "eval_summary.json",
    )
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # ---- Console summary -------------------------------------------------- #
    print(f"\nNavigation summary:")
    print(f"  Route complete : {route_complete}")
    print(f"  Steps used     : {final_step + 1}/{args.n_steps}")
    print(f"  Elapsed        : {elapsed:.1f}s")
    print(f"  Visit counts   : " + ", ".join(
        f"W{i + 1}={visit_counts[i]}" for i in range(len(waypoints))
    ))
    for arr in waypoint_arrival_steps:
        print(
            f"    Route {arr['route_idx'] + 1}: "
            f"W{arr['waypoint'] + 1} at step {arr['step']} "
            f"(dist={arr['dist']:.2f}, E={arr['ema_energy']:.2f})"
        )


if __name__ == "__main__":
    main()
