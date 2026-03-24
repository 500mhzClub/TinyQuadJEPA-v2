#!/usr/bin/env python3
"""
JEPA v2 — Perceptual Maze Explorer  (no pre-loaded map)

Fully perception-driven variant of 7_maze_explorer.py.  No maze geometry is
pre-seeded into the occupancy grid; walls are discovered entirely from the
onboard depth camera using a persistent inverse-sensor occupancy scheme:

  • Free space  — any cell a depth ray passes *through* is marked free.
  • Occupied    — a cell accumulates stronger positive evidence from depth-hit
                  endpoints, while free traversals apply weaker negative
                  evidence so observed walls persist unless the camera keeps
                  seeing through them consistently.

Occupied-hit updates are restricted to the upper ~66 % of the depth image
(rows that correspond to roughly horizontal or upward gaze) to suppress
ground-return false positives that appear in the downward-facing lower rows.

Everything else — JEPA encoding, energy-head beacon detection, CEM planning,
breadcrumb harvesting — is identical to 7_maze_explorer.py.

Usage:
    python scripts/8_maze_explorer_perceptual.py \\
        --jepa_ckpt jepa_checkpoints/epoch_17.pt \\
        --head_ckpt energy_head_checkpoints/energy_head_best.pt \\
        --ppo_ckpt  models/ppo/ckpt_20000.pt
"""
from __future__ import annotations

import argparse
import heapq
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
import genesis as gs

from tqjepa.models import CanonicalJEPA, GoalEnergyHead, ActorCritic
from tqjepa.math_utils import (
    clamp, wrap_to_pi, yaw_to_quat, body_to_world_xy,
    world_to_body_xy, forward_up_from_quat, world_to_body_vec,
)
from tqjepa.genesis_utils import init_genesis_once, to_genesis_target, to_numpy
from tqjepa.checkpoint_utils import load_jepa_checkpoint, load_ppo_checkpoint, clean_state_dict
from tqjepa.texture_utils import make_checkerboard


# --------------------------------------------------------------------------- #
# World constants
# --------------------------------------------------------------------------- #

WORLD_MIN = np.array([-2.2, -1.2], dtype=np.float32)
WORLD_MAX = np.array([ 3.8,  3.8], dtype=np.float32)

MAP_UNKNOWN, MAP_FREE, MAP_OCC = -1, 0, 1

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
URDF_PATH  = "assets/mini_pupper/mini_pupper.urdf"
ROBOT_SPAWN = (0.5, -0.5, 0.12)

# Detection thresholds
DETECT_DIST  = 2.2   # metres
DETECT_FOV   = 0.90  # radians half-angle (~52°)
ARRIVE_DIST  = 0.45  # metres — waypoint claimed
MAX_SEEK_STEPS = 2200
DETECT_ENERGY_THRESH  = -1.20
DETECT_ENERGY_MARGIN  = 0.50
DETECT_CONFIRM_STEPS  = 8
GLIMPSE_ENERGY_THRESH = -0.55
GLIMPSE_ENERGY_MARGIN = 0.15
GLIMPSE_CONFIRM_STEPS = 4
DETECT_STRIDE         = 4
GLIMPSE_COOLDOWN      = 45
PROXY_ROUTE_RADIUS    = 0.45
BREADCRUMB_YAW_OFFSETS_DEG = (-90.0, -55.0, -28.0, -12.0, 0.0, 12.0, 28.0, 55.0, 90.0)
LATENT_MEMORY_MAX = 512
LATENT_MEMORY_STRIDE = 4
LATENT_MEMORY_MIN_STEP_DIST = 0.10
LATENT_NOVELTY_WEIGHT = 1.10
PLACE_MEMORY_CELL_M = 0.28
PLACE_MEMORY_YAW_BINS = 12
PLACE_LATENT_EMA_DECAY = 0.80

# ── Perceptual OCC parameters ──────────────────────────────────────────── #
MIN_OCC_DEPTH   = 0.25  # ignore depth endpoints closer than this (body / feet)
ROBOT_SELF_FREE_RADIUS = 0.08   # don't erase nearby walls with an oversized free bubble
ROBOT_CLEARANCE_RADIUS = 0.14   # conservative footprint used for LOS / reachability tests
FREE_RAY_CLEARANCE    = 0.12    # stop free carving before the depth hit
OCC_INFLATION_RADIUS  = 0.08    # inflate perceived walls from depth only

BRAIN_CAM_FWD_OFFSET = 0.01     # keep the brain camera well inside the body collision hull
BRAIN_CAM_UP_OFFSET  = 0.09
BRAIN_CAM_FOV_DEG    = 58.0

DEPTH_OCC_FLOOR_MARGIN_FRAC   = 0.04
DEPTH_GUARD_FLOOR_MARGIN_FRAC = 0.03

FRONT_STOP_DIST        = 0.28
FRONT_BLOCKED_FRAC     = 0.35
FRONT_MAP_CONFIRM_DIST = 0.40
FRONT_MAP_SAMPLE_RADIUS = 0.05
FRONT_GUARD_CENTER_LO  = 0.42
FRONT_GUARD_CENTER_HI  = 0.58
SEEK_ANCHOR_OFFSET     = 0.35
SEEK_STALL_WINDOW      = 55
SEEK_STALL_DISP        = 0.05
SEEK_STALL_PROGRESS    = 0.03
SEEK_STALL_DEPTH_DELTA = 0.05
SEEK_RECOVERY_COOLDOWN = 30
LINE_CHECK_STEP_M      = 0.05
DETECTION_CLEARANCE_RADIUS = 0.08
FREE_LOG_ODDS_DEC      = 2
OCC_LOG_ODDS_INC       = 2
LOG_ODDS_MIN           = -24
LOG_ODDS_MAX           = 24
LOG_ODDS_OCC_THRESH    = 10
LOG_ODDS_FREE_THRESH   = -2
OCC_HIT_CLUSTER_EPS_M  = 0.10
OCC_HIT_MIN_ROWS       = 4
OCC_HIT_MIN_FRAC       = 0.16
DETECT_WALL_MARGIN_M   = 0.18
UNKNOWN_TRAVERSE_COST  = 2.6
VISIT_TRAVERSE_COST    = 0.10
PATH_VISIT_PENALTY     = 0.18
FRONTIER_VISIT_PENALTY = 0.35


# --------------------------------------------------------------------------- #
# Maze wall + waypoint definitions
# --------------------------------------------------------------------------- #

MAZE_WALL_SPECS: List[Tuple] = [
    ((0.0, -0.3,  0.4), (0.24, 1.8, 0.8), (0.55, 0.55, 0.60)),
    ((0.0,  2.7,  0.4), (0.24, 2.2, 0.8), (0.55, 0.55, 0.60)),
    ((2.0, -0.3,  0.4), (0.24, 1.8, 0.8), (0.55, 0.55, 0.60)),
    ((2.0,  2.7,  0.4), (0.24, 2.2, 0.8), (0.55, 0.55, 0.60)),
    ((1.4,  2.8,  0.4), (1.2, 0.24, 0.8), (0.55, 0.55, 0.60)),
]


@dataclass(frozen=True)
class MazeWaypoint:
    name:         str
    pos:          np.ndarray
    approach_dir: np.ndarray
    panel_pos:    Tuple
    panel_size:   Tuple
    color_rgb:    Tuple


def make_maze_waypoints() -> List[MazeWaypoint]:
    return [
        MazeWaypoint(
            name="W1-RED",
            pos=np.array([-1.4,  0.2], dtype=np.float32),
            approach_dir=np.array([-1.0,  0.0], dtype=np.float32),
            panel_pos=(-2.15,  0.2,  0.55),
            panel_size=(0.12, 0.9, 1.1),
            color_rgb=(0.92, 0.15, 0.15),
        ),
        MazeWaypoint(
            name="W2-GREEN",
            pos=np.array([-1.4,  2.8], dtype=np.float32),
            approach_dir=np.array([-1.0,  0.0], dtype=np.float32),
            panel_pos=(-2.15,  2.8,  0.55),
            panel_size=(0.12, 0.9, 1.1),
            color_rgb=(0.15, 0.88, 0.15),
        ),
        MazeWaypoint(
            name="W3-BLUE",
            pos=np.array([ 3.2,  2.8], dtype=np.float32),
            approach_dir=np.array([ 1.0,  0.0], dtype=np.float32),
            panel_pos=( 3.75,  2.8,  0.55),
            panel_size=(0.12, 0.9, 1.1),
            color_rgb=(0.15, 0.30, 0.92),
        ),
        MazeWaypoint(
            name="W4-YELLOW",
            pos=np.array([ 3.2,  0.2], dtype=np.float32),
            approach_dir=np.array([ 1.0,  0.0], dtype=np.float32),
            panel_pos=( 3.75,  0.2,  0.55),
            panel_size=(0.12, 0.9, 1.1),
            color_rgb=(0.95, 0.90, 0.10),
        ),
        MazeWaypoint(
            name="W5-PURPLE",
            pos=np.array([ 1.0,  3.4], dtype=np.float32),
            approach_dir=np.array([ 0.0,  1.0], dtype=np.float32),
            panel_pos=( 1.0,  3.78, 0.55),
            panel_size=(0.9, 0.12, 1.1),
            color_rgb=(0.80, 0.10, 0.90),
        ),
    ]


# --------------------------------------------------------------------------- #
# Scene construction
# --------------------------------------------------------------------------- #

def build_scene(waypoints: List[MazeWaypoint]):
    scene = gs.Scene(show_viewer=False)

    tex = make_checkerboard(grid=12, path="maze_checker.png")
    scene.add_entity(
        gs.morphs.Plane(),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(image_path=tex),
        ),
    )

    robot = scene.add_entity(
        gs.morphs.URDF(file=URDF_PATH, pos=ROBOT_SPAWN, fixed=False),
    )

    for (cpos, csz, ccol) in MAZE_WALL_SPECS:
        scene.add_entity(
            gs.morphs.Box(pos=cpos, size=csz, fixed=True),
            surface=gs.surfaces.Rough(color=ccol),
        )

    for wp in waypoints:
        scene.add_entity(
            gs.morphs.Box(pos=wp.panel_pos, size=wp.panel_size, fixed=True),
            surface=gs.surfaces.Rough(color=wp.color_rgb),
        )

    cam_brain = scene.add_camera(res=(64, 64),   fov=BRAIN_CAM_FOV_DEG)
    cam_eye   = scene.add_camera(res=(384, 384), fov=BRAIN_CAM_FOV_DEG)
    cam_over  = scene.add_camera(res=(512, 512), fov=55)

    scene.build()

    dofs = [robot.get_joint(jn).dofs_idx_local[0] for jn in JOINTS_ACTUATED]
    q0   = torch.tensor(Q0_VALUES, device=gs.device, dtype=torch.float32)

    robot.set_pos(np.array(ROBOT_SPAWN, dtype=np.float32))
    robot.set_quat(yaw_to_quat(math.pi / 2))
    robot.set_dofs_position(Q0_VALUES, dofs)
    robot.set_dofs_kp(torch.ones(12, device=gs.device) * 5.0, dofs)
    robot.set_dofs_kv(torch.ones(12, device=gs.device) * 0.5, dofs)

    return scene, robot, cam_brain, cam_eye, cam_over, dofs, q0


# --------------------------------------------------------------------------- #
# Observation helpers
# --------------------------------------------------------------------------- #

def move_cams(robot, cam_brain, cam_eye, cam_over):
    p = to_numpy(robot.get_pos())
    q = to_numpy(robot.get_quat())
    if p.ndim > 1: p = p[0]
    if q.ndim > 1: q = q[0]
    fw, up = forward_up_from_quat(q)
    brain_pos = p + fw * BRAIN_CAM_FWD_OFFSET + up * BRAIN_CAM_UP_OFFSET
    brain_lk  = brain_pos + fw * 1.0
    cam_brain.set_pose(pos=brain_pos, lookat=brain_lk, up=up)
    cam_eye.set_pose(pos=brain_pos, lookat=brain_lk, up=up)
    over_pos = p - fw * 1.8 + np.array([0.0, 0.0, 1.0], dtype=np.float32)
    over_lk  = p + fw * 0.45
    cam_over.set_pose(pos=over_pos, lookat=over_lk,
                      up=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    yaw = math.atan2(float(fw[1]), float(fw[0]))
    pitch = math.asin(clamp(-float(fw[2]), -1.0, 1.0))
    return p, yaw, brain_pos, pitch


def render_rgb(cam) -> np.ndarray:
    out = cam.render()
    arr = None
    if isinstance(out, (tuple, list)):
        for item in out:
            a = np.asarray(to_numpy(item))
            if a.ndim == 3 and a.shape[-1] >= 3:
                arr = a[..., :3]; break
    elif isinstance(out, dict):
        for k in ("rgb", "color", "image"):
            if k in out:
                arr = np.asarray(out[k])[..., :3]; break
    else:
        arr = np.asarray(to_numpy(out))[..., :3]
    if arr is None:
        raise RuntimeError("Camera render failed")
    if arr.dtype != np.uint8:
        mx = float(np.nanmax(arr)) if arr.size else 1.0
        arr = np.clip(arr * (255.0 / mx if mx > 1.0 else 255.0), 0, 255).astype(np.uint8)
    return arr


def render_rgb_depth(cam) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    rgb = None; depth = None
    for fn in (lambda: cam.render(rgb=True, depth=True),
               lambda: cam.render(depth=True),
               lambda: cam.render()):
        try:
            out = fn(); break
        except Exception:
            pass
    else:
        raise RuntimeError("Camera render failed entirely")

    def absorb(a):
        nonlocal rgb, depth
        a = np.asarray(a)
        if a.ndim == 3 and a.shape[-1] >= 3 and rgb is None:
            rgb = a[..., :3].copy(); return
        if (a.ndim == 2 or (a.ndim == 3 and a.shape[-1] == 1)) and depth is None:
            depth = (a[..., 0] if a.ndim == 3 else a).copy(); return

    if isinstance(out, dict):
        for k in ("rgb","color","image"): out.get(k) is not None and absorb(out[k])
        out.get("depth") is not None and absorb(out["depth"])
    elif isinstance(out, (tuple, list)):
        for item in out:
            if item is not None: absorb(to_numpy(item))
    else:
        absorb(to_numpy(out))

    if rgb is None: raise RuntimeError("No RGB from camera")
    if rgb.dtype != np.uint8:
        mx = float(np.nanmax(rgb)) if rgb.size else 1.0
        rgb = np.clip(rgb * (255.0/mx if mx > 1.0 else 255.0), 0, 255).astype(np.uint8)
    if depth is not None:
        depth = np.asarray(depth, dtype=np.float32)
    return rgb, depth


def get_sys1_obs(robot, q0, prev_action, cmd, dofs, dev):
    pos  = robot.get_pos().to(dev); quat = robot.get_quat().to(dev)
    vel  = robot.get_vel().to(dev); ang  = robot.get_ang().to(dev)
    pos, quat, vel, ang = [x.unsqueeze(0) if x.dim()==1 else x
                           for x in (pos, quat, vel, ang)]
    q  = robot.get_dofs_position(dofs).to(dev)
    dq = robot.get_dofs_velocity(dofs).to(dev)
    q  = q.unsqueeze(0) if q.dim()==1 else q
    dq = dq.unsqueeze(0) if dq.dim()==1 else dq
    return torch.cat([
        pos[:, 2:3], quat,
        world_to_body_vec(quat, vel), world_to_body_vec(quat, ang),
        q - q0.unsqueeze(0), dq, prev_action, cmd,
    ], dim=1)


@torch.no_grad()
def get_jepa_state(robot, cam_brain, q0, prev_action, dofs, dev):
    rgb, depth = render_rgb_depth(cam_brain)
    chw = np.transpose(rgb[:, :, :3], (2, 0, 1)).copy()
    vis = torch.from_numpy(chw).float().to(dev).unsqueeze(0) / 255.0
    obs = get_sys1_obs(robot, q0, prev_action,
                       torch.zeros((1, 3), device=dev), dofs, dev)
    prop = obs[:, :47]
    return vis, prop, rgb, depth


# --------------------------------------------------------------------------- #
# Sensor map  —  persistent occupancy from perception only
# --------------------------------------------------------------------------- #

@dataclass
class SensorMap:
    grid:        np.ndarray   # MAP_UNKNOWN / MAP_FREE / MAP_OCC
    free_visits: np.ndarray   # cumulative free-ray traversals per cell
    log_odds:    np.ndarray   # asymmetric free/occ evidence accumulator
    res:         float

    @property
    def h(self): return int(self.grid.shape[0])
    @property
    def w(self): return int(self.grid.shape[1])


def make_sensor_map(res: float) -> SensorMap:
    w = int(math.ceil((WORLD_MAX[0]-WORLD_MIN[0]) / res))
    h = int(math.ceil((WORLD_MAX[1]-WORLD_MIN[1]) / res))
    return SensorMap(
        np.full((h, w), MAP_UNKNOWN, np.int8),
        np.zeros((h, w), np.int32),
        np.zeros((h, w), np.int16),
        float(res),
    )


def world_to_grid(sm, xy):
    gx = int((float(xy[0])-float(WORLD_MIN[0])) / sm.res)
    gy = int((float(xy[1])-float(WORLD_MIN[1])) / sm.res)
    return (gy, gx) if 0<=gx<sm.w and 0<=gy<sm.h else None

def grid_to_world(sm, rc):
    r,c = int(rc[0]), int(rc[1])
    return np.array([float(WORLD_MIN[0])+(c+0.5)*sm.res,
                     float(WORLD_MIN[1])+(r+0.5)*sm.res], dtype=np.float32)


def _refresh_grid_cell(sm, r, c, prev_cell=None):
    prev = int(sm.grid[r, c]) if prev_cell is None else int(prev_cell)
    lo = int(sm.log_odds[r, c])
    if lo >= LOG_ODDS_OCC_THRESH:
        sm.grid[r, c] = MAP_OCC
    elif lo <= LOG_ODDS_FREE_THRESH or (
        sm.free_visits[r, c] > 0 and (lo <= 0 or prev == MAP_FREE)
    ):
        sm.grid[r, c] = MAP_FREE
    else:
        sm.grid[r, c] = MAP_UNKNOWN


def mark_disc(sm, xy, radius, value):
    g = world_to_grid(sm, xy)
    if g is None: return
    rr = max(1, int(radius/sm.res))
    r0, c0 = g
    for r in range(max(0,r0-rr), min(sm.h,r0+rr+1)):
        for c in range(max(0,c0-rr), min(sm.w,c0+rr+1)):
            p = grid_to_world(sm, (r,c))
            if float(np.linalg.norm(p - xy[:2])) <= radius:
                prev_cell = int(sm.grid[r, c])
                if value == MAP_FREE:
                    sm.free_visits[r, c] += 1
                    sm.log_odds[r, c] = max(
                        LOG_ODDS_MIN,
                        int(sm.log_odds[r, c]) - FREE_LOG_ODDS_DEC,
                    )
                    _refresh_grid_cell(sm, r, c, prev_cell)
                elif value == MAP_OCC:
                    sm.log_odds[r, c] = min(
                        LOG_ODDS_MAX,
                        int(sm.log_odds[r, c]) + OCC_LOG_ODDS_INC,
                    )
                    _refresh_grid_cell(sm, r, c, prev_cell)

def sample_cell(sm, xy):
    g = world_to_grid(sm, xy)
    return MAP_OCC if g is None else int(sm.grid[g[0],g[1]])


def sample_occ_with_clearance(sm, xy, radius=ROBOT_CLEARANCE_RADIUS) -> bool:
    g = world_to_grid(sm, xy)
    if g is None:
        return True
    rr = max(0, int(math.ceil(radius / sm.res)))
    r0, c0 = g
    for r in range(max(0, r0 - rr), min(sm.h, r0 + rr + 1)):
        for c in range(max(0, c0 - rr), min(sm.w, c0 + rr + 1)):
            p = grid_to_world(sm, (r, c))
            if float(np.linalg.norm(p - xy[:2])) <= radius and sm.grid[r, c] == MAP_OCC:
                return True
    return False


def sample_front_occ_hits(sm, robot_xy, robot_yaw,
                          dists=(0.10, 0.16, 0.22, 0.30),
                          radius=FRONT_MAP_SAMPLE_RADIUS):
    fwd = np.array([math.cos(robot_yaw), math.sin(robot_yaw)], np.float32)
    return [
        sample_occ_with_clearance(sm, robot_xy + float(dist) * fwd, radius=radius)
        for dist in dists
    ]


def sample_traversable_with_clearance(sm, xy, radius=ROBOT_CLEARANCE_RADIUS, allow_unknown=False) -> bool:
    g = world_to_grid(sm, xy)
    if g is None:
        return False
    rr = max(0, int(math.ceil(radius / sm.res)))
    r0, c0 = g
    for r in range(max(0, r0 - rr), min(sm.h, r0 + rr + 1)):
        for c in range(max(0, c0 - rr), min(sm.w, c0 + rr + 1)):
            p = grid_to_world(sm, (r, c))
            if float(np.linalg.norm(p - xy[:2])) > radius:
                continue
            cell = int(sm.grid[r, c])
            if cell == MAP_OCC:
                return False
            if not allow_unknown and cell != MAP_FREE:
                return False
    return True


def coverage_percent(sm):
    return 100.0 * float(np.count_nonzero(sm.grid != MAP_UNKNOWN)) / float(sm.grid.size)


def normalize_depth_image(depth_img, depth_max):
    if depth_img is None:
        return None
    d = np.asarray(depth_img, np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if not np.isfinite(d).any():
        return None
    d = np.where(np.isfinite(d), d, np.nan)
    mx, mn = float(np.nanmax(d)), float(np.nanmin(d))
    d = np.clip(d, 0, 1) * depth_max if mx <= 1.05 and mn >= 0 else np.clip(d, 0, depth_max)
    d = np.nan_to_num(d, nan=depth_max, posinf=depth_max).astype(np.float32)
    # Genesis/Vulkan reverse-Z: 0.0 means "no geometry" (sky / beyond range).
    # Only filter these near-zero no-hit pixels; keep real close-range wall
    # returns so that front_blocked_from_depth can detect them correctly.
    d[d < 0.02] = depth_max
    return d


def projected_floor_row_fraction(cam_pitch_rad, cam_height_m, depth_max, fov_deg):
    """Normalized image row where the floor at depth_max first becomes visible."""
    if cam_height_m is None or cam_height_m <= 1e-4:
        return 1.0
    pitch = 0.0 if cam_pitch_rad is None else float(cam_pitch_rad)
    vfov_rad = math.radians(max(float(fov_deg), 1e-3))
    floor_drop = math.atan2(max(float(cam_height_m), 1e-4), max(float(depth_max), 1e-3))
    return clamp(0.5 + (floor_drop + pitch) / vfov_rad, 0.0, 1.0)


def floor_safe_row_bounds(
    h,
    row_lo_frac,
    row_hi_cap_frac,
    depth_max,
    *,
    cam_pitch_rad=None,
    cam_height_m=None,
    fov_deg=BRAIN_CAM_FOV_DEG,
    floor_margin_frac=DEPTH_OCC_FLOOR_MARGIN_FRAC,
):
    row_lo = max(0, int(row_lo_frac * h))
    row_hi_frac = row_hi_cap_frac
    if cam_pitch_rad is not None and cam_height_m is not None:
        floor_row = projected_floor_row_fraction(
            cam_pitch_rad, cam_height_m, depth_max, fov_deg,
        )
        row_hi_frac = min(row_hi_frac, floor_row - floor_margin_frac)
    row_hi = min(h, int(clamp(row_hi_frac, 0.0, 1.0) * h))
    if row_hi <= row_lo:
        row_hi = min(h, row_lo + max(1, int(0.04 * h)))
    return row_lo, row_hi


def robust_depth_column_distance(ray, depth_max):
    ray = np.asarray(ray, np.float32)
    valid = ray[np.isfinite(ray)]
    if valid.size == 0:
        return depth_max, None

    hits = valid[(valid >= MIN_OCC_DEPTH) & (valid < depth_max * 0.985)]
    if hits.size > 0:
        nearest = np.sort(hits)
        cluster_hi = float(nearest[0]) + OCC_HIT_CLUSTER_EPS_M
        cluster = nearest[nearest <= cluster_hi]
        min_cluster = max(OCC_HIT_MIN_ROWS, int(math.ceil(OCC_HIT_MIN_FRAC * float(valid.size))))
        if cluster.size >= min_cluster:
            hit_dist = float(clamp(float(np.nanmedian(cluster)), MIN_OCC_DEPTH, depth_max))
            return hit_dist, hit_dist

    clear_dist = float(clamp(float(np.nanpercentile(valid, 60)), 0.05, depth_max))
    return clear_dist, None


def bearing_depth_stats(depth_img, bearing_err_rad, depth_max,
                        cam_pitch_rad=None, cam_height_m=None,
                        fov_deg=BRAIN_CAM_FOV_DEG):
    d = normalize_depth_image(depth_img, depth_max)
    if d is None:
        return None
    half_fov = math.radians(max(float(fov_deg) * 0.5, 1e-3))
    if abs(bearing_err_rad) > half_fov + math.radians(2.0):
        return None

    h, w = d.shape
    x_norm = clamp(float(bearing_err_rad) / half_fov, -1.0, 1.0)
    c_mid = int(round((x_norm * 0.5 + 0.5) * max(w - 1, 1)))
    c0 = max(0, c_mid - 2)
    c1 = min(w, c_mid + 3)
    row_lo, row_hi = floor_safe_row_bounds(
        h, 0.02, 0.48, depth_max,
        cam_pitch_rad=cam_pitch_rad,
        cam_height_m=cam_height_m,
        fov_deg=fov_deg,
        floor_margin_frac=DEPTH_OCC_FLOOR_MARGIN_FRAC,
    )
    patch = d[row_lo:row_hi, c0:c1]
    if not patch.size:
        return None
    clear_fwd, hit_fwd = robust_depth_column_distance(patch.reshape(-1), depth_max)
    return {
        "x_norm": x_norm,
        "clear_range": clear_fwd,
        "hit_range": hit_fwd,
    }


def depth_guard_stats(depth_img, depth_max, cam_pitch_rad=None, cam_height_m=None,
                      fov_deg=BRAIN_CAM_FOV_DEG):
    d = normalize_depth_image(depth_img, depth_max)
    if d is None:
        return None
    h, w = d.shape
    row_lo, row_hi = floor_safe_row_bounds(
        h, 0.10, 0.70, depth_max,
        cam_pitch_rad=cam_pitch_rad,
        cam_height_m=cam_height_m,
        fov_deg=fov_deg,
        floor_margin_frac=DEPTH_GUARD_FLOOR_MARGIN_FRAC,
    )
    c0 = max(0, int(0.30 * w))
    c1 = min(w, max(c0 + 1, int(0.70 * w)))
    left   = d[row_lo:row_hi, :max(1, w // 3)]
    center = d[row_lo:row_hi, c0:c1]
    right  = d[row_lo:row_hi, max(0, 2 * w // 3):]
    return {
        "left_med": float(np.nanmedian(left)),
        "center_q35": float(np.nanpercentile(center, 35)),
        "center_close_frac": float(np.mean(center < FRONT_STOP_DIST)),
        "right_med": float(np.nanmedian(right)),
    }


def front_depth_guard_stats(depth_img, depth_max, cam_pitch_rad=None, cam_height_m=None,
                            fov_deg=BRAIN_CAM_FOV_DEG):
    d = normalize_depth_image(depth_img, depth_max)
    if d is None:
        return None
    h, w = d.shape
    row_lo, row_hi = floor_safe_row_bounds(
        h, 0.08, 0.62, depth_max,
        cam_pitch_rad=cam_pitch_rad,
        cam_height_m=cam_height_m,
        fov_deg=fov_deg,
        floor_margin_frac=DEPTH_GUARD_FLOOR_MARGIN_FRAC,
    )
    c0 = max(0, int(FRONT_GUARD_CENTER_LO * w))
    c1 = min(w, max(c0 + 1, int(FRONT_GUARD_CENTER_HI * w)))
    center = d[row_lo:row_hi, c0:c1]
    if not center.size:
        return None
    clear_vals = []
    hit_vals = []
    close_cols = 0
    n_cols = max(int(center.shape[1]), 1)
    for col in range(center.shape[1]):
        clear_dist, hit_dist = robust_depth_column_distance(center[:, col], depth_max)
        clear_vals.append(clear_dist)
        if hit_dist is not None:
            hit_vals.append(hit_dist)
            if hit_dist < FRONT_STOP_DIST:
                close_cols += 1
    min_hit_cols = max(2, int(math.ceil(0.20 * float(n_cols))))
    center_hit_q35 = (
        None if len(hit_vals) < min_hit_cols
        else float(np.nanpercentile(np.asarray(hit_vals, np.float32), 35))
    )
    return {
        "center_clear_q35": float(np.nanpercentile(np.asarray(clear_vals, np.float32), 35)),
        "center_hit_q35": center_hit_q35,
        "center_close_frac": float(close_cols) / float(n_cols),
    }


def depth_view_signature(depth_img, depth_max, cam_pitch_rad=None, cam_height_m=None,
                         fov_deg=BRAIN_CAM_FOV_DEG):
    stats = depth_guard_stats(
        depth_img, depth_max,
        cam_pitch_rad=cam_pitch_rad,
        cam_height_m=cam_height_m,
        fov_deg=fov_deg,
    )
    if stats is None:
        return None
    return np.array(
        [stats["left_med"], stats["center_q35"], stats["right_med"]],
        dtype=np.float32,
    )


def front_blocked_from_depth(depth_img, depth_max, cam_pitch_rad=None, cam_height_m=None,
                             fov_deg=BRAIN_CAM_FOV_DEG) -> bool:
    stats = front_depth_guard_stats(
        depth_img, depth_max,
        cam_pitch_rad=cam_pitch_rad,
        cam_height_m=cam_height_m,
        fov_deg=fov_deg,
    )
    if stats is None:
        return False
    hit_q35 = stats["center_hit_q35"]
    if hit_q35 is None:
        return False
    return (
        hit_q35 < FRONT_STOP_DIST
        or stats["center_close_frac"] > FRONT_BLOCKED_FRAC
    )


def make_escape_cmd_from_depth(depth_img, depth_max, dev,
                               reverse_speed=-0.15, turn_speed=0.55,
                               cam_pitch_rad=None, cam_height_m=None,
                               fov_deg=BRAIN_CAM_FOV_DEG):
    stats = depth_guard_stats(
        depth_img, depth_max,
        cam_pitch_rad=cam_pitch_rad,
        cam_height_m=cam_height_m,
        fov_deg=fov_deg,
    )
    if stats is None:
        return torch.tensor([[reverse_speed, 0.0, abs(turn_speed)]], device=dev, dtype=torch.float32)
    if stats["left_med"] > stats["right_med"] + 0.08:
        wz = -abs(turn_speed)
    elif stats["right_med"] > stats["left_med"] + 0.08:
        wz = abs(turn_speed)
    else:
        wz = abs(turn_speed)
    return torch.tensor([[reverse_speed, 0.0, wz]], device=dev, dtype=torch.float32)


def reinforce_front_obstacle(sm, robot_xy, robot_yaw, depth_img, depth_max,
                             cam_pitch_rad=None, cam_height_m=None,
                             fov_deg=BRAIN_CAM_FOV_DEG):
    stats = front_depth_guard_stats(
        depth_img, depth_max,
        cam_pitch_rad=cam_pitch_rad,
        cam_height_m=cam_height_m,
        fov_deg=fov_deg,
    )
    if stats is None or stats["center_hit_q35"] is None:
        hit_dist = FRONT_STOP_DIST
    else:
        hit_dist = float(clamp(stats["center_hit_q35"], 0.12, depth_max * 0.7))
    hit_xy = robot_xy + np.array([math.cos(robot_yaw), math.sin(robot_yaw)], np.float32) * hit_dist
    for _ in range(2):
        mark_disc(sm, hit_xy, max(OCC_INFLATION_RADIUS, 0.14), MAP_OCC)


def update_sensor_map_from_depth(sm, robot_xy, robot_yaw, depth_img, fov_deg, depth_max,
                                 cam_pitch_rad=None, cam_height_m=None):
    """Update occupancy purely from depth using asymmetric free/occ evidence."""
    mark_disc(sm, robot_xy, ROBOT_SELF_FREE_RADIUS, MAP_FREE)

    d = normalize_depth_image(depth_img, depth_max)
    if d is None:
        for a in np.linspace(-0.35, 0.35, 11):
            ry = robot_yaw + float(a)
            for t in np.linspace(0.08, 0.55, 8):
                mark_disc(sm, robot_xy + np.array([math.cos(ry), math.sin(ry)],
                               np.float32) * float(t), 0.05, MAP_FREE)
        return

    h, w = d.shape

    # Restrict OCC endpoint rows to the upper portion of the image.
    # These rows correspond to roughly horizontal or upward-facing rays and
    # will NOT hit the floor within depth_max.  Rows below this line look
    # downward and produce ground-return false positives.
    free_row_lo, free_row_hi = floor_safe_row_bounds(
        h, 0.08, 0.68, depth_max,
        cam_pitch_rad=cam_pitch_rad,
        cam_height_m=cam_height_m,
        fov_deg=fov_deg,
        floor_margin_frac=DEPTH_OCC_FLOOR_MARGIN_FRAC,
    )
    hit_row_lo, hit_row_hi = floor_safe_row_bounds(
        h, 0.02, 0.48, depth_max,
        cam_pitch_rad=cam_pitch_rad,
        cam_height_m=cam_height_m,
        fov_deg=fov_deg,
        floor_margin_frac=DEPTH_OCC_FLOOR_MARGIN_FRAC,
    )

    cols = np.unique(np.clip(np.linspace(int(0.08*w), int(0.92*w), 49).astype(int), 0, w-1))
    for c in cols:
        clear_ray = d[free_row_lo:free_row_hi, c]
        hit_ray = d[hit_row_lo:hit_row_hi, c]
        if not clear_ray.size and not hit_ray.size:
            continue
        clear_dist, _ = robust_depth_column_distance(clear_ray, depth_max)
        _, hit_dist = robust_depth_column_distance(hit_ray, depth_max) if hit_ray.size else (depth_max, None)
        x_norm = (float(c)/max(float(w-1),1.0))*2.0-1.0
        ray_ang = math.radians(0.5*fov_deg) * x_norm
        ray_yaw = robot_yaw + ray_ang
        ray_dir = np.array([math.cos(ray_yaw), math.sin(ray_yaw)], np.float32)

        # Mark free space along ray.
        free_until = max(0.0, clear_dist - FREE_RAY_CLEARANCE)
        for t in np.linspace(0.06, free_until, max(2, int(free_until/max(sm.res*0.7,0.04)))):
            mark_disc(sm, robot_xy + ray_dir * float(t), 0.04, MAP_FREE)

        if hit_dist is not None:
            mark_disc(sm, robot_xy + ray_dir * hit_dist, OCC_INFLATION_RADIUS, MAP_OCC)


def _frontier_reachable(sm, robot_xy, target_xy, n_samples=10, allow_unknown=False):
    """True if the straight line robot→target stays traversable."""
    seg = target_xy - robot_xy
    dist = float(np.linalg.norm(seg))
    if dist < 1e-6:
        return sample_traversable_with_clearance(sm, robot_xy, allow_unknown=allow_unknown)
    if not sample_traversable_with_clearance(sm, robot_xy, allow_unknown=allow_unknown):
        return False
    if not sample_traversable_with_clearance(sm, target_xy, allow_unknown=allow_unknown):
        return False
    n_samples = max(
        int(n_samples),
        int(math.ceil(dist / max(LINE_CHECK_STEP_M, sm.res * 0.5))),
    )
    for t in np.linspace(0.05, 0.95, n_samples):
        if not sample_traversable_with_clearance(
            sm,
            robot_xy + t * (target_xy - robot_xy),
            allow_unknown=allow_unknown,
        ):
            return False
    return True


def waypoint_seek_anchor(wp: MazeWaypoint) -> np.ndarray:
    return (wp.pos - SEEK_ANCHOR_OFFSET * wp.approach_dir).astype(np.float32)


def nearest_cell_with_value(sm, goal_xy, value, max_radius_m=0.45):
    goal_rc = world_to_grid(sm, goal_xy)
    if goal_rc is None:
        return None
    rr_max = max(1, int(math.ceil(max_radius_m / sm.res)))
    best_rc = None
    best_dist = float("inf")
    r0, c0 = goal_rc
    for r in range(max(0, r0 - rr_max), min(sm.h, r0 + rr_max + 1)):
        for c in range(max(0, c0 - rr_max), min(sm.w, c0 + rr_max + 1)):
            if sm.grid[r, c] != value:
                continue
            p = grid_to_world(sm, (r, c))
            d = float(np.linalg.norm(p - goal_xy))
            if d <= max_radius_m and d < best_dist:
                best_dist = d
                best_rc = (r, c)
    return best_rc


def local_visit_score(sm, rc, radius_cells=2):
    r0, c0 = int(rc[0]), int(rc[1])
    acc = 0.0
    count = 0
    rr2 = float(radius_cells * radius_cells)
    for r in range(max(0, r0 - radius_cells), min(sm.h, r0 + radius_cells + 1)):
        for c in range(max(0, c0 - radius_cells), min(sm.w, c0 + radius_cells + 1)):
            if float((r - r0) ** 2 + (c - c0) ** 2) > rr2 + 0.25:
                continue
            acc += math.log1p(max(int(sm.free_visits[r, c]), 0))
            count += 1
    return 0.0 if count == 0 else acc / float(count)


def bfs_next_waypoint(
    sm: "SensorMap", robot_xy: np.ndarray, goal_xy: np.ndarray,
    lookahead_m: float = 0.7, allow_unknown: bool = True,
    snap_goal_to_free: bool = True,
) -> Optional[np.ndarray]:
    start = world_to_grid(sm, robot_xy)
    end   = world_to_grid(sm, goal_xy)
    if start is None or end is None:
        return None
    end_cell = int(sm.grid[end[0], end[1]])
    if end_cell == MAP_OCC:
        if not snap_goal_to_free:
            return None
        snapped = nearest_cell_with_value(sm, goal_xy, MAP_FREE, max_radius_m=0.45)
        if snapped is None and allow_unknown:
            snapped = nearest_cell_with_value(sm, goal_xy, MAP_UNKNOWN, max_radius_m=0.45)
        if snapped is None:
            return None
        end = snapped
        end_cell = int(sm.grid[end[0], end[1]])
    if not allow_unknown and end_cell != MAP_FREE:
        if not snap_goal_to_free:
            return None
        snapped = nearest_cell_with_value(sm, goal_xy, MAP_FREE, max_radius_m=0.45)
        if snapped is None:
            return None
        end = snapped
    if start == end:
        return grid_to_world(sm, end)

    prev: dict = {start: None}
    best_cost: Dict[Tuple[int, int], float] = {start: 0.0}
    heap: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]

    while heap:
        cur_cost, cur = heapq.heappop(heap)
        if cur_cost > best_cost.get(cur, float("inf")) + 1e-8:
            continue
        if cur == end:
            break
        r, c = cur
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < sm.h and 0 <= nc < sm.w:
                if dr != 0 and dc != 0:
                    if sm.grid[r, nc] == MAP_OCC or sm.grid[nr, c] == MAP_OCC:
                        continue
                nxt = (nr, nc)
                cell = sm.grid[nr, nc]
                if cell == MAP_OCC:
                    continue
                if not allow_unknown and cell != MAP_FREE:
                    continue
                step_cost = math.sqrt(2.0) if dr != 0 and dc != 0 else 1.0
                if cell == MAP_UNKNOWN:
                    step_cost *= UNKNOWN_TRAVERSE_COST
                step_cost += VISIT_TRAVERSE_COST * math.log1p(max(int(sm.free_visits[nr, nc]), 0))
                new_cost = cur_cost + step_cost
                if new_cost + 1e-8 < best_cost.get(nxt, float("inf")):
                    best_cost[nxt] = new_cost
                    prev[nxt] = cur
                    heapq.heappush(heap, (new_cost, nxt))
    else:
        return None

    path: list = []
    cur: Optional[tuple] = end
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()

    idx = min(max(1, int(lookahead_m / sm.res)), len(path) - 1)
    return grid_to_world(sm, path[idx])


def select_frontier(sm, robot_xy, blacklist=None, bl_radius=0.40):
    bl = blacklist or []
    cands = []
    for r in range(1, sm.h-1):
        for c in range(1, sm.w-1):
            if sm.grid[r,c] != MAP_FREE: continue
            unk = sum(1 for rr in range(r-1,r+2) for cc in range(c-1,c+2)
                      if not (rr==r and cc==c) and sm.grid[rr,cc]==MAP_UNKNOWN)
            if not unk: continue
            occ = sum(1 for rr in range(r-1,r+2) for cc in range(c-1,c+2)
                      if not (rr==r and cc==c) and sm.grid[rr,cc]==MAP_OCC)
            wp = grid_to_world(sm, (r,c))
            dist = float(np.linalg.norm(wp - robot_xy))
            if dist < 0.20: continue
            if any(float(np.linalg.norm(wp-bp)) < bl_radius for bp in bl): continue
            reach = 2.0 if _frontier_reachable(sm, robot_xy, wp) else -1.5
            visit_pen = min(FRONTIER_VISIT_PENALTY * local_visit_score(sm, (r, c)), 1.8)
            cands.append((0.45*float(unk) - 0.30*dist - 0.35*float(occ) + reach - visit_pen, wp))
    if not cands:
        if bl:
            return select_frontier(sm, robot_xy)
        best_d, best_wp = float("inf"), robot_xy.copy()
        for r in range(sm.h):
            for c in range(sm.w):
                if sm.grid[r, c] == MAP_UNKNOWN:
                    wp = grid_to_world(sm, (r, c))
                    d = float(np.linalg.norm(wp - robot_xy))
                    if 0.20 < d < best_d:
                        best_d, best_wp = d, wp
        return best_wp, 0.0
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1], cands[0][0]


def find_far_unknown(sm, robot_xy, min_dist=0.35):
    best_d = 0.0
    best_wp = robot_xy.copy()
    for r in range(sm.h):
        for c in range(sm.w):
            if sm.grid[r, c] != MAP_UNKNOWN:
                continue
            wp = grid_to_world(sm, (r, c))
            d = float(np.linalg.norm(wp - robot_xy))
            if d > best_d:
                best_d = d
                best_wp = wp
    return best_wp if best_d >= min_dist else robot_xy.copy()


def select_goal_proxy(sm, robot_xy, goal_xy, blacklist=None, bl_radius=0.35):
    """Reachable free-space proxy that moves the robot toward an occluded goal."""
    bl = blacklist or []
    best_wp = None
    best_score = float("-inf")
    goal_dist0 = float(np.linalg.norm(goal_xy - robot_xy))

    for r in range(1, sm.h - 1):
        for c in range(1, sm.w - 1):
            if sm.grid[r, c] != MAP_FREE:
                continue
            wp = grid_to_world(sm, (r, c))
            if any(float(np.linalg.norm(wp - bp)) < bl_radius for bp in bl):
                continue
            dist_robot = float(np.linalg.norm(wp - robot_xy))
            if dist_robot < 0.20:
                continue
            if not _frontier_reachable(sm, robot_xy, wp):
                continue

            dist_goal = float(np.linalg.norm(goal_xy - wp))
            progress = goal_dist0 - dist_goal
            unk = sum(
                1 for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)
                if not (rr == r and cc == c) and sm.grid[rr, cc] == MAP_UNKNOWN
            )
            occ = sum(
                1 for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)
                if not (rr == r and cc == c) and sm.grid[rr, cc] == MAP_OCC
            )
            score = (
                1.35 * progress
                + 0.22 * float(unk)
                - 0.20 * dist_robot
                - 0.30 * float(occ)
            )
            if score > best_score:
                best_score = score
                best_wp = wp

    return best_wp


# --------------------------------------------------------------------------- #
# Kinematic rollout helpers
# --------------------------------------------------------------------------- #

def rollout_cmds_batched(start_xy, start_yaw, cmds, horizon, dt=0.10, cmds2=None, split=None):
    N   = cmds.shape[0]
    px  = cmds.new_full((N,), float(start_xy[0]))
    py  = cmds.new_full((N,), float(start_xy[1]))
    yaw = cmds.new_full((N,), float(start_yaw))
    vx1, vy1, wyaw1 = cmds[:,0], cmds[:,1], cmds[:,2]
    if cmds2 is not None and split is not None:
        vx2, vy2, wyaw2 = cmds2[:,0], cmds2[:,1], cmds2[:,2]
    else:
        vx2, vy2, wyaw2 = vx1, vy1, wyaw1
        split = horizon
    for t in range(horizon):
        vx = vx1 if t < split else vx2
        vy = vy1 if t < split else vy2
        wyaw = wyaw1 if t < split else wyaw2
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        px  = px  + dt*(cy*vx - sy*vy)
        py  = py  + dt*(sy*vx + cy*vy)
        yaw = yaw + dt*wyaw
    return torch.stack([px, py], dim=1), yaw


def rollout_cmds_batched_paths(start_xy, start_yaw, cmds, horizon, dt=0.10, cmds2=None, split=None):
    N   = cmds.shape[0]
    px  = cmds.new_full((N,), float(start_xy[0]))
    py  = cmds.new_full((N,), float(start_xy[1]))
    yaw = cmds.new_full((N,), float(start_yaw))
    vx1, vy1, wyaw1 = cmds[:,0], cmds[:,1], cmds[:,2]
    if cmds2 is not None and split is not None:
        vx2, vy2, wyaw2 = cmds2[:,0], cmds2[:,1], cmds2[:,2]
    else:
        vx2, vy2, wyaw2 = vx1, vy1, wyaw1
        split = horizon
    xs, ys = [], []
    for t in range(horizon):
        xs.append(px); ys.append(py)
        vx = vx1 if t < split else vx2
        vy = vy1 if t < split else vy2
        wyaw = wyaw1 if t < split else wyaw2
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        px  = px  + dt*(cy*vx - sy*vy)
        py  = py  + dt*(sy*vx + cy*vy)
        yaw = yaw + dt*wyaw
    return torch.stack([torch.stack(xs,1), torch.stack(ys,1)], 2), torch.stack([px,py],1)


def path_collision_penalty_batched(sm, paths_np):
    N, hz, _ = paths_np.shape
    pts = paths_np.reshape(-1, 2)
    gx = np.clip(((pts[:,0]-float(WORLD_MIN[0]))/sm.res).astype(np.int32), 0, sm.w-1)
    gy = np.clip(((pts[:,1]-float(WORLD_MIN[1]))/sm.res).astype(np.int32), 0, sm.h-1)
    occ0 = (sm.grid[gy, gx] == MAP_OCC).astype(np.float32)
    pen = occ0 * 2.8
    orth_hits = np.zeros_like(occ0)
    for dr,dc,w in (
        (-1,0,0.28), (1,0,0.28), (0,-1,0.28), (0,1,0.28),
        (-1,-1,0.20), (-1,1,0.20), (1,-1,0.20), (1,1,0.20),
        (-2,0,0.10), (2,0,0.10), (0,-2,0.10), (0,2,0.10),
    ):
        gyn = np.clip(gy+dr,0,sm.h-1); gxn = np.clip(gx+dc,0,sm.w-1)
        hit = (sm.grid[gyn,gxn] == MAP_OCC).astype(np.float32)
        pen += hit * w
        if abs(dr) + abs(dc) == 1:
            orth_hits += hit
    pen += (orth_hits >= 2).astype(np.float32) * 0.75
    return pen.reshape(N,hz).sum(axis=1)


def path_collision_penalty_batched_torch(sm, paths_t, occ_grid=None):
    N, hz, _ = paths_t.shape
    if occ_grid is None:
        occ_grid = torch.as_tensor(sm.grid == MAP_OCC, device=paths_t.device)
    pts = paths_t.reshape(-1, 2)
    gx = (((pts[:, 0] - float(WORLD_MIN[0])) / sm.res).long()).clamp_(0, sm.w - 1)
    gy = (((pts[:, 1] - float(WORLD_MIN[1])) / sm.res).long()).clamp_(0, sm.h - 1)
    pen = occ_grid[gy, gx].to(paths_t.dtype) * 2.8
    orth_hits = torch.zeros_like(pen)
    for dr, dc, w in (
        (-1, 0, 0.28), (1, 0, 0.28), (0, -1, 0.28), (0, 1, 0.28),
        (-1, -1, 0.20), (-1, 1, 0.20), (1, -1, 0.20), (1, 1, 0.20),
        (-2, 0, 0.10), (2, 0, 0.10), (0, -2, 0.10), (0, 2, 0.10),
    ):
        gyn = (gy + dr).clamp_(0, sm.h - 1)
        gxn = (gx + dc).clamp_(0, sm.w - 1)
        hit = occ_grid[gyn, gxn].to(paths_t.dtype)
        pen = pen + hit * w
        if abs(dr) + abs(dc) == 1:
            orth_hits = orth_hits + hit
    pen = pen + (orth_hits >= 2).to(paths_t.dtype) * 0.75
    return pen.view(N, hz).sum(dim=1)


def local_unknown_gain_batched(sm, end_xy, radius=0.55):
    N = end_xy.shape[0]
    gx0 = ((end_xy[:,0]-float(WORLD_MIN[0]))/sm.res).astype(np.int32)
    gy0 = ((end_xy[:,1]-float(WORLD_MIN[1]))/sm.res).astype(np.int32)
    rr = max(1, int(radius/sm.res))
    counts = np.zeros(N, np.float32)
    for dr in range(-rr, rr+1):
        for dc in range(-rr, rr+1):
            gyn = gy0+dr; gxn = gx0+dc
            valid = (gyn>=0)&(gyn<sm.h)&(gxn>=0)&(gxn<sm.w)
            px = float(WORLD_MIN[0])+(gxn+0.5)*sm.res
            py = float(WORLD_MIN[1])+(gyn+0.5)*sm.res
            in_r = np.sqrt((px-end_xy[:,0])**2+(py-end_xy[:,1])**2)<=radius
            gyn_c = np.clip(gyn,0,sm.h-1); gxn_c = np.clip(gxn,0,sm.w-1)
            counts += (valid & in_r & (sm.grid[gyn_c,gxn_c]==MAP_UNKNOWN)).astype(np.float32)
    return counts * 0.08


def local_unknown_gain_batched_torch(sm, end_xy_t, unknown_grid=None, radius=0.55):
    N = int(end_xy_t.shape[0])
    if unknown_grid is None:
        unknown_grid = torch.as_tensor(sm.grid == MAP_UNKNOWN, device=end_xy_t.device)
    gx0 = (((end_xy_t[:, 0] - float(WORLD_MIN[0])) / sm.res).long())
    gy0 = (((end_xy_t[:, 1] - float(WORLD_MIN[1])) / sm.res).long())
    rr = max(1, int(radius / sm.res))
    counts = torch.zeros(N, device=end_xy_t.device, dtype=end_xy_t.dtype)
    for dr in range(-rr, rr + 1):
        for dc in range(-rr, rr + 1):
            gyn = gy0 + dr
            gxn = gx0 + dc
            valid = (gyn >= 0) & (gyn < sm.h) & (gxn >= 0) & (gxn < sm.w)
            gyn_c = gyn.clamp(0, sm.h - 1)
            gxn_c = gxn.clamp(0, sm.w - 1)
            px = float(WORLD_MIN[0]) + (gxn_c.to(end_xy_t.dtype) + 0.5) * sm.res
            py = float(WORLD_MIN[1]) + (gyn_c.to(end_xy_t.dtype) + 0.5) * sm.res
            in_r = torch.sqrt((px - end_xy_t[:, 0]) ** 2 + (py - end_xy_t[:, 1]) ** 2) <= radius
            counts = counts + (valid & in_r & unknown_grid[gyn_c, gxn_c]).to(end_xy_t.dtype)
    return counts * 0.08


def path_visit_penalty_batched_torch(sm, paths_t, visit_grid=None):
    N, hz, _ = paths_t.shape
    if visit_grid is None:
        visit_grid = torch.as_tensor(sm.free_visits, device=paths_t.device)
    pts = paths_t.reshape(-1, 2)
    gx = (((pts[:, 0] - float(WORLD_MIN[0])) / sm.res).long()).clamp_(0, sm.w - 1)
    gy = (((pts[:, 1] - float(WORLD_MIN[1])) / sm.res).long()).clamp_(0, sm.h - 1)
    visits = torch.log1p(visit_grid[gy, gx].to(paths_t.dtype))
    return visits.view(N, hz).mean(dim=1)


def normalize_latents_t(latents: torch.Tensor) -> torch.Tensor:
    return latents / latents.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def latent_bank_novelty_torch(
    latents: torch.Tensor,
    memory_bank_norm: Optional[torch.Tensor],
) -> torch.Tensor:
    if memory_bank_norm is None or memory_bank_norm.numel() == 0:
        return torch.ones(latents.shape[0], device=latents.device, dtype=latents.dtype)
    sims = normalize_latents_t(latents) @ memory_bank_norm.t()
    max_sim = sims.max(dim=1).values
    return (1.0 - max_sim).clamp_min(0.0)


@torch.no_grad()
def place_memory_key(robot_xy: np.ndarray, robot_yaw: float):
    cell_x = int(math.floor((float(robot_xy[0]) - float(WORLD_MIN[0])) / PLACE_MEMORY_CELL_M))
    cell_y = int(math.floor((float(robot_xy[1]) - float(WORLD_MIN[1])) / PLACE_MEMORY_CELL_M))
    yaw_norm = (wrap_to_pi(robot_yaw) + math.pi) / (2.0 * math.pi)
    yaw_bin = int(math.floor(yaw_norm * PLACE_MEMORY_YAW_BINS)) % PLACE_MEMORY_YAW_BINS
    return cell_x, cell_y, yaw_bin


@torch.no_grad()
def update_place_memory(
    memory_slots: Dict[Tuple[int, int, int], torch.Tensor],
    z_online: torch.Tensor,
    robot_xy: np.ndarray,
    robot_yaw: float,
    max_size: int = LATENT_MEMORY_MAX,
):
    z = z_online.detach()
    if z.dim() == 1:
        z = z.unsqueeze(0)
    z = z[:1].clone()
    key = place_memory_key(robot_xy, robot_yaw)

    added = key not in memory_slots
    if key in memory_slots:
        prev = memory_slots.pop(key)
        z = PLACE_LATENT_EMA_DECAY * prev + (1.0 - PLACE_LATENT_EMA_DECAY) * z
    memory_slots[key] = z

    while len(memory_slots) > max_size:
        oldest_key = next(iter(memory_slots))
        memory_slots.pop(oldest_key)

    if not memory_slots:
        return memory_slots, None, None, added

    memory_bank = torch.cat(list(memory_slots.values()), dim=0)
    memory_bank_norm = normalize_latents_t(memory_bank)
    return memory_slots, memory_bank, memory_bank_norm, added


# --------------------------------------------------------------------------- #
# Breadcrumb harvesting
# --------------------------------------------------------------------------- #

@torch.no_grad()
def harvest_breadcrumb(
    robot, cam_brain, q0, jepa, ppo, dofs, dev, scene, wp: MazeWaypoint,
    n_avg=5, warmup=10, speed=0.25, start_offset=0.45,
) -> Tuple[np.ndarray, torch.Tensor]:
    primary = wp.approach_dir / max(float(np.linalg.norm(wp.approach_dir)), 1e-8)
    base_yaw = math.atan2(float(primary[1]), float(primary[0]))
    dir_list = []
    for yaw_deg in BREADCRUMB_YAW_OFFSETS_DEG:
        yaw = base_yaw + math.radians(float(yaw_deg))
        dir_list.append(np.array([math.cos(yaw), math.sin(yaw)], np.float32))

    latents = []
    for d in dir_list:
        start_xy  = wp.pos - start_offset * d
        start_yaw = math.atan2(float(d[1]), float(d[0]))

        robot.set_pos(np.array([start_xy[0], start_xy[1], 0.12], np.float32))
        robot.set_quat(yaw_to_quat(start_yaw))
        robot.set_dofs_position(q0.detach().cpu().numpy(), dofs)
        for _ in range(8): scene.step()

        pa  = torch.zeros((1,12), device=dev)
        cmd = torch.tensor([[speed, 0.0, 0.0]], device=dev)
        for _ in range(warmup):
            obs = get_sys1_obs(robot, q0, pa, cmd, dofs, dev)
            pa  = ppo.act_deterministic(obs)
            robot.control_dofs_position(to_genesis_target(q0 + 0.3*pa[0]), dofs)
            for _ in range(4): scene.step()

        zs = []
        for _ in range(n_avg):
            obs = get_sys1_obs(robot, q0, pa, cmd, dofs, dev)
            pa  = ppo.act_deterministic(obs)
            robot.control_dofs_position(to_genesis_target(q0 + 0.3*pa[0]), dofs)
            for _ in range(4): scene.step()
            vis, prop, _, _ = get_jepa_state(robot, cam_brain, q0, pa, dofs, dev)
            zs.append(jepa.encode_target(vis, prop).detach())
        latents.append(torch.stack(zs).mean(0))

    dirs_np   = np.stack(dir_list, axis=0)
    latents_t = torch.stack(latents, dim=0).squeeze(1)
    return dirs_np, latents_t


@torch.no_grad()
def harvest_explore_reference(
    robot, cam_brain, q0, jepa, ppo, dofs, dev, scene,
    n_avg=8, warmup=15, speed=0.30,
) -> torch.Tensor:
    robot.set_pos(np.array(ROBOT_SPAWN, np.float32))
    robot.set_quat(yaw_to_quat(math.pi / 2))
    robot.set_dofs_position(q0.detach().cpu().numpy(), dofs)
    for _ in range(8): scene.step()

    pa  = torch.zeros((1, 12), device=dev)
    cmd = torch.tensor([[speed, 0.0, 0.0]], device=dev)
    for _ in range(warmup):
        obs = get_sys1_obs(robot, q0, pa, cmd, dofs, dev)
        pa  = ppo.act_deterministic(obs)
        robot.control_dofs_position(to_genesis_target(q0 + 0.3 * pa[0]), dofs)
        for _ in range(4): scene.step()

    zs = []
    for _ in range(n_avg):
        obs = get_sys1_obs(robot, q0, pa, cmd, dofs, dev)
        pa  = ppo.act_deterministic(obs)
        robot.control_dofs_position(to_genesis_target(q0 + 0.3 * pa[0]), dofs)
        for _ in range(4): scene.step()
        vis, prop, _, _ = get_jepa_state(robot, cam_brain, q0, pa, dofs, dev)
        zs.append(jepa.encode_target(vis, prop).detach())

    return torch.stack(zs).mean(0)


# --------------------------------------------------------------------------- #
# Goal-directed planner
# --------------------------------------------------------------------------- #

@torch.no_grad()
def plan_seek_cmd(
    jepa, head, z_current, z_goal, sm, robot_xy, robot_yaw, goal_xy,
    goal_body_xy, dist_to_goal, heading_error, n_candidates, horizon, dev,
    prev_cmd=None,
    energy_scale=1.0,
    depth_sig=None,
):
    far = dist_to_goal > 0.9
    speed_scale = float(np.clip(dist_to_goal / 0.9, 0.2, 1.0))
    vx_clamp    = float(np.clip(dist_to_goal * 0.55, 0.15, 0.40))

    if abs(heading_error) > math.pi * 0.55:
        mean = torch.tensor([0.0, 0.0, math.copysign(0.75, heading_error)],
                            device=dev, dtype=torch.float32)
        std  = torch.tensor([0.04, 0.04, 0.15], device=dev, dtype=torch.float32)
    else:
        ts = (0.30 if far else 0.18) * speed_scale
        if abs(heading_error) > 0.65: ts *= 0.35
        mean = torch.tensor([
            clamp(float(goal_body_xy[0])*ts, -0.35, 0.35),
            clamp(float(goal_body_xy[1])*ts, -0.20, 0.20),
            clamp(0.65*heading_error, -0.72, 0.72),
        ], device=dev, dtype=torch.float32)
        std = torch.tensor([
            (0.12 if far else 0.09)*speed_scale,
            (0.10 if far else 0.08)*speed_scale,
            0.22 if far else 0.18,
        ], device=dev, dtype=torch.float32)

    # Phase-2 mean: bias toward goal-facing forward drive after initial manoeuvre.
    mean2 = torch.tensor([
        clamp(0.28 * speed_scale, 0.08, 0.36),
        0.0,
        clamp(0.40 * heading_error, -0.55, 0.55),
    ], device=dev, dtype=torch.float32)
    std2 = torch.tensor([0.10, 0.04, 0.20], device=dev, dtype=torch.float32)

    split = max(3, horizon // 2)

    best_cmd  = mean.view(1,3)
    best_cost = None
    best_path = None
    energy_weight = float(np.clip((dist_to_goal-0.35)/0.65, 0.0, 1.0)) * float(energy_scale)
    goal_t = torch.tensor(goal_xy, device=dev, dtype=torch.float32)
    occ_grid = torch.as_tensor(sm.grid == MAP_OCC, device=dev)

    # Depth-based obstacle penalty: replaces BFS wall-avoidance.
    DEPTH_PEN_ONSET = 0.50
    DEPTH_PEN_FULL  = 0.25
    DEPTH_PEN_COEFF = 9.0
    depth_fwd_scale = 0.0
    depth_turn_bias = 0.0
    if depth_sig is not None:
        d_center = float(depth_sig[1])
        depth_fwd_scale = float(np.clip(
            (DEPTH_PEN_ONSET - d_center) / (DEPTH_PEN_ONSET - DEPTH_PEN_FULL),
            0.0, 1.0,
        ))
        d_left  = float(depth_sig[0])
        d_right = float(depth_sig[2])
        if depth_fwd_scale > 0.05 and abs(d_left - d_right) > 0.05:
            depth_turn_bias = float(np.clip(
                (d_left - d_right) / max(d_center, 0.1), -1.0, 1.0,
            )) * depth_fwd_scale * 1.5

    for _ in range(5):
        cmds1 = mean + std * torch.randn((n_candidates,3), device=dev)
        cmds1[:,0].clamp_(0.0, vx_clamp)
        cmds1[:,1].clamp_(-0.25, 0.25)
        cmds1[:,2].clamp_(-0.80, 0.80)
        cmds2 = mean2 + std2 * torch.randn((n_candidates,3), device=dev)
        cmds2[:,0].clamp_(0.0, 0.40)
        cmds2[:,1].clamp_(-0.15, 0.15)
        cmds2[:,2].clamp_(-0.75, 0.75)

        z_roll = z_current.expand(n_candidates, -1)
        h_t    = torch.zeros((n_candidates, jepa.latent_dim), device=dev, dtype=z_roll.dtype)
        for t in range(horizon):
            cmd_t = cmds1 if t < split else cmds2
            z_roll, h_t = jepa.predictor(z_roll, cmd_t, h_t)

        eng = head(z_roll, z_goal.expand_as(z_roll))

        paths_t, end_xy_t = rollout_cmds_batched_paths(
            robot_xy, robot_yaw, cmds1, horizon, cmds2=cmds2, split=split)
        _, end_yaw_t = rollout_cmds_batched(
            robot_xy, robot_yaw, cmds1, horizon, cmds2=cmds2, split=split)
        coll_t     = path_collision_penalty_batched_torch(sm, paths_t, occ_grid)
        end_dist_t = (end_xy_t - goal_t).norm(dim=1)
        end_ang_t  = torch.atan2(goal_t[1]-end_xy_t[:,1], goal_t[0]-end_xy_t[:,0])
        end_herr_t = ((end_ang_t - end_yaw_t + math.pi) % (2*math.pi) - math.pi).abs()
        geo_cost   = 0.75*end_dist_t + 0.30*end_herr_t

        cost = (
            energy_weight * eng
            + geo_cost
            + coll_t
            + 0.45 * cmds1[:, 1].abs()
            + 0.06 * cmds1[:, 2].abs()
        )
        if prev_cmd is not None:
            cost = cost + 0.10*(cmds1 - prev_cmd).pow(2).sum(dim=-1)

        if depth_fwd_scale > 0.01:
            cost = cost + cmds1[:, 0].clamp(min=0) * depth_fwd_scale * DEPTH_PEN_COEFF
        if abs(depth_turn_bias) > 0.05:
            cost = cost - depth_turn_bias * cmds1[:, 2]

        k = max(n_candidates//10, 8)
        elite = torch.topk(cost, k=k, largest=False).indices
        mean  = cmds1[elite].mean(0)
        std   = cmds1[elite].std(0) + 1e-4
        mean2 = cmds2[elite].mean(0)
        std2  = cmds2[elite].std(0) + 1e-4

        ib = int(torch.argmin(cost).item())
        if best_cost is None or float(cost[ib]) < best_cost:
            best_cost = float(cost[ib])
            best_cmd  = cmds1[ib].view(1,3).detach().clone()
            best_path = paths_t[ib].detach().cpu().numpy()

    return best_cmd, best_path if best_path is not None else np.zeros((horizon,2), np.float32)


def _kin_path(start_xy, start_yaw, cmd, horizon, dt=0.10):
    pos, yaw, path = np.array(start_xy, np.float32).copy(), float(start_yaw), []
    for _ in range(horizon):
        path.append(pos.copy())
        pos += dt * body_to_world_xy(yaw, cmd[:2])
        yaw  = wrap_to_pi(yaw + dt * float(cmd[2]))
    return np.stack(path), pos, yaw


# --------------------------------------------------------------------------- #
# Exploration planner
# --------------------------------------------------------------------------- #

@torch.no_grad()
def plan_explore_cmd(jepa, zc, latent_memory_norm, sm, robot_xy, robot_yaw, frontier_xy,
                     prev_cmd, cands, hz, dev):
    goal_vec  = frontier_xy - robot_xy
    goal_dist = float(np.linalg.norm(goal_vec))
    if goal_dist < 1e-6:
        nudge = torch.tensor([[0.16, 0.0, 0.45]], device=dev, dtype=torch.float32)
        return nudge, np.zeros((hz, 2), np.float32)

    hdg_err = wrap_to_pi(math.atan2(float(goal_vec[1]), float(goal_vec[0])) - robot_yaw)
    far = goal_dist > 0.8
    ts = (0.32 if far else 0.22) * float(np.clip(goal_dist / 0.9, 0.35, 1.0))
    if abs(hdg_err) > 0.90:
        ts *= 0.55

    if abs(hdg_err) > math.pi * 0.55:
        mean = torch.tensor([0.0, 0.0, math.copysign(0.70, hdg_err)],
                            device=dev, dtype=torch.float32)
        std = torch.tensor([0.04, 0.025, 0.12], device=dev, dtype=torch.float32)
    else:
        mean = torch.tensor([
            clamp(ts, 0.0, 0.36),
            0.0,
            clamp(0.60 * hdg_err, -0.72, 0.72),
        ], device=dev, dtype=torch.float32)
        std = torch.tensor([
            0.10 if far else 0.07,
            0.035 if far else 0.025,
            0.22 if far else 0.16,
        ], device=dev, dtype=torch.float32)

    # Phase-2 mean: drive forward toward the goal after initial manoeuvre.
    mean2 = torch.tensor([
        clamp(ts * 1.2, 0.10, 0.36),
        0.0,
        clamp(0.40 * hdg_err, -0.55, 0.55),
    ], device=dev, dtype=torch.float32)
    std2 = torch.tensor([0.10, 0.03, 0.18], device=dev, dtype=torch.float32)

    split = max(3, hz // 2)

    occ_grid = torch.as_tensor(sm.grid == MAP_OCC, device=dev)
    unknown_grid = torch.as_tensor(sm.grid == MAP_UNKNOWN, device=dev)
    visit_grid = torch.as_tensor(sm.free_visits, device=dev)
    goal_t = torch.tensor(frontier_xy, device=dev, dtype=torch.float32)
    robot_xy_t = torch.tensor(robot_xy, device=dev, dtype=torch.float32)
    z_ref_norm = float(zc.norm(dim=-1).mean().item()) + 1e-6

    best_cmd = mean.view(1, 3)
    best_cost = None
    best_path = None

    for _ in range(4):
        cmds1 = mean + std * torch.randn((cands, 3), device=dev)
        cmds1[:, 0].clamp_(0.0, 0.38)
        cmds1[:, 1].clamp_(-0.08, 0.08)
        cmds1[:, 2].clamp_(-0.75, 0.75)
        cmds2 = mean2 + std2 * torch.randn((cands, 3), device=dev)
        cmds2[:, 0].clamp_(0.0, 0.38)
        cmds2[:, 1].clamp_(-0.08, 0.08)
        cmds2[:, 2].clamp_(-0.75, 0.75)

        z_roll = zc.expand(cands, -1)
        h_t = torch.zeros((cands, jepa.latent_dim), device=dev, dtype=z_roll.dtype)
        for t in range(hz):
            cmd_t = cmds1 if t < split else cmds2
            z_roll, h_t = jepa.predictor(z_roll, cmd_t, h_t)

        latent_mag = z_roll.norm(dim=-1) / z_ref_norm
        novelty_t = latent_bank_novelty_torch(z_roll, latent_memory_norm)

        paths_t, end_xy_t = rollout_cmds_batched_paths(
            robot_xy, robot_yaw, cmds1, hz, cmds2=cmds2, split=split)
        _, end_yaw_t = rollout_cmds_batched(
            robot_xy, robot_yaw, cmds1, hz, cmds2=cmds2, split=split)

        coll_t = path_collision_penalty_batched_torch(sm, paths_t, occ_grid)
        visit_t = path_visit_penalty_batched_torch(sm, paths_t, visit_grid)
        frontier_t = local_unknown_gain_batched_torch(sm, end_xy_t, unknown_grid, radius=0.60)
        end_dist_t = (end_xy_t - goal_t).norm(dim=1)
        end_disp_t = (end_xy_t - robot_xy_t).norm(dim=1)
        progress_t = goal_dist - end_dist_t
        end_ang_t = torch.atan2(goal_t[1] - end_xy_t[:, 1], goal_t[0] - end_xy_t[:, 0])
        end_herr_t = ((end_ang_t - end_yaw_t + math.pi) % (2 * math.pi) - math.pi).abs()

        latent_pen = (latent_mag - 1.5).clamp(min=0.0) * 0.15
        smooth_pen = (
            torch.zeros(cands, device=dev) if prev_cmd is None
            else 0.10 * (cmds1 - prev_cmd).pow(2).sum(dim=-1)
        )
        stasis_pen = (0.16 - end_disp_t).clamp(min=0.0) * 2.5
        novelty_bonus = LATENT_NOVELTY_WEIGHT * novelty_t * progress_t.clamp(min=0.0, max=0.30)

        cost = (
            coll_t
            + PATH_VISIT_PENALTY * visit_t
            + latent_pen
            + smooth_pen
            + 1.10 * cmds1[:, 1].abs()
            + 0.10 * cmds1[:, 2].abs()
            + 0.20 * end_herr_t
            + stasis_pen
            + (cmds1[:, 0] < 0.08).float() * 0.40
            - 3.40 * progress_t
            - 0.95 * frontier_t
            - novelty_bonus
        )

        elite_k = max(8, cands // 10)
        elite = torch.topk(cost, k=elite_k, largest=False).indices
        mean = cmds1[elite].mean(dim=0)
        std = cmds1[elite].std(dim=0) + 1e-4
        mean2 = cmds2[elite].mean(dim=0)
        std2 = cmds2[elite].std(dim=0) + 1e-4

        ib = int(torch.argmin(cost).item())
        if best_cost is None or float(cost[ib]) < best_cost:
            best_cost = float(cost[ib])
            best_cmd = cmds1[ib].detach().clone().view(1, 3)
            best_path = paths_t[ib].detach().cpu().numpy()

    return best_cmd, best_path if best_path is not None else np.zeros((hz, 2), np.float32)


# --------------------------------------------------------------------------- #
# Waypoint detection
# --------------------------------------------------------------------------- #

def _detection_los(sm, robot_xy, wp_pos, n_samples=20):
    """True if the robot→waypoint line is clear of OCC cells.

    Checks 5–90 % of the line.  Also rejects detections when the robot itself
    is inside a perceived wall (physics clipping artefact).
    """
    if sample_occ_with_clearance(sm, robot_xy, radius=DETECTION_CLEARANCE_RADIUS):
        return False
    seg = wp_pos - robot_xy
    dist = float(np.linalg.norm(seg))
    if dist < 1e-6:
        return True
    n_samples = max(
        int(n_samples),
        int(math.ceil(dist / max(LINE_CHECK_STEP_M, sm.res * 0.5))),
    )
    for t in np.linspace(0.05, 0.90, n_samples):
        if sample_occ_with_clearance(
            sm,
            robot_xy + t * seg,
            radius=DETECTION_CLEARANCE_RADIUS,
        ):
            return False
    return True


@torch.no_grad()
def check_detections(z_current, head, bc_lats, found, seeking_idx,
                     seek_timeout_cd, glimpse_timeout_cd, detect_streaks, glimpse_streaks):
    candidates: List[Tuple[int, float]] = []
    for i, z_goals in enumerate(bc_lats):
        if found[i] or i == seeking_idx or seek_timeout_cd[i] > 0 or glimpse_timeout_cd[i] > 0:
            detect_streaks[i] = 0
            glimpse_streaks[i] = 0
            continue
        zc = z_current.expand(z_goals.shape[0], -1)
        e_min = float(head(zc, z_goals).min().item())
        candidates.append((i, e_min))

    if not candidates:
        return None, None, None, None

    candidates.sort(key=lambda x: x[1])
    best_i, best_e = candidates[0]
    second_e = candidates[1][1] if len(candidates) > 1 else float("inf")

    for i, _ in candidates:
        if i == best_i and best_e <= DETECT_ENERGY_THRESH and second_e >= best_e + DETECT_ENERGY_MARGIN:
            detect_streaks[i] += 1
        else:
            detect_streaks[i] = max(0, detect_streaks[i] - 1)
        if i == best_i and best_e <= GLIMPSE_ENERGY_THRESH and second_e >= best_e + GLIMPSE_ENERGY_MARGIN:
            glimpse_streaks[i] += 1
        else:
            glimpse_streaks[i] = max(0, glimpse_streaks[i] - 1)

    if detect_streaks[best_i] >= DETECT_CONFIRM_STEPS:
        detect_streaks[best_i] = 0
        glimpse_streaks[best_i] = 0
        return best_i, best_e, second_e, "spotted"
    if glimpse_streaks[best_i] >= GLIMPSE_CONFIRM_STEPS:
        glimpse_streaks[best_i] = 0
        return best_i, best_e, second_e, "glimpsed"
    return None, best_e, second_e, None


# --------------------------------------------------------------------------- #
# HUD
# --------------------------------------------------------------------------- #

def _wp_to_map(wp_xy, mx, my, mw, mh):
    nx = (float(wp_xy[0])-float(WORLD_MIN[0])) / max(float(WORLD_MAX[0]-WORLD_MIN[0]),1e-8)
    ny = (float(wp_xy[1])-float(WORLD_MIN[1])) / max(float(WORLD_MAX[1]-WORLD_MIN[1]),1e-8)
    return (mx+int(np.clip(nx,0,1)*mw), my+mh-int(np.clip(ny,0,1)*mh))


def draw_minimap(draw, sm, robot_xy, robot_yaw, target_xy, trail, plan_path,
                 waypoints, found, seeking_idx,
                 mx=514, my=494, mw=372, mh=155):
    draw.rectangle([mx, my, mx+mw, my+mh], fill=(18,18,18), outline=(95,95,95))

    for r in range(sm.h):
        for c in range(sm.w):
            x0 = mx+int(c/sm.w*mw);       y0 = my+mh-int((r+1)/sm.h*mh)
            x1 = mx+int((c+1)/sm.w*mw);   y1 = my+mh-int(r/sm.h*mh)
            v = int(sm.grid[r,c])
            fill = (48,48,48) if v==MAP_FREE else (180,70,70) if v==MAP_OCC else (25,25,25)
            draw.rectangle([x0,y0,x1,y1], fill=fill)

    if len(trail) > 1:
        pts = [_wp_to_map(t,mx,my,mw,mh) for t in trail[-300:]]
        draw.line(pts, fill=(255,220,80), width=2)

    if plan_path is not None and len(plan_path) > 1:
        pts = [_wp_to_map(t,mx,my,mw,mh) for t in plan_path]
        draw.line(pts, fill=(0,170,255), width=3)

    for i, wp in enumerate(waypoints):
        px, py = _wp_to_map(wp.pos, mx, my, mw, mh)
        r_int = tuple(int(x*255) for x in wp.color_rgb)
        if found[i]:
            draw.ellipse([px-6,py-6,px+6,py+6], fill=r_int, outline=(255,255,255), width=2)
        elif i == seeking_idx:
            draw.ellipse([px-7,py-7,px+7,py+7], fill=r_int, outline=(255,255,0), width=3)
        else:
            draw.ellipse([px-4,py-4,px+4,py+4], fill=(60,60,60), outline=(120,120,120), width=1)

    tx, ty = _wp_to_map(target_xy, mx, my, mw, mh)
    draw.ellipse([tx-5,ty-5,tx+5,ty+5], fill=(255,255,255), outline=(10,10,10), width=2)

    rx, ry = _wp_to_map(robot_xy, mx, my, mw, mh)
    fw = np.array([math.cos(robot_yaw), math.sin(robot_yaw)], np.float32)
    lf = np.array([math.cos(robot_yaw+2.5), math.sin(robot_yaw+2.5)], np.float32)
    rt = np.array([math.cos(robot_yaw-2.5), math.sin(robot_yaw-2.5)], np.float32)
    s  = 10.0
    tri = [(rx+int(fw[0]*s), ry-int(fw[1]*s)),
           (rx+int(lf[0]*s*0.8), ry-int(lf[1]*s*0.8)),
           (rx+int(rt[0]*s*0.8), ry-int(rt[1]*s*0.8))]
    draw.polygon(tri, fill=(255,255,255), outline=(10,10,10))


def compose_frame(over_rgb, eye_rgb, sm, robot_xy, robot_yaw, target_xy,
                  trail, plan_path, waypoints, found, seeking_idx,
                  status_lines, event_log=None):
    canvas = Image.new("RGB", (896, 660), (20, 20, 20))
    draw   = ImageDraw.Draw(canvas)

    # ── Header ─────────────────────────────────────────────────── #
    draw.rectangle([0, 0, 895, 55], fill=(12, 12, 12), outline=(55, 55, 55))
    draw.text((12, 19), "JEPA v2  Perceptual Maze Explorer", fill=(0, 200, 255))

    pill_x0, pill_w, pill_gap = 310, 94, 5
    for i, wp in enumerate(waypoints):
        bx     = pill_x0 + i * (pill_w + pill_gap)
        c_full = tuple(int(x * 255) for x in wp.color_rgb)
        label  = wp.name.split("-")[1]
        if found[i]:
            draw.rectangle([bx, 10, bx+pill_w, 45], fill=c_full, outline=(255,255,255), width=2)
            draw.text((bx+6, 21), f"[OK] {label}", fill=(0, 0, 0))
        elif i == seeking_idx:
            draw.rectangle([bx, 10, bx+pill_w, 45], fill=(30, 30, 30), outline=c_full, width=2)
            draw.text((bx+6, 21), f"[>>] {label}", fill=c_full)
        else:
            draw.rectangle([bx, 10, bx+pill_w, 45], fill=(25, 25, 25), outline=(55, 55, 55), width=1)
            draw.text((bx+6, 21), label, fill=(65, 65, 65))

    n_found = sum(found)
    draw.text((pill_x0 + 5*(pill_w+pill_gap) + 8, 21),
              f"{n_found}/{len(waypoints)}", fill=(160, 160, 160))

    # ── Overhead view (left) — clean ───────────────────────────── #
    canvas.paste(Image.fromarray(over_rgb[:,:,:3].astype(np.uint8)), (0, 56))

    # ── Eye view (right top) — clean ───────────────────────────── #
    canvas.paste(Image.fromarray(eye_rgb[:,:,:3].astype(np.uint8)).resize((384, 384)), (512, 56))

    # ── Right HUD panel (below eye view) ───────────────────────── #
    draw.rectangle([512, 440, 895, 659], fill=(14, 14, 14), outline=(55, 55, 55))
    for i, line in enumerate(status_lines):
        draw.text((520, 447 + i * 16), line, fill=(200, 200, 200))
    draw_minimap(draw, sm, robot_xy, robot_yaw, target_xy, trail, plan_path,
                 waypoints, found, seeking_idx)

    # ── Event log (below overhead view) ────────────────────────── #
    draw.rectangle([0, 568, 511, 659], fill=(14, 14, 14), outline=(55, 55, 55))
    draw.text((8, 572), "events", fill=(70, 70, 70))
    for i, (ev_type, ev_text) in enumerate((event_log or [])[-4:]):
        col = (255, 210, 60) if ev_type == "SPOTTED" else (80, 255, 120)
        draw.text((8, 587 + i * 18), ev_text, fill=col)

    return np.asarray(canvas)


# --------------------------------------------------------------------------- #
# PPO step helper
# --------------------------------------------------------------------------- #

@torch.no_grad()
def ppo_step(scene, robot, q0, prev_action, cmd, dofs, ppo, dev):
    obs    = get_sys1_obs(robot, q0, prev_action, cmd, dofs, dev)
    action = ppo.act_deterministic(obs).detach()
    robot.control_dofs_position(to_genesis_target(q0 + 0.3*action[0]), dofs)
    for _ in range(4): scene.step()
    return action


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="JEPA v2 Perceptual Maze Explorer")
    parser.add_argument("--jepa_ckpt", required=True)
    parser.add_argument("--head_ckpt", required=True)
    parser.add_argument("--ppo_ckpt",  required=True)
    parser.add_argument("--device",      type=str, default="auto",
                        help="Torch device: auto | cuda | cpu.")
    parser.add_argument("--sim_backend", type=str, default="auto",
                        help="Genesis backend: auto | amdgpu | vulkan | gpu | cuda | metal | cpu.")
    parser.add_argument("--n_steps",     type=int,   default=4000)
    parser.add_argument("--cands",       type=int,   default=512)
    parser.add_argument("--horizon",     type=int,   default=15)
    parser.add_argument("--map_res",     type=float, default=0.10)
    parser.add_argument("--depth_max",   type=float, default=1.80)
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--no_video",    action="store_true")
    parser.add_argument("--out", default="jepa_logs/maze_perceptual.mp4")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.")
        dev = torch.device("cpu")
    else:
        dev = torch.device(args.device)

    if dev.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print(f"Torch device: {dev}")
    if getattr(torch.version, "hip", None):
        print(f"PyTorch ROCm/HIP: {torch.version.hip}")

    # ── Load models ──────────────────────────────────────────────────────── #
    init_genesis_once(args.sim_backend)
    print(f"Genesis device: {gs.device}")

    jepa = CanonicalJEPA().to(dev)
    sd, _ = load_jepa_checkpoint(args.jepa_ckpt, device=dev)
    jepa.load_state_dict(sd, strict=False)
    jepa.eval()
    print(f"Loaded CanonicalJEPA from {args.jepa_ckpt}")

    head = GoalEnergyHead().to(dev)
    hckpt = torch.load(args.head_ckpt, map_location=dev)
    key   = "energy_head_state_dict" if "energy_head_state_dict" in hckpt else "model_state_dict"
    head.load_state_dict(clean_state_dict(hckpt[key]))
    head.eval()
    print(f"Loaded GoalEnergyHead from {args.head_ckpt}")

    ppo = ActorCritic().to(dev)
    ppo.load_state_dict(load_ppo_checkpoint(args.ppo_ckpt, device=dev), strict=False)
    ppo.eval()
    print(f"Loaded ActorCritic from {args.ppo_ckpt}")

    # ── Build scene ───────────────────────────────────────────────────────  #
    waypoints = make_maze_waypoints()
    scene, robot, cam_brain, cam_eye, cam_over, dofs, q0 = build_scene(waypoints)
    print(f"Scene built: {len(waypoints)} hidden beacons, {len(MAZE_WALL_SPECS)} maze walls")

    for _ in range(20): scene.step()

    # ── Harvest breadcrumbs ───────────────────────────────────────────────  #
    print("\nHarvesting latent breadcrumbs for each beacon ...")
    bc_dirs: List[np.ndarray]   = []
    bc_lats: List[torch.Tensor] = []
    for i, wp in enumerate(waypoints):
        print(f"  {wp.name}: {wp.pos}")
        d, z = harvest_breadcrumb(robot, cam_brain, q0, jepa, ppo, dofs, dev, scene, wp)
        bc_dirs.append(d)
        bc_lats.append(z)

    # ── Reset — the harvest teleports shouldn't pre-build the wall map ────  #
    sm = make_sensor_map(args.map_res)

    robot.set_pos(np.array(ROBOT_SPAWN, np.float32))
    robot.set_quat(yaw_to_quat(math.pi / 2))
    robot.set_dofs_position(q0.detach().cpu().numpy(), dofs)
    for _ in range(20): scene.step()
    print("  Done.\n")

    # ── Navigation state ─────────────────────────────────────────────────  #
    prev_action = torch.zeros((1,12), device=dev)
    prev_cmd:   Optional[torch.Tensor] = None
    place_memory_slots: Dict[Tuple[int, int, int], torch.Tensor] = {}
    latent_memory: Optional[torch.Tensor] = None
    latent_memory_norm: Optional[torch.Tensor] = None
    place_ema_latent: Optional[torch.Tensor] = None
    last_memory_xy = np.array(ROBOT_SPAWN[:2], np.float32)
    last_memory_key: Optional[Tuple[int, int, int]] = None

    trail:      List[np.ndarray] = []
    recent_pos: Deque[np.ndarray] = deque(maxlen=18)
    recent_cov: Deque[float]      = deque(maxlen=40)

    found            = [False] * len(waypoints)
    seeking_idx      = -1
    seek_steps       = 0
    seek_ema_e       = 4.0
    seek_timeout_cd  = [0] * len(waypoints)
    glimpse_timeout_cd = [0] * len(waypoints)
    seek_fail_counts = [0] * len(waypoints)
    detect_streaks   = [0] * len(waypoints)
    glimpse_streaks  = [0] * len(waypoints)
    seek_recent_pos:  Deque[np.ndarray] = deque(maxlen=SEEK_STALL_WINDOW)
    seek_recent_dist: Deque[float]      = deque(maxlen=SEEK_STALL_WINDOW)
    seek_recent_sig:  Deque[np.ndarray] = deque(maxlen=SEEK_STALL_WINDOW)
    seek_recovery_cd = 0
    seek_stall_count = 0   # consecutive stalls without escaping
    seek_start_dist  = 0.0  # d2g when seeking started
    guard_trip_pos:   Deque[np.ndarray] = deque(maxlen=12)

    # Hard long-stall escape: if robot hasn't moved far in a long window, force retreat.
    LONG_STALL_WINDOW = 150
    LONG_STALL_RADIUS = 0.30
    long_stall_pos: Deque[np.ndarray] = deque(maxlen=LONG_STALL_WINDOW)

    frontier_xy       = np.array([0.5, 0.5], np.float32)
    frontier_age      = 0
    cov_start         = 0.0
    FRONTIER_PATIENCE = 200
    frontier_bl:      List[np.ndarray] = []
    frontier_switches = 0
    guard_mode        = "none"
    guard_steps       = 0
    guard_cmd         = torch.zeros((1,3), device=dev)
    stuck_count       = 0
    stuck_cooldown    = 0
    clip_retreat_steps = 0

    discoveries: List[dict] = []
    event_log:   List[Tuple[str, str]] = []
    plan_path:   Optional[np.ndarray] = None
    nav_target:  np.ndarray = frontier_xy.copy()
    t0 = time.time()

    writer = None
    if not args.no_video:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        writer = imageio.get_writer(args.out, fps=30)

    print(f"Running perceptual maze exploration ({args.n_steps} steps)")
    print(f"  Candidates: {args.cands}  |  Horizon: {args.horizon}")
    print(f"  OCC log-odds threshold: {LOG_ODDS_OCC_THRESH}  |  Min OCC depth: {MIN_OCC_DEPTH}m")
    print()

    # ── Main loop ────────────────────────────────────────────────────────  #
    for step in range(args.n_steps):

        # Camera + robot pose.
        robot_pos_3d, robot_yaw, brain_pos, cam_pitch = move_cams(robot, cam_brain, cam_eye, cam_over)
        robot_xy = robot_pos_3d[:2].astype(np.float32)
        cam_height = float(max(brain_pos[2], 0.02))

        # ── Clip detection: physics has pushed robot into a perceived wall ─ #
        if sample_cell(sm, robot_xy) == MAP_OCC and clip_retreat_steps <= 0:
            clip_retreat_steps = 20

        # ── Hard long-stall escape: override everything if truly stuck ──── #
        long_stall_pos.append(robot_xy.copy())
        if (len(long_stall_pos) == long_stall_pos.maxlen
                and float(np.linalg.norm(long_stall_pos[-1] - long_stall_pos[0])) < LONG_STALL_RADIUS):
            # Force a large retreat in a random direction
            escape_yaw = robot_yaw + math.pi + np.random.uniform(-0.8, 0.8)
            escape_dir = np.array([math.cos(escape_yaw), math.sin(escape_yaw)], np.float32)
            retreat_xy = np.clip(robot_xy + escape_dir * 1.5, WORLD_MIN + 0.3, WORLD_MAX - 0.3)
            frontier_xy = retreat_xy.astype(np.float32)
            frontier_age = 0; cov_start = cov if step > 0 else 0.0
            frontier_bl.append(robot_xy.copy())
            guard_cmd = torch.tensor([[-0.20, 0.0, float(np.random.choice([-0.75, 0.75]))]], device=dev, dtype=torch.float32)
            guard_steps = 20
            stuck_cooldown = 80
            long_stall_pos.clear()
            if seeking_idx >= 0:
                seek_timeout_cd[seeking_idx] = max(seek_timeout_cd[seeking_idx], 90)
                seeking_idx = -1
                seek_recent_pos.clear(); seek_recent_dist.clear(); seek_recent_sig.clear()
            # Clear OCC around robot — camera may have clipped through wall
            mark_disc(sm, robot_xy, 0.25, MAP_FREE)
            print(f"\n  [LONG-STALL] Forced retreat at step {step}  pos=({robot_xy[0]:.2f},{robot_xy[1]:.2f})")

        # JEPA encode.
        vis, prop, _, depth = get_jepa_state(robot, cam_brain, q0, prev_action, dofs, dev)
        with torch.no_grad():
            z_current = jepa.encode_online(vis, prop).detach()
        if place_ema_latent is None:
            place_ema_latent = z_current.clone()
        else:
            place_ema_latent = (
                PLACE_LATENT_EMA_DECAY * place_ema_latent
                + (1.0 - PLACE_LATENT_EMA_DECAY) * z_current
            )

        # Update occupancy map from depth only.
        update_sensor_map_from_depth(
            sm, robot_xy, robot_yaw, depth,
            fov_deg=BRAIN_CAM_FOV_DEG, depth_max=args.depth_max,
            cam_pitch_rad=cam_pitch, cam_height_m=cam_height,
        )
        depth_sig = depth_view_signature(
            depth, args.depth_max,
            cam_pitch_rad=cam_pitch, cam_height_m=cam_height,
            fov_deg=BRAIN_CAM_FOV_DEG,
        )
        cov = coverage_percent(sm)
        trail.append(robot_xy.copy())
        recent_pos.append(robot_xy.copy())
        recent_cov.append(cov)

        if step == 0 or step % LATENT_MEMORY_STRIDE == 0:
            mem_key = place_memory_key(robot_xy, robot_yaw)
            mem_disp = float(np.linalg.norm(robot_xy - last_memory_xy))
            if (latent_memory is None
                    or mem_disp >= LATENT_MEMORY_MIN_STEP_DIST
                    or mem_key != last_memory_key):
                place_memory_slots, latent_memory, latent_memory_norm, _ = update_place_memory(
                    place_memory_slots, place_ema_latent, robot_xy, robot_yaw,
                )
                last_memory_xy = robot_xy.copy()
                last_memory_key = mem_key

        # ── Waypoint detection ─────────────────────────────────────────── #
        seek_timeout_cd = [max(0, c - 1) for c in seek_timeout_cd]
        glimpse_timeout_cd = [max(0, c - 1) for c in glimpse_timeout_cd]
        detected = None
        detect_e = None
        detect_second_e = None
        detect_kind = None
        if (seeking_idx < 0 and clip_retreat_steps <= 0
                and step % DETECT_STRIDE == 0):
            detected, detect_e, detect_second_e, detect_kind = check_detections(
                z_current, head, bc_lats, found, seeking_idx,
                seek_timeout_cd, glimpse_timeout_cd, detect_streaks, glimpse_streaks,
            )
            # Energy-head detections are trusted — no bearing or depth gate.
            # The JEPA latent space can detect beacons from peripheral views
            # and non-line-of-sight contexts that geometric gates would reject.

            # Diagnostic: log energy values when near an unfound beacon
            if step % 100 == 0:
                for i, wp in enumerate(waypoints):
                    if found[i] or i == seeking_idx:
                        continue
                    d2b = float(np.linalg.norm(wp.pos - robot_xy))
                    if d2b < DETECT_DIST + 0.5:
                        zc_diag = z_current.expand(bc_lats[i].shape[0], -1)
                        e_diag = float(head(zc_diag, bc_lats[i]).min().item())
                        bear = math.degrees(wrap_to_pi(
                            math.atan2(float(wp.pos[1] - robot_xy[1]),
                                       float(wp.pos[0] - robot_xy[0])) - robot_yaw))
                        print(f"  [ENERGY-DIAG] {wp.name} d={d2b:.1f}m  E={e_diag:.2f}  bear={bear:.0f}°"
                              f"  streak={glimpse_streaks[i]}  cd={glimpse_timeout_cd[i]}"
                              f"  thr={GLIMPSE_ENERGY_THRESH}")
        if detected is not None and seeking_idx < 0:
            detect_wp = waypoints[detected]
            seek_goal_xy = waypoint_seek_anchor(detect_wp)
            proxy_goal_xy = select_goal_proxy(sm, robot_xy, seek_goal_xy, frontier_bl)
            proxy_ready = (
                proxy_goal_xy is not None
                and float(np.linalg.norm(proxy_goal_xy - seek_goal_xy)) <= PROXY_ROUTE_RADIUS
            )
            # Allow routing through unknown cells — known walls still block.
            # This lets the robot seek beacons in unexplored territory without
            # falsely requiring the whole path to already be mapped free.
            route_ready = (
                (
                    sample_traversable_with_clearance(sm, seek_goal_xy, allow_unknown=True)
                    and (
                        _frontier_reachable(sm, robot_xy, seek_goal_xy, allow_unknown=True)
                        or bfs_next_waypoint(sm, robot_xy, seek_goal_xy,
                                             lookahead_m=0.7, allow_unknown=True,
                                             snap_goal_to_free=False) is not None
                    )
                )
                or proxy_ready
            )
            dist_spotted = float(np.linalg.norm(detect_wp.pos - robot_xy))
            if route_ready and detect_kind in ("spotted", "glimpsed"):
                seeking_idx    = detected
                seek_steps     = 0
                seek_ema_e     = 4.0
                prev_cmd       = None
                guard_steps    = 0
                stuck_cooldown = 0
                seek_recent_pos.clear()
                seek_recent_dist.clear()
                seek_recent_sig.clear()
                seek_recovery_cd = 0
                seek_stall_count = 0
                seek_start_dist  = dist_spotted
                print(f"\n  [{detect_kind.upper()}] {detect_wp.name} at step {step}  "
                      f"dist={dist_spotted:.2f}m  E={detect_e:.2f}")
                event_log.append((
                    detect_kind.upper(),
                    f"step {step:4d}  {detect_kind.upper()} {detect_wp.name}  d={dist_spotted:.1f}m  E={detect_e:.2f}",
                ))
            else:
                if proxy_goal_xy is not None:
                    frontier_xy = proxy_goal_xy.astype(np.float32)
                    frontier_age = 0
                    cov_start = cov
                glimpse_timeout_cd[detected] = max(glimpse_timeout_cd[detected], GLIMPSE_COOLDOWN)
                seek_timeout_cd[detected] = max(seek_timeout_cd[detected], 30)
                print(f"\n  [GLIMPSED] {detect_wp.name} at step {step}  "
                      f"dist={dist_spotted:.2f}m  E={detect_e:.2f}  — no clear path (known wall blocks)")
                event_log.append((
                    "GLIMPSED",
                    f"step {step:4d}  GLIMPSED {detect_wp.name}  d={dist_spotted:.1f}m  E={detect_e:.2f}",
                ))

        # ── Goal-direction latent selection ────────────────────────────── #
        if seeking_idx >= 0:
            seek_goal_xy = waypoint_seek_anchor(waypoints[seeking_idx])
            goal_vec_raw  = seek_goal_xy - robot_xy
            gv_norm = float(np.linalg.norm(goal_vec_raw))
            goal_vec_unit = goal_vec_raw/gv_norm if gv_norm > 1e-6 else np.array([1.,0.])
            dirs_k  = bc_dirs[seeking_idx]
            best_k  = int(np.argmax(dirs_k @ goal_vec_unit))
            z_goal  = bc_lats[seeking_idx][best_k:best_k+1]
            with torch.no_grad():
                raw_e = float(head(z_current, z_goal).item())
            seek_ema_e = 0.30*raw_e + 0.70*seek_ema_e
            dist_to_seek = float(gv_norm)
        else:
            dist_to_seek = 999.0
            raw_e = 0.0

        if seeking_idx >= 0:
            seek_recent_pos.append(robot_xy.copy())
            seek_recent_dist.append(dist_to_seek)
            if depth_sig is not None:
                seek_recent_sig.append(depth_sig.copy())
        else:
            seek_recent_pos.clear()
            seek_recent_dist.clear()
            seek_recent_sig.clear()
            seek_recovery_cd = 0

        # ── Arrival check ──────────────────────────────────────────────── #
        if seeking_idx >= 0 and dist_to_seek < ARRIVE_DIST:
            claimed_idx = seeking_idx
            wp = waypoints[claimed_idx]
            found[claimed_idx] = True
            seek_fail_counts[claimed_idx] = 0
            discoveries.append({"name": wp.name, "step": step,
                                 "dist": round(dist_to_seek, 3),
                                 "ema_energy": round(seek_ema_e, 3)})
            n_found = sum(found)
            print(f"\n  [CLAIMED] {wp.name} at step {step}  "
                  f"dist={dist_to_seek:.2f}  ({n_found}/{len(waypoints)} found)")
            event_log.append(("CLAIMED",
                               f"step {step:4d}  CLAIMED {wp.name}  ({n_found}/{len(waypoints)})"))
            seeking_idx = -1; prev_cmd = None
            seek_recent_pos.clear()
            seek_recent_dist.clear()
            seek_recent_sig.clear()
            seek_recovery_cd = 0

            if all(found):
                print("\n  All beacons claimed!  Route complete.")
                if writer:
                    over_rgb  = render_rgb(cam_over)
                    eye_rgb   = render_rgb(cam_eye)
                    frame = compose_frame(
                        over_rgb, eye_rgb, sm, robot_xy, robot_yaw,
                        robot_xy, trail, None, waypoints, found, -1,
                        [f"ALL {len(waypoints)} BEACONS FOUND — step {step}"],
                        event_log=event_log,
                    )
                    writer.append_data(frame)
                break

        # ── Seek timeout ───────────────────────────────────────────────── #
        if seeking_idx >= 0:
            seek_steps += 1
            if seek_steps > MAX_SEEK_STEPS:
                timed_idx = seeking_idx
                timed_wp = waypoints[timed_idx]
                seek_fail_counts[timed_idx] += 1
                cooldown = min(360, 120 + 60 * (seek_fail_counts[timed_idx] - 1))
                print(f"\n  [TIMEOUT] Gave up seeking {timed_wp.name} "
                      f"at step {step} — resuming exploration")
                seek_timeout_cd[timed_idx] = cooldown
                away = robot_xy - timed_wp.pos
                away_norm = float(np.linalg.norm(away))
                if away_norm > 1e-6:
                    away = away / away_norm
                retreat_xy = np.clip(robot_xy + away * 2.0, WORLD_MIN + 0.3, WORLD_MAX - 0.3)
                frontier_xy = retreat_xy.astype(np.float32)
                frontier_age = 0; cov_start = cov
                seeking_idx = -1; prev_cmd = None
                seek_recent_pos.clear()
                seek_recent_dist.clear()
                seek_recent_sig.clear()
                seek_recovery_cd = 0

        # ── Seek stall recovery ────────────────────────────────────────── #
        if seek_recovery_cd > 0:
            seek_recovery_cd -= 1
        if (seeking_idx >= 0 and guard_steps <= 0
                and clip_retreat_steps <= 0 and seek_recovery_cd <= 0
                and len(seek_recent_pos) == seek_recent_pos.maxlen
                and len(seek_recent_dist) == seek_recent_dist.maxlen):
            seek_disp = float(np.linalg.norm(seek_recent_pos[-1] - seek_recent_pos[0]))
            seek_progress = float(seek_recent_dist[0] - seek_recent_dist[-1])
            sig_delta = 999.0
            front_blocked = front_blocked_from_depth(
                depth, args.depth_max,
                cam_pitch_rad=cam_pitch, cam_height_m=cam_height,
                fov_deg=BRAIN_CAM_FOV_DEG,
            )
            if len(seek_recent_sig) == seek_recent_sig.maxlen:
                sig_delta = float(np.max(np.abs(seek_recent_sig[-1] - seek_recent_sig[0])))
            if (seek_disp < SEEK_STALL_DISP
                    and seek_progress < SEEK_STALL_PROGRESS):
                reinforce_front_obstacle(
                    sm, robot_xy, robot_yaw, depth, args.depth_max,
                    cam_pitch_rad=cam_pitch, cam_height_m=cam_height,
                    fov_deg=BRAIN_CAM_FOV_DEG,
                )
                guard_cmd = make_escape_cmd_from_depth(
                    depth, args.depth_max, dev,
                    reverse_speed=-0.18, turn_speed=0.60,
                    cam_pitch_rad=cam_pitch, cam_height_m=cam_height,
                    fov_deg=BRAIN_CAM_FOV_DEG,
                )
                seek_stall_count += 1
                guard_steps = 12
                seek_steps = min(MAX_SEEK_STEPS, seek_steps + 35)
                seek_recovery_cd = SEEK_RECOVERY_COOLDOWN
                seek_recent_pos.clear()
                seek_recent_dist.clear()
                seek_recent_sig.clear()
                print(f"\n  [SEEK-STALL] {waypoints[seeking_idx].name} at step {step}  "
                      f"disp={seek_disp:.2f}m  progress={seek_progress:.2f}m  "
                      f"view={sig_delta:.3f}  stalls={seek_stall_count} — backing off")
                event_log.append((
                    "SEEK-STALL",
                    f"step {step:4d}  SEEK-STALL {waypoints[seeking_idx].name}  "
                    f"disp={seek_disp:.2f} progress={seek_progress:.2f}",
                ))
                # If we've stalled many times and haven't gotten closer, give up
                if seek_stall_count >= 6:
                    ab_idx = seeking_idx
                    ab_wp  = waypoints[ab_idx]
                    seek_fail_counts[ab_idx] += 1
                    cooldown = min(360, 120 + 60 * (seek_fail_counts[ab_idx] - 1))
                    print(f"\n  [SEEK-ABANDON] {ab_wp.name} at step {step}  "
                          f"stalls={seek_stall_count}  start_dist={seek_start_dist:.2f}m — wall-blocked")
                    seek_timeout_cd[ab_idx] = cooldown
                    away = robot_xy - ab_wp.pos
                    away_norm = float(np.linalg.norm(away))
                    if away_norm > 1e-6:
                        away = away / away_norm
                    retreat_xy = np.clip(robot_xy + away * 2.0, WORLD_MIN + 0.3, WORLD_MAX - 0.3)
                    frontier_xy = retreat_xy.astype(np.float32)
                    frontier_age = 0; cov_start = cov
                    seeking_idx = -1; prev_cmd = None
                    guard_steps = 0
                    seek_recent_pos.clear()
                    seek_recent_dist.clear()
                    seek_recent_sig.clear()
                    seek_recovery_cd = 0
                    event_log.append((
                        "SEEK-ABANDON",
                        f"step {step:4d}  SEEK-ABANDON {ab_wp.name}  stalls={seek_stall_count}",
                    ))

        # ── Command selection ──────────────────────────────────────────── #
        energy_scale = 1.0  # updated inside seek branch; readable by log line
        if clip_retreat_steps > 0:
            cmd = torch.tensor([[-0.25, 0.0, 0.0]], device=dev, dtype=torch.float32)
            clip_retreat_steps -= 1; plan_path = None
        elif guard_steps > 0:
            cmd = guard_cmd; guard_steps -= 1; plan_path = None
        elif seeking_idx >= 0:
            # Goal-directed: JEPA energy + depth-based obstacle avoidance.
            # No BFS — the world model guides direction; the depth penalty
            # in plan_seek_cmd stops the robot from driving into walls the
            # camera sees but that may not yet be in the occupancy map.
            wp    = waypoints[seeking_idx]
            seek_goal_xy = waypoint_seek_anchor(wp)
            seek_nav = seek_goal_xy
            energy_scale = 1.0

            navvec  = seek_nav - robot_xy
            navdist = float(np.linalg.norm(navvec))
            navdir  = navvec / max(navdist, 1e-8)
            gbody   = world_to_body_xy(robot_yaw, navdir)
            gang    = math.atan2(float(navvec[1]), float(navvec[0]))
            herr    = wrap_to_pi(gang - robot_yaw)
            cmd, plan_path = plan_seek_cmd(
                jepa, head, z_current, z_goal, sm,
                robot_xy, robot_yaw, seek_nav, gbody,
                navdist, herr, args.cands, args.horizon, dev, prev_cmd,
                energy_scale=energy_scale,
                depth_sig=depth_sig,
            )
            prev_cmd = cmd.clone()
        else:
            # Frontier exploration.
            frontier_age += 1
            cov_gain = cov - cov_start
            force_new = (frontier_age >= FRONTIER_PATIENCE and cov_gain < 0.5)
            if force_new:
                frontier_bl.append(frontier_xy.copy())
                frontier_age = 0; cov_start = cov

            new_f, _ = select_frontier(sm, robot_xy, frontier_bl)
            if not _frontier_reachable(sm, robot_xy, new_f):
                reachable = None
                best_score = float("-inf")
                for r in range(1, sm.h-1):
                    for c in range(1, sm.w-1):
                        if sm.grid[r,c] != MAP_FREE:
                            continue
                        wp = grid_to_world(sm, (r, c))
                        if any(float(np.linalg.norm(wp - bp)) < 0.40 for bp in frontier_bl):
                            continue
                        if float(np.linalg.norm(wp - robot_xy)) < 0.20:
                            continue
                        if not _frontier_reachable(sm, robot_xy, wp):
                            continue
                        unk = sum(1 for rr in range(r-1,r+2) for cc in range(c-1,c+2)
                                  if not (rr==r and cc==c) and sm.grid[rr,cc]==MAP_UNKNOWN)
                        occ = sum(1 for rr in range(r-1,r+2) for cc in range(c-1,c+2)
                                  if not (rr==r and cc==c) and sm.grid[rr,cc]==MAP_OCC)
                        dist = float(np.linalg.norm(wp - robot_xy))
                        visit_pen = min(FRONTIER_VISIT_PENALTY * local_visit_score(sm, (r, c)), 1.8)
                        score = 0.45*float(unk) - 0.30*dist - 0.35*float(occ) + 2.0 - visit_pen
                        if score > best_score:
                            best_score = score
                            reachable = wp
                if reachable is not None:
                    new_f = reachable
                else:
                    new_f = find_far_unknown(sm, robot_xy)
                    frontier_bl = frontier_bl[-4:]

            if float(np.linalg.norm(new_f - frontier_xy)) > 0.50 or force_new:
                frontier_switches += 1
                frontier_xy = new_f; frontier_age = 0; cov_start = cov

            if not _frontier_reachable(sm, robot_xy, frontier_xy):
                bfs_wp = bfs_next_waypoint(sm, robot_xy, frontier_xy)
                nav_target = bfs_wp if bfs_wp is not None else frontier_xy
            else:
                nav_target = frontier_xy

            cmd, plan_path = plan_explore_cmd(
                jepa, z_current, latent_memory_norm, sm, robot_xy, robot_yaw,
                nav_target, prev_cmd, args.cands, args.horizon, dev,
            )
            prev_cmd = cmd.clone()

        # ── Stuck detection (skipped during seek — use a short safety retreat) ─── #
        disp = (float(np.linalg.norm(recent_pos[-1]-recent_pos[0]))
                if len(recent_pos) >= recent_pos.maxlen else 1.0)
        if stuck_cooldown > 0:
            stuck_cooldown -= 1
        if guard_steps <= 0 and stuck_cooldown <= 0 and disp < 0.08 and seeking_idx < 0:
            stuck_count += 1
            frontier_bl.append(frontier_xy.copy())
            if stuck_count >= 3:
                guard_cmd = make_escape_cmd_from_depth(
                    depth, args.depth_max, dev,
                    reverse_speed=-0.22, turn_speed=0.80,
                    cam_pitch_rad=cam_pitch, cam_height_m=cam_height,
                    fov_deg=BRAIN_CAM_FOV_DEG,
                )
                guard_steps = 12
                stuck_count = 0
                stuck_cooldown = 65
                frontier_bl = frontier_bl[-12:]
                frontier_xy, _ = select_frontier(sm, robot_xy, frontier_bl)
                frontier_age = 0
                cov_start = cov
                prev_cmd = None
                plan_path = None
            else:
                depth_now = depth
                if depth_now is not None:
                    h, w = depth_now.shape[:2]
                    band  = depth_now[int(0.45*h):int(0.90*h), :]
                    left  = float(np.nanmedian(band[:, :max(1, w//3)]))
                    right = float(np.nanmedian(band[:, 2*w//3:]))
                    if left > right + 0.10:
                        guard_cmd = torch.tensor([[-0.15, 0.0, -0.55]], device=dev, dtype=torch.float32)
                    elif right > left + 0.10:
                        guard_cmd = torch.tensor([[-0.15, 0.0,  0.55]], device=dev, dtype=torch.float32)
                    else:
                        guard_cmd = torch.tensor([[-0.15, 0.0,  0.60]], device=dev, dtype=torch.float32)
                else:
                    guard_cmd = torch.tensor([[-0.15, 0.0, 0.55]], device=dev, dtype=torch.float32)
                guard_steps = 16; stuck_cooldown = 55
        else:
            if disp >= 0.12: stuck_count = max(0, stuck_count - 1)

        # ── Perception safety filter (seek + explore) ─────────────────── #
        if guard_steps <= 0 and clip_retreat_steps <= 0:
            left_vec = np.array([-math.sin(robot_yaw),  math.cos(robot_yaw)], np.float32)
            depth_stats = front_depth_guard_stats(
                depth, args.depth_max,
                cam_pitch_rad=cam_pitch, cam_height_m=cam_height,
                fov_deg=BRAIN_CAM_FOV_DEG,
            )
            depth_blocked = (
                depth_stats is not None
                and (
                    (
                        depth_stats["center_hit_q35"] is not None
                        and depth_stats["center_hit_q35"] < FRONT_STOP_DIST
                    )
                    or depth_stats["center_close_frac"] > FRONT_BLOCKED_FRAC
                )
            )
            depth_clearance = (
                args.depth_max if depth_stats is None
                else float(
                    depth_stats["center_hit_q35"]
                    if depth_stats["center_hit_q35"] is not None
                    else depth_stats["center_clear_q35"]
                )
            )
            occ_hits = sample_front_occ_hits(sm, robot_xy, robot_yaw)
            occ_near = int(occ_hits[0]) + int(occ_hits[1])
            map_immediate_blocked = (
                occ_near >= 2
                and (depth_stats is None or depth_clearance < FRONT_MAP_CONFIRM_DIST)
            )
            left_occ = sample_occ_with_clearance(sm, robot_xy + 0.14 * left_vec, radius=0.08)
            right_occ = sample_occ_with_clearance(sm, robot_xy - 0.14 * left_vec, radius=0.08)

            # Zero lateral "wall scraping" commands and turn away from the close wall.
            if abs(float(cmd[0, 1].item())) > 0.05 and (left_occ or right_occ):
                turn = (0.45 if left_occ and not right_occ
                        else -0.45 if right_occ and not left_occ
                        else float(cmd[0, 2].item()))
                cmd = torch.tensor([[
                    min(float(cmd[0, 0].item()), 0.18),
                    0.0,
                    turn,
                ]], device=dev, dtype=torch.float32)
                prev_cmd = cmd.clone()
                plan_path = None

            if float(cmd[0, 0].item()) > 0.02 and (depth_blocked or map_immediate_blocked):
                if depth_blocked:
                    reinforce_front_obstacle(
                        sm, robot_xy, robot_yaw, depth, args.depth_max,
                        cam_pitch_rad=cam_pitch, cam_height_m=cam_height,
                        fov_deg=BRAIN_CAM_FOV_DEG,
                    )
                cmd = make_escape_cmd_from_depth(
                    depth, args.depth_max, dev,
                    reverse_speed=(-0.16 if seeking_idx >= 0 else -0.12),
                    turn_speed=0.65,
                    cam_pitch_rad=cam_pitch, cam_height_m=cam_height,
                    fov_deg=BRAIN_CAM_FOV_DEG,
                )
                guard_steps = 12 if seeking_idx >= 0 else 8
                guard_steps -= 1
                if seeking_idx >= 0:
                    seek_recent_pos.clear()
                    seek_recent_dist.clear()
                    seek_recent_sig.clear()
                    seek_recovery_cd = max(seek_recovery_cd, 8)
                    seek_steps = min(MAX_SEEK_STEPS, seek_steps + 12)
                guard_trip_pos.append(robot_xy.copy())
                local_guard_hits = sum(
                    1 for p in guard_trip_pos
                    if float(np.linalg.norm(p - robot_xy)) < 0.18
                )
                if local_guard_hits >= 4:
                    frontier_bl.append(frontier_xy.copy())
                    frontier_xy, _ = select_frontier(sm, robot_xy, frontier_bl[-12:])
                    frontier_age = 0
                    cov_start = cov
                    guard_cmd = make_escape_cmd_from_depth(
                        depth, args.depth_max, dev,
                        reverse_speed=-0.22, turn_speed=0.80,
                        cam_pitch_rad=cam_pitch, cam_height_m=cam_height,
                        fov_deg=BRAIN_CAM_FOV_DEG,
                    )
                    guard_steps = 16
                    stuck_cooldown = max(stuck_cooldown, 60)
                    if seeking_idx >= 0:
                        seek_timeout_cd[seeking_idx] = max(seek_timeout_cd[seeking_idx], 60)
                        seeking_idx = -1
                        seek_recent_pos.clear()
                        seek_recent_dist.clear()
                        seek_recent_sig.clear()
                        seek_recovery_cd = 0
                    prev_cmd = None
                    cmd = guard_cmd
                prev_cmd = cmd.clone()
                plan_path = None

        # ── Physics step ──────────────────────────────────────────────── #
        prev_action = ppo_step(scene, robot, q0, prev_action, cmd, dofs, ppo, dev)

        # ── Video frame ───────────────────────────────────────────────── #
        if writer and step % 2 == 0:
            over_rgb = render_rgb(cam_over)
            eye_rgb  = render_rgb(cam_eye)
            n_found  = sum(found)
            if seeking_idx >= 0:
                mode_str = f"SEEK -> {waypoints[seeking_idx].name}  E={seek_ema_e:.2f}"
                tgt_xy   = waypoints[seeking_idx].pos
            else:
                mode_str = f"EXPLORE  cov={cov:.1f}%"
                tgt_xy   = frontier_xy
            frame = compose_frame(
                over_rgb, eye_rgb, sm, robot_xy, robot_yaw, tgt_xy,
                trail, plan_path,
                waypoints, found, seeking_idx,
                [
                    f"step {step}/{args.n_steps}   cov={cov:.1f}%",
                    f"mode: {mode_str}",
                    f"pos ({robot_xy[0]:.2f},{robot_xy[1]:.2f})  yaw={math.degrees(robot_yaw):+.0f}deg",
                ],
                event_log=event_log,
            )
            writer.append_data(frame)

        if step % 50 == 0:
            n_found = sum(found)
            cv = cmd[0].cpu().numpy()
            if guard_steps > 0:
                mode_dbg = f"GUARD({guard_steps})"
            elif seeking_idx >= 0:
                _d_fwd = f"  dfwd={float(depth_sig[1]):.2f}m" if depth_sig is not None else ""
                mode_dbg = f"SEEK:{waypoints[seeking_idx].name}  d2g={dist_to_seek:.2f}m{_d_fwd}"
            else:
                hdg_to_f = wrap_to_pi(
                    math.atan2(float(frontier_xy[1]-robot_xy[1]),
                               float(frontier_xy[0]-robot_xy[0])) - robot_yaw)
                via_str = (f" via=({nav_target[0]:.2f},{nav_target[1]:.2f})"
                           if float(np.linalg.norm(nav_target - frontier_xy)) > 0.15 else "")
                mode_dbg = f"EXPLORE  front=({frontier_xy[0]:.2f},{frontier_xy[1]:.2f}){via_str}  hdg={math.degrees(hdg_to_f):+.0f}deg"
            print(f"  step {step:4d}  pos=({robot_xy[0]:.2f},{robot_xy[1]:.2f})"
                  f"  yaw={math.degrees(robot_yaw):+.0f}deg"
                  f"  cmd=[{cv[0]:+.2f},{cv[1]:+.2f},{cv[2]:+.2f}]"
                  f"  disp={disp:.2f}  cov={cov:.1f}%  cd={stuck_cooldown}"
                  f"  mem={0 if latent_memory is None else int(latent_memory.shape[0])}"
                  f"  {mode_dbg}")

    if writer: writer.close()

    elapsed = time.time() - t0
    n_found = sum(found)
    print(f"\nPerceptual Maze Explorer Summary")
    print(f"  Steps used : {step+1}/{args.n_steps}")
    print(f"  Elapsed    : {elapsed:.1f}s")
    print(f"  Coverage   : {coverage_percent(sm):.1f}%")
    print(f"  Beacons    : {n_found}/{len(waypoints)}")
    for d in discoveries:
        print(f"    {d['name']:12s} at step {d['step']:4d}  "
              f"dist={d['dist']:.2f}  E={d['ema_energy']:.2f}")
    print(f"\nVideo saved to {args.out}")


if __name__ == "__main__":
    main()
