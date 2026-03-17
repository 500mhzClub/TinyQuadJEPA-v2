"""Random obstacle layout generation and collision detection."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class ObstacleSpec:
    """Axis-aligned box obstacle."""
    pos: Tuple[float, float, float]
    size: Tuple[float, float, float]
    color: Tuple[float, float, float] = (0.55, 0.55, 0.60)


@dataclass
class ObstacleLayout:
    """A full obstacle configuration for one chunk / scene."""
    obstacles: List[ObstacleSpec] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps([asdict(o) for o in self.obstacles])

    @staticmethod
    def from_json(s: str) -> "ObstacleLayout":
        data = json.loads(s)
        return ObstacleLayout([ObstacleSpec(**d) for d in data])


def random_obstacle_color() -> Tuple[float, float, float]:
    """Sample a muted obstacle color."""
    base = np.random.uniform(0.3, 0.7)
    tint = np.random.uniform(-0.1, 0.1, size=3)
    c = np.clip(base + tint, 0.1, 0.9)
    return (float(c[0]), float(c[1]), float(c[2]))


def generate_random_layout(
    n_range: Tuple[int, int] = (3, 8),
    size_range: Tuple[float, float] = (0.15, 0.40),
    height_range: Tuple[float, float] = (0.10, 0.50),
    spawn_radius: Tuple[float, float] = (0.5, 2.5),
    robot_clearance: float = 0.40,
    seed: int | None = None,
) -> ObstacleLayout:
    """Generate a random obstacle layout for one scene.

    Obstacles are placed in a ring around the origin at varying distances.
    A clearance zone around (0, 0) is kept free so the robot can spawn safely.
    """
    rng = np.random.RandomState(seed)
    n = rng.randint(n_range[0], n_range[1] + 1)

    obstacles: List[ObstacleSpec] = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(spawn_radius[0], spawn_radius[1])
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)

        # Skip if too close to robot spawn.
        if math.sqrt(x ** 2 + y ** 2) < robot_clearance:
            continue

        sx = rng.uniform(size_range[0], size_range[1])
        sy = rng.uniform(size_range[0], size_range[1])
        sz = rng.uniform(height_range[0], height_range[1])
        z = sz / 2.0

        obstacles.append(ObstacleSpec(
            pos=(float(x), float(y), float(z)),
            size=(float(sx), float(sy), float(sz)),
            color=random_obstacle_color(),
        ))

    return ObstacleLayout(obstacles)


def add_obstacles_to_scene(scene, layout: ObstacleLayout) -> None:
    """Add all obstacles from a layout to a Genesis scene (before build)."""
    import genesis as gs  # noqa: delay import
    for obs in layout.obstacles:
        scene.add_entity(
            gs.morphs.Box(pos=obs.pos, size=obs.size, fixed=True),
            surface=gs.surfaces.Rough(color=obs.color),
        )


def detect_collisions(
    robot_pos_xy: torch.Tensor,
    layout: ObstacleLayout,
    margin: float = 0.15,
) -> torch.Tensor:
    """Per-env boolean: is the robot within margin of any obstacle AABB?

    Args:
        robot_pos_xy: (N, 2) tensor of robot XY positions.
        layout: obstacle layout for this scene.
        margin: clearance buffer (metres).

    Returns:
        (N,) bool tensor — True if the robot is clipping / too close.
    """
    N = robot_pos_xy.shape[0]
    colliding = torch.zeros(N, dtype=torch.bool, device=robot_pos_xy.device)

    for obs in layout.obstacles:
        cx, cy = obs.pos[0], obs.pos[1]
        hx, hy = obs.size[0] / 2.0 + margin, obs.size[1] / 2.0 + margin
        in_x = (robot_pos_xy[:, 0] > cx - hx) & (robot_pos_xy[:, 0] < cx + hx)
        in_y = (robot_pos_xy[:, 1] > cy - hy) & (robot_pos_xy[:, 1] < cy + hy)
        colliding |= in_x & in_y

    return colliding
