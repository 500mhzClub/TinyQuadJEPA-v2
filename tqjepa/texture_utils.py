"""Procedural texture generation for visual domain randomization."""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from PIL import Image


def _save(img: np.ndarray, path: str) -> str:
    Image.fromarray(img).save(path)
    return os.path.abspath(path)


def make_checkerboard(
    grid: int = 16,
    color_a: Tuple[int, int, int] = (255, 255, 255),
    color_b: Tuple[int, int, int] = (40, 40, 40),
    path: str = "checker.png",
    res: int = 1024,
) -> str:
    img = np.zeros((res, res, 3), dtype=np.uint8)
    for i in range(res):
        for j in range(res):
            c = color_a if ((i // grid) + (j // grid)) % 2 == 0 else color_b
            img[i, j] = c
    return _save(img, path)


def make_stripes(
    width: int = 20,
    horizontal: bool = True,
    color_a: Tuple[int, int, int] = (200, 200, 200),
    color_b: Tuple[int, int, int] = (60, 60, 60),
    path: str = "stripes.png",
    res: int = 1024,
) -> str:
    img = np.zeros((res, res, 3), dtype=np.uint8)
    for i in range(res):
        idx = i if horizontal else 0
        for j in range(res):
            idx2 = i if horizontal else j
            c = color_a if (idx2 // width) % 2 == 0 else color_b
            img[i, j] = c
    return _save(img, path)


def make_noise_texture(
    path: str = "noise.png",
    res: int = 1024,
    scale: float = 0.3,
) -> str:
    base = np.random.randint(80, 180, size=3, dtype=np.uint8)
    noise = (np.random.randn(res, res, 3) * 255 * scale).astype(np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return _save(img, path)


def make_solid(
    color: Tuple[int, int, int] = (160, 140, 120),
    path: str = "solid.png",
    res: int = 1024,
) -> str:
    img = np.full((res, res, 3), color, dtype=np.uint8)
    return _save(img, path)


def generate_texture_set(output_dir: str, count: int = 10) -> List[str]:
    """Generate a diverse set of ground textures and return their paths."""
    os.makedirs(output_dir, exist_ok=True)
    textures: List[str] = []

    checker_configs = [
        (16, (255, 255, 255), (40, 40, 40)),
        (32, (200, 180, 140), (120, 100, 70)),
        (64, (180, 200, 180), (60, 80, 60)),
        (24, (200, 200, 220), (80, 80, 100)),
    ]
    for i, (grid, ca, cb) in enumerate(checker_configs):
        textures.append(make_checkerboard(grid, ca, cb, f"{output_dir}/checker_{i}.png"))

    stripe_configs = [
        (20, True, (200, 200, 200), (60, 60, 60)),
        (30, False, (180, 160, 140), (100, 80, 60)),
    ]
    for i, (w, h, ca, cb) in enumerate(stripe_configs):
        textures.append(make_stripes(w, h, ca, cb, f"{output_dir}/stripes_{i}.png"))

    for i in range(2):
        textures.append(make_noise_texture(f"{output_dir}/noise_{i}.png"))

    solid_colors = [(160, 140, 120), (100, 110, 100)]
    for i, c in enumerate(solid_colors):
        textures.append(make_solid(c, f"{output_dir}/solid_{i}.png"))

    return textures[:count]
