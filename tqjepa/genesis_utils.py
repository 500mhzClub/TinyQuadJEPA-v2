"""Genesis simulation helpers."""
from __future__ import annotations

import numpy as np
import torch


def resolve_sim_backend(sim_backend_arg: str):
    """Return a (backend, description) tuple for gs.init()."""
    import genesis as gs  # noqa: delay import

    sim_backend_arg = sim_backend_arg.lower().strip()
    gpu_backend = getattr(gs, "gpu", getattr(gs, "cuda", None))

    if sim_backend_arg in ("cpu",):
        return gs.cpu, "CPU requested"
    if sim_backend_arg in ("gpu", "cuda"):
        return gpu_backend if gpu_backend is not None else gs.cpu, "GPU requested"
    if sim_backend_arg == "auto":
        if gpu_backend is not None:
            return gpu_backend, "AUTO (GPU preferred)"
        return gs.cpu, "AUTO (CPU fallback)"
    return gs.cpu, "CPU fallback"


def init_genesis_once(sim_backend_arg: str = "auto") -> None:
    import genesis as gs  # noqa: delay import
    backend, msg = resolve_sim_backend(sim_backend_arg)
    print(f"Initialising Genesis ({msg}) ...")
    gs.init(backend=backend)


def to_genesis_target(x: torch.Tensor) -> torch.Tensor:
    """Detach, move to CPU numpy, then re-wrap for Genesis device."""
    import genesis as gs  # noqa: delay import
    x_np = x.detach().to("cpu").numpy().astype(np.float32, copy=True)
    return torch.tensor(x_np, device=gs.device, dtype=torch.float32)


def to_numpy(x) -> np.ndarray | None:
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)
