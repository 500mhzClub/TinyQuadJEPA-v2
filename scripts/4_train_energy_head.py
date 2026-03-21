#!/usr/bin/env python3
"""Train a GoalEnergyHead on frozen CanonicalJEPA representations.

Adapted from v1 but uses the new student-teacher (online/target encoder)
architecture.  The predictor maps from online-encoder space into target-encoder
space, so ``z_pred`` and ``z_goal`` (encoded via the target encoder) are
directly comparable.

Example:
    python scripts/4_train_energy_head.py \
        --jepa_ckpt jepa_checkpoints/jepa_best.pt \
        --data_dir jepa_final_dataset \
        --device cuda
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from tqjepa.models import CanonicalJEPA, GoalEnergyHead
from tqjepa.data import StreamingJEPADataset
from tqjepa.checkpoint_utils import load_jepa_checkpoint

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ── Latent extraction / cache ─────────────────────────────────────────


def _extract_latents(
    loader: DataLoader,
    model: CanonicalJEPA,
    device: torch.device,
    horizon: int,
    amp_enabled: bool,
    desc: str = "Extracting latents",
) -> Tuple[np.ndarray, np.ndarray]:
    """One-pass extraction of (z_pred, z_goal) pairs from a frozen backbone."""
    z_preds, z_goals = [], []
    with torch.no_grad():
        for vis, prop, cmds, _dones, _cols in tqdm(loader, desc=desc):
            vis = vis.to(device, non_blocking=True).float().div_(255.0)
            prop = prop.to(device, non_blocking=True)
            cmds = cmds.to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=amp_enabled):
                zp, zg = rollout_terminal_latent(model, vis, prop, cmds, horizon)
            z_preds.append(zp.float().cpu().numpy())
            z_goals.append(zg.float().cpu().numpy())
    return np.concatenate(z_preds, axis=0), np.concatenate(z_goals, axis=0)


@torch.no_grad()
def run_validation_cached(
    head: GoalEnergyHead,
    z_pred_val: torch.Tensor,
    z_goal_val: torch.Tensor,
    device: torch.device,
    batch_size: int,
    num_negatives: int,
    margin: float,
    reg_weight: float,
    max_batches: int,
) -> StepStats:
    head.eval()
    losses, pos_vals, neg_vals, gaps, accs = [], [], [], [], []
    n = z_pred_val.shape[0]
    for b0 in range(0, min(n, max_batches * batch_size), batch_size):
        zp = z_pred_val[b0 : b0 + batch_size].to(device)
        zg = z_goal_val[b0 : b0 + batch_size].to(device)
        z_neg = sample_negative_goals(zg, num_negatives)
        _, stats = energy_ranking_loss(head, zp, zg, z_neg, margin, reg_weight)
        losses.append(stats.loss)
        pos_vals.append(stats.pos_energy)
        neg_vals.append(stats.neg_energy)
        gaps.append(stats.gap)
        accs.append(stats.ranking_acc)
    if not losses:
        return StepStats(math.nan, math.nan, math.nan, math.nan, math.nan)
    n = len(losses)
    return StepStats(
        loss=sum(losses) / n,
        pos_energy=sum(pos_vals) / n,
        neg_energy=sum(neg_vals) / n,
        gap=sum(gaps) / n,
        ranking_acc=sum(accs) / n,
    )


# ── Helpers ──────────────────────────────────────────────────────────


@dataclass
class StepStats:
    loss: float
    pos_energy: float
    neg_energy: float
    gap: float
    ranking_acc: float


def rollout_terminal_latent(
    model: CanonicalJEPA,
    vis: torch.Tensor,
    prop: torch.Tensor,
    cmds: torch.Tensor,
    horizon: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode start via online encoder, roll predictor H steps, encode goal via target encoder.

    Returns ``(z_pred_H, z_goal)`` both living in target-encoder space.
    """
    # Starting state through the online encoder (predictor input).
    z_roll = model.encode_online(vis[:, 0], prop[:, 0])
    h_t = torch.zeros(
        z_roll.shape[0], model.latent_dim,
        device=z_roll.device, dtype=z_roll.dtype,
    )

    # Roll the predictor forward H steps under the recorded commands.
    for t in range(horizon):
        z_roll, h_t = model.predictor(z_roll, cmds[:, t], h_t)

    # Goal through the target encoder — same space as predictor output.
    z_goal = model.encode_target(vis[:, horizon], prop[:, horizon])
    return z_roll, z_goal


def sample_negative_goals(z_goal: torch.Tensor, num_negatives: int) -> torch.Tensor:
    """Vectorised in-batch negative sampling (no CPU-GPU sync).

    Returns a ``[B, K, D]`` tensor of shuffled goal latents.
    """
    bsz = z_goal.shape[0]
    if bsz <= 1:
        return z_goal.unsqueeze(1).expand(-1, num_negatives, -1)

    # Random offsets in [1, bsz-1] guarantee no self-match.
    offsets = torch.randint(1, bsz, (bsz, num_negatives), device=z_goal.device)
    base_idx = torch.arange(bsz, device=z_goal.device).unsqueeze(1)
    neg_idx = (base_idx + offsets) % bsz
    return z_goal[neg_idx]  # [B, K, D]


def energy_ranking_loss(
    head: GoalEnergyHead,
    z_pred: torch.Tensor,
    z_goal: torch.Tensor,
    z_neg: torch.Tensor,
    margin: float,
    reg_weight: float,
) -> Tuple[torch.Tensor, StepStats]:
    bsz, k_neg, dim = z_neg.shape

    pos_energy = head(z_pred, z_goal)  # [B]

    z_pred_rep = z_pred[:, None, :].expand(-1, k_neg, -1).reshape(bsz * k_neg, dim)
    z_neg_flat = z_neg.reshape(bsz * k_neg, dim)
    neg_energy = head(z_pred_rep, z_neg_flat).view(bsz, k_neg)  # [B, K]

    # Ranking loss: softplus(E_pos - E_neg + margin) averaged over negatives.
    rank_loss = F.softplus(pos_energy[:, None] - neg_energy + margin).mean()

    # Small regularisation on energy magnitudes.
    reg_loss = reg_weight * (pos_energy.square().mean() + neg_energy.square().mean())

    loss = rank_loss + reg_loss

    with torch.no_grad():
        stats = StepStats(
            loss=loss.item(),
            pos_energy=pos_energy.mean().item(),
            neg_energy=neg_energy.mean().item(),
            gap=(neg_energy.mean() - pos_energy.mean()).item(),
            ranking_acc=(pos_energy[:, None] < neg_energy).float().mean().item(),
        )
    return loss, stats


# ── Validation ───────────────────────────────────────────────────────


@torch.no_grad()
def run_validation(
    model: CanonicalJEPA,
    head: GoalEnergyHead,
    loader: DataLoader,
    device: torch.device,
    horizon: int,
    num_negatives: int,
    margin: float,
    reg_weight: float,
    amp_enabled: bool,
    max_batches: int,
) -> StepStats:
    head.eval()
    losses: List[float] = []
    pos_vals: List[float] = []
    neg_vals: List[float] = []
    gaps: List[float] = []
    accs: List[float] = []

    it = iter(loader)
    for _ in range(max_batches):
        try:
            vis, prop, cmds, _dones, _cols = next(it)
        except StopIteration:
            break

        vis = vis.to(device, non_blocking=True).float().div_(255.0)
        prop = prop.to(device, non_blocking=True)
        cmds = cmds.to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            z_pred, z_goal = rollout_terminal_latent(model, vis, prop, cmds, horizon)
            z_neg = sample_negative_goals(z_goal, num_negatives)
            _, stats = energy_ranking_loss(head, z_pred, z_goal, z_neg, margin, reg_weight)

        losses.append(stats.loss)
        pos_vals.append(stats.pos_energy)
        neg_vals.append(stats.neg_energy)
        gaps.append(stats.gap)
        accs.append(stats.ranking_acc)

    if not losses:
        return StepStats(math.nan, math.nan, math.nan, math.nan, math.nan)

    n = len(losses)
    return StepStats(
        loss=sum(losses) / n,
        pos_energy=sum(pos_vals) / n,
        neg_energy=sum(neg_vals) / n,
        gap=sum(gaps) / n,
        ranking_acc=sum(accs) / n,
    )


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GoalEnergyHead on frozen CanonicalJEPA representations.",
    )
    parser.add_argument("--jepa_ckpt", type=str, required=True,
                        help="Path to a CanonicalJEPA checkpoint.")
    parser.add_argument("--data_dir", type=str, default="jepa_final_dataset")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--n_negatives", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default="energy_head_checkpoints")
    parser.add_argument("--log_dir", type=str, default="energy_head_logs")
    # Additional knobs.
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--reg_weight", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--val_batches", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Seed ──────────────────────────────────────────────────────
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    seq_len = args.horizon + 1  # need start frame + H future frames

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ── Load frozen backbone ──────────────────────────────────────
    print(f"Loading CanonicalJEPA backbone from {args.jepa_ckpt} ...")
    sd, meta = load_jepa_checkpoint(args.jepa_ckpt, device)
    model = CanonicalJEPA(latent_dim=args.latent_dim).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("  Backbone loaded and frozen.")

    # ── Energy head + optimiser ───────────────────────────────────
    head = GoalEnergyHead(latent_dim=args.latent_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(device.type, enabled=amp_enabled)

    # ── Datasets (90/10 split via file sharding) ──────────────────
    all_files = sorted(
        f for f in os.listdir(args.data_dir) if f.endswith("_rgb.h5")
    )
    n_total = len(all_files)
    rng = random.Random(args.seed)
    rng.shuffle(all_files)
    n_val = max(1, int(n_total * args.val_frac))
    val_files = all_files[:n_val]
    train_files = all_files[n_val:]

    # Write split file lists to temporary sub-dirs so each
    # StreamingJEPADataset instance only sees its own files.
    train_dir = os.path.join(args.log_dir, "_split_train")
    val_dir = os.path.join(args.log_dir, "_split_val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    def _symlink_files(file_list: List[str], dest_dir: str) -> None:
        for name in file_list:
            src = os.path.join(os.path.abspath(args.data_dir), name)
            dst = os.path.join(dest_dir, name)
            if not os.path.exists(dst):
                os.symlink(src, dst)

    _symlink_files(train_files, train_dir)
    _symlink_files(val_files, val_dir)

    print(f"  Dataset: {n_total} files -> {len(train_files)} train / {len(val_files)} val")

    # ── Latent cache (extracted once, reused every epoch) ─────────
    ckpt_tag = os.path.splitext(os.path.basename(args.jepa_ckpt))[0]
    cache_tag = f"{ckpt_tag}_H{args.horizon}"
    train_cache = os.path.join(args.log_dir, f"latent_cache_{cache_tag}_train.npz")
    val_cache   = os.path.join(args.log_dir, f"latent_cache_{cache_tag}_val.npz")

    def _build_extract_loader(data_dir: str, n_workers: int) -> DataLoader:
        ds = StreamingJEPADataset(
            data_dir=data_dir, seq_len=seq_len, batch_size=256,
            require_no_done=True, require_no_collision=True,
            num_workers=n_workers,
        )
        kw = dict(dataset=ds, batch_size=None, num_workers=n_workers,
                  pin_memory=(device.type == "cuda"))
        if n_workers > 0:
            kw["prefetch_factor"] = 2
        return DataLoader(**kw)

    if not os.path.exists(train_cache):
        print("Building train latent cache (one-time)...")
        loader = _build_extract_loader(train_dir, args.num_workers)
        zp, zg = _extract_latents(loader, model, device, args.horizon, amp_enabled,
                                   desc="  train")
        np.savez(train_cache, z_pred=zp, z_goal=zg)
        print(f"  Saved {zp.shape[0]:,} pairs -> {train_cache}")
    else:
        print(f"  Using cached train latents: {train_cache}")

    if not os.path.exists(val_cache):
        print("Building val latent cache (one-time)...")
        loader = _build_extract_loader(val_dir, max(1, args.num_workers // 2))
        zp, zg = _extract_latents(loader, model, device, args.horizon, amp_enabled,
                                   desc="  val")
        np.savez(val_cache, z_pred=zp, z_goal=zg)
        print(f"  Saved {zp.shape[0]:,} pairs -> {val_cache}")
    else:
        print(f"  Using cached val latents: {val_cache}")

    train_np = np.load(train_cache)
    val_np   = np.load(val_cache)
    z_pred_train = torch.from_numpy(train_np["z_pred"])
    z_goal_train = torch.from_numpy(train_np["z_goal"])
    z_pred_val   = torch.from_numpy(val_np["z_pred"])
    z_goal_val   = torch.from_numpy(val_np["z_goal"])

    train_latent_ds = torch.utils.data.TensorDataset(z_pred_train, z_goal_train)
    train_loader = DataLoader(
        train_latent_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"  Train: {len(z_pred_train):,} pairs | Val: {len(z_pred_val):,} pairs")

    # ── CSV log ───────────────────────────────────────────────────
    csv_path = os.path.join(args.log_dir, "energy_head_metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "epoch", "split", "train_loss", "val_loss",
                "mean_pos_energy", "mean_neg_energy", "gap", "ranking_acc", "lr",
            ])

    # ── Training loop ─────────────────────────────────────────────
    best_val = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        head.train()

        train_losses: List[float] = []
        train_gaps: List[float] = []
        train_accs: List[float] = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (z_pred, z_goal) in enumerate(pbar, start=1):
            z_pred = z_pred.to(device, non_blocking=True)
            z_goal = z_goal.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                z_neg = sample_negative_goals(z_goal, args.n_negatives)
                loss, stats = energy_ranking_loss(
                    head, z_pred, z_goal, z_neg,
                    args.margin, args.reg_weight,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(stats.loss)
            train_gaps.append(stats.gap)
            train_accs.append(stats.ranking_acc)
            global_step += 1
            lr = optimizer.param_groups[0]["lr"]

            # CSV row (train).
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    global_step, epoch, "train", f"{stats.loss:.6f}", "",
                    f"{stats.pos_energy:.6f}", f"{stats.neg_energy:.6f}",
                    f"{stats.gap:.6f}", f"{stats.ranking_acc:.4f}", f"{lr:.2e}",
                ])

            if batch_idx % args.log_every == 0:
                n = min(args.log_every, len(train_losses))
                pbar.set_postfix({
                    "loss": f"{sum(train_losses[-n:]) / n:.4f}",
                    "gap": f"{sum(train_gaps[-n:]) / n:.4f}",
                    "acc": f"{sum(train_accs[-n:]) / n:.3f}",
                    "lr": f"{lr:.2e}",
                })

            if args.save_every > 0 and global_step % args.save_every == 0:
                torch.save(
                    {
                        "energy_head_state_dict": head.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": stats.loss,
                        "jepa_ckpt_path": args.jepa_ckpt,
                        "config": vars(args),
                    },
                    os.path.join(args.out_dir, f"energy_head_step_{global_step}.pt"),
                )

        scheduler.step()

        # ── Validation (on cached latents) ────────────────────────
        val_stats = run_validation_cached(
            head=head,
            z_pred_val=z_pred_val,
            z_goal_val=z_goal_val,
            device=device,
            batch_size=args.batch_size,
            num_negatives=args.n_negatives,
            margin=args.margin,
            reg_weight=args.reg_weight,
            max_batches=args.val_batches,
        )

        # CSV row (val).
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                global_step, epoch, "val", "", f"{val_stats.loss:.6f}",
                f"{val_stats.pos_energy:.6f}", f"{val_stats.neg_energy:.6f}",
                f"{val_stats.gap:.6f}", f"{val_stats.ranking_acc:.4f}",
                f"{optimizer.param_groups[0]['lr']:.2e}",
            ])

        mean_train_loss = sum(train_losses) / max(1, len(train_losses))
        print(
            f"\n  Epoch {epoch} | "
            f"train_loss={mean_train_loss:.4f} | "
            f"val_loss={val_stats.loss:.4f} | "
            f"val_gap={val_stats.gap:.4f} | "
            f"val_acc={val_stats.ranking_acc:.3f}"
        )

        # Save epoch checkpoint.
        ckpt = {
            "energy_head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_stats.loss,
            "val_gap": val_stats.gap,
            "val_ranking_acc": val_stats.ranking_acc,
            "jepa_ckpt_path": args.jepa_ckpt,
            "config": vars(args),
        }
        torch.save(ckpt, os.path.join(args.out_dir, f"energy_head_epoch_{epoch}.pt"))
        torch.save(ckpt, os.path.join(args.out_dir, "energy_head_last.pt"))

        if math.isfinite(val_stats.loss) and val_stats.loss < best_val:
            best_val = val_stats.loss
            torch.save(ckpt, os.path.join(args.out_dir, "energy_head_best.pt"))
            print(f"  New best checkpoint saved (val_loss={best_val:.4f})")

    print("\nEnergy head training complete.")
    print(f"  Best checkpoint : {os.path.join(args.out_dir, 'energy_head_best.pt')}")
    print(f"  Last checkpoint : {os.path.join(args.out_dir, 'energy_head_last.pt')}")
    print(f"  CSV log         : {csv_path}")


if __name__ == "__main__":
    main()
