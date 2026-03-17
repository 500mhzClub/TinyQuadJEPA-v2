#!/usr/bin/env python3
"""Canonical JEPA training loop with EMA target encoder.

Replaces VICReg loss from v1 with student-teacher MSE, using an exponential
moving-average target encoder to prevent representation collapse.

Usage:
    python scripts/3_train_jepa.py --data_dir jepa_final_dataset
    python scripts/3_train_jepa.py --resume_from jepa_checkpoints/step_3000.pt
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from tqjepa.models import CanonicalJEPA
from tqjepa.data import StreamingJEPADataset
from tqjepa.checkpoint_utils import clean_state_dict

torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CanonicalJEPA (EMA student-teacher)")
    p.add_argument("--data_dir", type=str, default="jepa_final_dataset")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--ema_tau_start", type=float, default=0.996)
    p.add_argument("--ema_tau_end", type=float, default=0.999)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="jepa_checkpoints")
    p.add_argument("--log_dir", type=str, default="jepa_logs")
    return p.parse_args()


# ---------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    ema_tau: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "ema_tau": ema_tau,
        },
        path,
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Initializing JEPA training on {device}")

    # ---- Dataset / DataLoader ----------------------------------------
    dataset = StreamingJEPADataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        require_no_done=False,
        require_no_collision=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=2,
    )

    # ---- Model -------------------------------------------------------
    model = CanonicalJEPA(latent_dim=256).to(device)

    start_epoch = 0
    global_step = 0

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        cleaned_sd = clean_state_dict(ckpt["model_state_dict"])
        model.load_state_dict(cleaned_sd)
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)

    model = torch.compile(model)

    # ---- Optimizer (online_encoder + predictor only) -----------------
    trainable_params = (
        list(model.online_encoder.parameters())
        + list(model.predictor.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.resume_from and os.path.exists(args.resume_from):
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print(f"Restored optimizer & scheduler state. Resuming at epoch {start_epoch}, step {global_step}.")
        except Exception as e:
            print(f"Warning: could not restore optimizer/scheduler state: {e}")

    scaler = GradScaler("cuda")

    # ---- Logging setup -----------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    csv_path = os.path.join(args.log_dir, "training_metrics.csv")
    write_header = not os.path.exists(csv_path)
    if write_header:
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "mse_loss", "ema_tau", "lr", "z_target_std"])

    # ---- Training loop -----------------------------------------------
    num_steps_per_timestep = args.seq_len - 1

    for epoch in range(start_epoch, args.epochs):
        model.train()

        # EMA tau: linear interpolation from tau_start to tau_end over epochs
        if args.epochs > 1:
            ema_tau = args.ema_tau_start + (args.ema_tau_end - args.ema_tau_start) * (
                epoch / (args.epochs - 1)
            )
        else:
            ema_tau = args.ema_tau_end
        model.ema_tau = ema_tau

        epoch_loss_sum = 0.0
        epoch_batches = 0
        t_epoch_start = time.time()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            vision, proprio, cmds, dones, collisions = batch

            # Move to device; vision: uint8 -> float [0,1]
            vision = vision.to(device, non_blocking=True).float().div_(255.0)
            proprio = proprio.to(device, non_blocking=True)
            cmds = cmds.to(device, non_blocking=True)
            dones = dones.to(device, non_blocking=True)
            collisions = collisions.to(device, non_blocking=True)

            B = vision.shape[0]
            h_t = torch.zeros(B, model.latent_dim, device=device)

            optimizer.zero_grad(set_to_none=True)

            total_loss = torch.tensor(0.0, device=device)
            valid_steps = torch.tensor(0.0, device=device)
            z_target_for_monitoring = []

            with autocast("cuda", dtype=torch.bfloat16):
                for t in range(num_steps_per_timestep):
                    loss_t, h_t, z_pred, z_target = model.forward_step(
                        vision[:, t],
                        proprio[:, t],
                        cmds[:, t],
                        vision[:, t + 1],
                        proprio[:, t + 1],
                        h_t,
                    )
                    # loss_t shape: (B,) -- per-sample MSE

                    # Mask out frames where the next step has a done or collision
                    mask = ~(dones[:, t + 1] | collisions[:, t + 1])  # (B,) bool
                    n_valid = mask.float().sum()

                    if n_valid > 0:
                        masked_loss = (loss_t * mask.float()).sum() / n_valid
                        total_loss = total_loss + masked_loss
                        valid_steps = valid_steps + 1.0

                    # Detach hidden state to avoid backprop through the full
                    # sequence graph (truncated BPTT per timestep)
                    h_t = h_t.detach()

                    # Collect z_target for collapse monitoring
                    z_target_for_monitoring.append(z_target.detach())

                # Average over valid timesteps
                if valid_steps > 0:
                    seq_loss = total_loss / valid_steps
                else:
                    seq_loss = total_loss

            # Backward pass
            scaler.scale(seq_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # EMA update AFTER optimizer step
            model.update_target_encoder()

            # ---- Collapse monitoring ---------------------------------
            z_target_cat = torch.cat(z_target_for_monitoring, dim=0)  # (B*T', latent_dim)
            z_target_std = z_target_cat.float().std(dim=0).mean().item()

            if z_target_std < 0.1:
                print(f"\n  WARNING: z_target_std={z_target_std:.4f} < 0.1 -- possible representation collapse!")

            # ---- Logging ---------------------------------------------
            global_step += 1
            current_lr = scheduler.get_last_lr()[0]
            loss_val = seq_loss.item()
            epoch_loss_sum += loss_val
            epoch_batches += 1

            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    global_step,
                    epoch + 1,
                    f"{loss_val:.6f}",
                    f"{ema_tau:.6f}",
                    f"{current_lr:.2e}",
                    f"{z_target_std:.6f}",
                ])

            if global_step % 5 == 0:
                pbar.set_postfix({
                    "loss": f"{loss_val:.4f}",
                    "z_std": f"{z_target_std:.3f}",
                    "tau": f"{ema_tau:.4f}",
                    "lr": f"{current_lr:.1e}",
                })

            # ---- Intra-epoch checkpoint ------------------------------
            if global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.out_dir, f"step_{global_step}.pt")
                save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, global_step, ema_tau)
                print(f"\n  Checkpoint saved: {ckpt_path}")

        # ---- End of epoch --------------------------------------------
        scheduler.step()

        avg_epoch_loss = epoch_loss_sum / max(1, epoch_batches)
        elapsed = time.time() - t_epoch_start
        print(
            f"Epoch {epoch + 1} complete | "
            f"avg_loss={avg_epoch_loss:.4f} | "
            f"ema_tau={ema_tau:.4f} | "
            f"time={elapsed:.0f}s"
        )

        epoch_ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch + 1}.pt")
        save_checkpoint(epoch_ckpt_path, model, optimizer, scheduler, epoch, global_step, ema_tau)
        print(f"  Epoch checkpoint saved: {epoch_ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
