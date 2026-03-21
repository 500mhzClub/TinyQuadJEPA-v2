# TinyQuadJEPA v2

Canonical JEPA latent world model for quadruped navigation with obstacle awareness.

Successor to [TinyQuadJEPA v1](../TinyQuadJEPA/), which demonstrated that VICReg-trained latent dynamics work for open-floor navigation but fail near obstacles the model has never seen. This project fixes both problems: the architecture is now a proper student-teacher JEPA, and the training data includes randomly placed obstacles with diverse textures.

## What changed from v1

| | v1 | v2 |
|---|---|---|
| **Architecture** | Single encoder + VICReg (sim+var+cov) | Student-teacher with EMA target encoder + MSE |
| **Training data** | Flat plane, single checkerboard texture | Boxes + wall structures, 27 ground textures, random obstacle colors |
| **Collapse prevention** | VICReg variance/covariance terms | EMA asymmetry (proven by BYOL, DINO, I-JEPA) |
| **Collision handling** | None (never saw obstacles) | AABB detection per step, reset + mask in training |
| **Code structure** | Model classes duplicated across 8 files | Shared `tqjepa/` Python package |

## Architecture

### System 1: Blind Walking Policy (PPO)

Unchanged from v1. A PPO-trained gait controller that turns body-frame velocity commands `(vx, vy, wz)` into 12 joint position targets. Trained on flat ground with domain-randomized friction, mass, and motor latency.

### System 2: Canonical JEPA World Model

```
Training:
  z_t      = online_encoder(vis_t, prop_t)          # gradients flow
  z_pred   = predictor(z_t, cmd_t, h_t)             # gradients flow
  z_target = target_encoder(vis_{t+1}, prop_{t+1})   # NO gradients (EMA copy)
  loss     = MSE(z_pred, z_target.detach())

After each optimizer step:
  target_params = tau * target_params + (1 - tau) * online_params

Inference (MPC / energy head):
  z_start = online_encoder(current_obs)    # predictor expects online space
  z_goal  = target_encoder(goal_obs)       # energy head expects target space
  z_roll  = predictor.rollout(z_start, cmds)  # output is in target space
  energy  = head(z_roll, z_goal)           # both in target space
```

**Online encoder** (student): `VisionEncoder(3→128) + ProprioEncoder(47→128) → JointEncoder(256→256)` with LayerNorm. Gets gradient updates.

**Target encoder** (teacher): Deep copy of the online encoder. Updated via EMA (`tau` anneals from 0.996 to 0.999). No gradients.

**Predictor**: Action-conditioned GRUCell. Input: `z_t(256) + cmd(3)` → projected → GRU → output projection → `z_pred(256)`.

**Energy head**: Post-hoc trained compatibility scorer. Input: `[z_pred, z_goal, z_pred-z_goal, z_pred*z_goal]` → 1024 → 512 → 1. Lower energy = more compatible.

### Training data with obstacles

Physics rollout now mixes free-standing boxes with long wall segments, corridors, L-shapes, dead ends, and optional perimeter walls. Each chunk gets a different layout; all envs within a chunk share the layout (Genesis constraint). Collision detection resets the robot when it clips into obstacles, and collision frames are masked during JEPA training.

Visual rendering applies a larger procedural texture bank spanning checkerboards, stripes, gradients, fractal noise, tile, concrete, wood, carpet, grass, gravel, and sand. Workers keep several textured scene variants alive so texture choice changes across environments, then add per-frame domain randomization (brightness, contrast, noise, hue shift).

## Runbook

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- Genesis simulator
- `pip install h5py imageio pillow matplotlib pandas`

### 0. Verify the robot loads

```bash
cd TinyQuadJEPA-v2
python sim/create_assets.py
```

### 1. Train the PPO walking policy (or reuse from v1)

```bash
python sim/train_blind.py
```

Or copy a trained checkpoint from v1.

### 2. Collect physics rollouts with obstacles

```bash
python scripts/1_physics_rollout.py \
  --ckpt <ppo_checkpoint> \
  --chunks 5 \
  --n_envs 2048 \
  --steps 1000
```

Output: `jepa_raw_data/chunk_*.npz` (each with a different obstacle layout).

### 3. Render egocentric vision

```bash
python scripts/2_visual_renderer.py \
  --raw_dir jepa_raw_data \
  --out_dir jepa_final_dataset \
  --workers 4
```

Output: `jepa_final_dataset/*_rgb.h5`

### 4. Spot-check the dataset

```bash
python tools/verify_dataset.py --data_dir jepa_final_dataset
```

Output: `verification_videos/clip_*.gif`

### 5. Train the JEPA backbone

```bash
python scripts/3_train_jepa.py \
  --data_dir jepa_final_dataset \
  --device cuda \
  --epochs 20
```

Resume from checkpoint:

```bash
python scripts/3_train_jepa.py \
  --data_dir jepa_final_dataset \
  --device cuda \
  --resume_from jepa_checkpoints/jepa_epoch_10_step_5000.pt
```

Monitor collapse: watch `z_target_std` in the CSV. If it drops below 0.1, representations are collapsing.

```bash
python tools/plot_metrics.py --csv jepa_logs/training_metrics.csv
```

### 6. Train the energy head

```bash
python scripts/4_train_energy_head.py \
  --jepa_ckpt jepa_checkpoints/jepa_epoch_20.pt \
  --data_dir jepa_final_dataset \
  --device cuda
```

### 7. Run the navigation demo

```bash
python scripts/5_genesis_eval.py \
  --jepa_ckpt jepa_checkpoints/jepa_epoch_20.pt \
  --head_ckpt energy_head_checkpoints/energy_head_best.pt \
  --ppo_ckpt <ppo_checkpoint> \
  --with_obstacles
```

### 8. Run the exploration demo

```bash
python scripts/6_explore_demo.py \
  --jepa_ckpt jepa_checkpoints/jepa_epoch_20.pt \
  --ppo_ckpt <ppo_checkpoint>
```

## Project structure

```
tqjepa/                        Importable Python package
  models/
    encoders.py                VisionEncoder, ProprioEncoder, JointEncoder
    predictor.py               LatentPredictor (GRU action-conditioned)
    jepa.py                    CanonicalJEPA (online + target + predictor + EMA)
    energy_head.py             GoalEnergyHead
    ppo.py                     ActorCritic
  math_utils.py                Quaternion ops, frame transforms
  genesis_utils.py             Backend selection, scene helpers
  texture_utils.py             Procedural texture generation
  obstacle_utils.py            Random obstacle layouts, collision detection
  checkpoint_utils.py          Load / save helpers
  data/
    streaming_dataset.py       StreamingJEPADataset

scripts/
  1_physics_rollout.py         Data collection with obstacles + collision tracking
  2_visual_renderer.py         RGB rendering with texture randomization
  3_train_jepa.py              Canonical JEPA training (EMA + MSE)
  4_train_energy_head.py       Energy head training
  5_genesis_eval.py            Closed-loop waypoint navigation demo
  6_explore_demo.py            Sensor-frontier exploration demo

tools/
  verify_dataset.py            GIF spot-check
  plot_metrics.py              Training curve plotter
  visualise_energy_landscape.py  2D energy heatmap
  visualise_feature_maps.py    CNN activation inspector

sim/
  actuator.py                  STS3215 servo simulator
  create_assets.py             URDF generation
  train_blind.py               PPO locomotion training

assets/mini_pupper/            Robot URDF + meshes
hardware/README.md             Hardware build guide
```

## Design decisions

**Why EMA instead of VICReg?** The student-teacher asymmetry prevents collapse without explicit variance/covariance regularization. This is simpler (one loss term instead of three), more stable (no weight-ratio sensitivity), and better validated (BYOL, DINO, I-JEPA all use this pattern).

**Why not retrain PPO with obstacles?** The PPO handles locomotion; obstacle avoidance is the world model's job. If the planner works correctly, the robot never touches obstacles. PPO retraining is a Phase 2 stretch goal if needed.

**Collision handling strategy:** AABB detection with 0.15m margin resets the robot before the camera clips inside obstacles. Collision frames in the dataset are masked during training so the predictor never learns from physically impossible transitions.

**Texture randomization rationale:** The v1 model overfits to a single checkerboard. A broader texture bank with per-environment variation forces the visual encoder to learn structural features (edges, depth cues) rather than texture-specific patterns.

## Collapse monitoring

Without VICReg's variance term, monitor `z_target_std` (logged per step in the training CSV). This is the mean per-dimension standard deviation of the target encoder's outputs across the batch. Healthy training keeps this above 0.2. Below 0.1 is a collapse warning.

Mitigations if collapse occurs:
1. Lower `ema_tau_start` (e.g. 0.990) — the target moves faster, reducing the asymmetry trap
2. Add a small variance penalty as a safety net
3. Increase predictor capacity
