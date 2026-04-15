# LOBE Benchmarks

Standardized evaluation protocol for all policies. When adding a new method, check this doc to know exactly what dataset to train on, what suites to eval on, and what numbers to compare against.

---

## Environments

### PushT (2D, quick sanity check)
- **Task**: Push a T-shaped block to a target pose
- **Metric**: IoU (Intersection over Union) with target
- **Dataset**: `lerobot/pusht` (25k frames)
- **Obs**: 96×96 top-down image + agent position
- **Action**: 2D (dx, dy)
- **Use for**: Quick iteration, sanity-checking new policy implementations (<10 min to train)

### LIBERO (7-DOF manipulation, primary benchmark)
- **Task**: 130 tasks across 5 suites, 7-DOF Franka robot
- **Dataset**: `HuggingFaceVLA/libero` (273k frames, 1693 episodes, 2 cameras at 256×256)
- **Obs**: 2 camera images (agentview + eye-in-hand) + robot state (8-dim: eef_pos + axis_angle + gripper)
- **Action**: 7-dim (6 DOF + gripper)
- **Rendering**: `MUJOCO_GL=egl` (GPU-accelerated, 21 it/s) preferred. Fallback: `MUJOCO_GL=osmesa` (CPU, 7 it/s, 3× slower)
- **Use for**: Primary benchmark. All methods must report numbers here.

#### LIBERO Eval Suites

**IMPORTANT**: The published SmolVLA/pi0 numbers (87.3%, 82.5%, etc.) are averages over 4 suites: spatial + object + goal + long. NOT libero_10.

| Suite | Tasks | Description | Difficulty |
|-------|-------|-------------|------------|
| `libero_spatial` | 10 | Same objects, different spatial arrangements | Easy |
| `libero_object` | 10 | Different objects, same spatial layout | Easy-Medium |
| `libero_goal` | 10 | Same objects/layout, different goals | Medium |
| `libero_10` | 10 | Long-horizon multi-step tasks (called "Long" in SmolVLA paper) | Hard |
| `libero_90` | 90 | Large-scale suite (90 tasks) | Mixed |

**Standard eval (for paper comparison)**: `libero_spatial,libero_object,libero_goal,libero_10`
Note: SmolVLA paper calls `libero_10` "Long-horizon". There is no separate `libero_long` suite in the codebase.

#### LIBERO Eval Command Template
```bash
MUJOCO_GL=osmesa lerobot-eval \
  --policy.path=<checkpoint> \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_long \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.n_action_steps=<see per-policy notes>
```

#### Per-Policy Eval Notes
| Policy | n_action_steps | rename_map needed? | Notes |
|--------|---------------|-------------------|-------|
| SmolVLA | 10 | Yes (image→camera1, image2→camera2) | 1 gave worse results (41% vs 54%) |
| Diffusion | default (use checkpoint config) | No | — |
| pi0 | 10 | TBD | Per OpenPI docs |
| FM (ours) | N/A | N/A | Needs custom eval (not lerobot-eval compatible) |

---

## Training Datasets

| Dataset | Repo ID | Frames | Episodes | Cameras | Resolution | Notes |
|---------|---------|--------|----------|---------|------------|-------|
| PushT | `lerobot/pusht` | 25k | — | 1 | 96×96 | 2D top-down |
| LIBERO | `HuggingFaceVLA/libero` | 273k | 1,693 | 2 | 256×256 | Image format (PNG in parquet) |

**Data loading optimization**: We patched `lerobot_dataset.py:_query_hf_dataset` to bypass `set_transform` for non-image columns. Without this, querying 50-step action chunks decodes 100 throwaway PNG images per sample (12× slower).

---

## Results

### LIBERO Standard (spatial + object + goal + long, 40 tasks, 10 episodes each)

| Model | Source | Params | Avg | Spatial | Object | Goal | Long (libero_10) | Train Time | Config |
|-------|--------|--------|-----|---------|--------|------|------------------|------------|--------|
| SmolVLA | published | 450M | **87.3%** | 90 | 96 | 92 | 71 | — | batch=64, 100k, LR=1e-4 |
| SmolVLA | official ckpt, our eval | 450M | **62.8%** | ~68 | ~77 | ~60 | ~30 | — | n_action_steps=10, osmesa |
| **SmolVLA** | **ours (paper config)** | 450M | **82.0%** | ~90 | ~95 | ~90 | ~51 | 4h (8×H100) | batch=64, 100k, LR=1e-4, bf16 |
| **SmolVLA** | **ours (scaled)** | 450M | **80.5%** | ~88 | ~93 | ~88 | ~48 | **1.5h (8×H100)** | batch=256, 25k, LR=4e-4, bf16 |
| Diffusion | published | ~50M | **72.4%** | 78.3 | 92.5 | 68.3 | 50.5 | — | — |
| Diffusion | ours (v3) | ~50M | **40.3%** | ~93 | ~3 | ~3 | ~63 | 50min (8×H100) | batch=256, 25k, bf16 |
| FM | ours | 16M | needs eval | — | — | — | — | 1.9h (1×H100) | batch=64, 50k |
| pi0-FAST | published | 3B | **82.5%** | — | — | — | — | — | batch=32, 20k |
| pi0.5 | published | 3B | **97.5%** | — | — | — | — | — | batch=32×8GPU, 6k |
| X-VLA | published | 0.9B | **98.1%** | — | — | — | — | — | ~30k steps |
| **X-VLA** | **ours v1.0 (V14)** | 0.9B | 85.75% | 86 | 95 | 93 | 69 | 3h40m (8×H100) | batch=128, 60k, constant LR 1e-4, upstream `2toINF/Libero-XVLA-format` data. See `docs/workflows/xvla_finetune.md`. |
| **X-VLA** | **ours v1.1 (V15)** | 0.9B | 87.00% | 88 | 93 | 81 | 86 | 3h40m (8×H100) | V14 + libero_90 aux data (5525 total eps, 3.3× V14). +17 libero_10 from diverse long-horizon scenes, -12 goal from diluted goal-conditioning. Net +1.25 avg. |
| **X-VLA** | **ours v1.2 (V16)** | 0.9B | **90.50%** | **91** | **97** | **91** | 83 | 5h30m cumulative | Two-stage: V15 (60k on libero_all_v15) → V16 (continue 30k on V12 only). Goal recovers from V15's 81 → 91, libero_10 holds at 83 vs V15's 86, spatial/object reach new highs 91/97. Gap to paper: 7.6 pp. |

**Key observations:**
- Our SmolVLA (80.5-82%) significantly beats the official HF checkpoint (62.8%) on our eval
- The ~5% gap to published (87.3%) is likely osmesa rendering + mujoco version (see eval notes)
- Diffusion v3 failed (40.3%) — batch=256 is too large for diffusion on LIBERO. Needs proper hyperparameter search.
- Scaled SmolVLA (1.5h) is nearly as good as paper config (4h) — great efficiency
- X-VLA v1.0 hit 85.75% (vs paper 98.1%) by training on `2toINF/Libero-XVLA-format` (upstream precomputed `abs_action_6d`) with constant LR post-warmup. Biggest remaining gap is on libero_10 (long-horizon) where we're at 69% — likely helped by adding libero_90 auxiliary training data, which is the current v1.1 experiment.

### LIBERO-10 (10 mixed tasks, harder, separate from standard eval)

| Model | Success | Notes |
|-------|---------|-------|
| SmolVLA (ours) | 51% | n_action_steps=1 |
| SmolVLA (ours, scaled) | 54% | n_action_steps=10, batch=256, LR=4e-4 |
| SmolVLA (official HF) | 41-44% | Known community reproduction gap |

### PushT

| Model | IoU / Success | Steps | Notes |
|-------|--------------|-------|-------|
| FM (transformer) | ~40% | 10k | Quick sanity check |
| FM (unet) | ~60% | 10k | Better backbone for PushT |
| Diffusion (published) | ~91% IoU | 50k | Gold standard |

---

## Adding a New Method

When benchmarking a new policy on LIBERO:
1. **Train** on `HuggingFaceVLA/libero` (273k frames)
2. **Eval** on `libero_spatial,libero_object,libero_goal,libero_long` (standard 4 suites)
3. **Report** per-suite breakdown + average (compare to table above)
4. **Log** to `experiments.tsv` with training config details
5. **Optional**: also eval on `libero_10` for extended comparison

Match total training samples (~6.4M) for fair comparison unless the method's paper specifies otherwise.
