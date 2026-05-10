# Fine-tune X-VLA on a lerobot dataset

This is the recipe that produced LOBE's 84%+ X-VLA reproduction on LIBERO. It
generalizes to any single-arm `observation.state`/`action`-style lerobot dataset,
with LIBERO as the reference.

## What you get

The pipeline:

1. Starts from `2toINF/X-VLA-Pt` (0.9 B params, pretrained on a mixed multi-embodiment corpus).
2. Fine-tunes with `xvla-adamw`, effective batch 128, 60k steps, constant LR 1e-4, flow matching on 30-step action chunks.
3. Runs through a `lobe/patches.py` wrapper around `lerobot.policies.factory.make_policy` that prevents lerobot's env-inferred `output_features["action"]` override from silently truncating a trained 10-D head to 7-D at eval time. Without this patch X-VLA in `action_mode=auto` eval-crashes silently to 0% success with low train loss — this is the root cause of most failed lerobot X-VLA reproductions.
4. Runs through a second patch on `make_env_pre_post_processors` that strips the redundant `XVLAImageNetNormalizeProcessorStep` from env_preprocessor, because the policy preprocessor inherited from `xvla-pt` already normalizes and double-normalization silently errors out on images outside [0, 1].

Both patches are applied automatically when anything imports `lobe`.

## Required dataset format

To fine-tune X-VLA with `action_mode=auto` on a single-arm robot, your lerobot v3.0 dataset must have:

**Observation state** — 20-D in the "grip-site EE6D" frame that X-VLA expects at eval time:

| index | meaning |
|---|---|
| 0-2 | `eef_xyz` (world frame, meters) |
| 3-8 | `rot6d_site` = first two columns of the grip-site rotation matrix, flattened |
| 9 | `extra` (0) |
| 10-19 | zeros — right-arm pad for the 20-D dual-arm canonical format |

**Action** — 10-D `abs_action_6d` in the same grip-site frame, single chunk step:

| index | meaning |
|---|---|
| 0-2 | `abs_xyz` |
| 3-8 | `abs_rot6d` (first two columns) |
| 9 | `gripper` ∈ {-1, +1} |

**Images** — 256×256 RGB, usually two views (agent + wrist), stored as PNG or JPEG bytes in parquet.

### Getting `abs_action_6d` and grip-site state right

Two subtleties that cost a lot of debug time in v0.x:

1. **Body frame vs grip-site frame.** `robot0_eef_quat` from robosuite is in the `hand_body` frame. `controller.ee_ori_mat` and `LiberoProcessorStep` at eval time read from the `grip_site` frame. These differ by a constant `R_z(-90°)` defined in the gripper XML. If your dataset stores rotation derived from `eef_quat` (like raw HuggingFaceVLA/libero does), you must rotate to grip-site before training or your eval will be 0% despite perfect train loss. The rotation reduces to a column swap: `rot6d_site = [-body_col1; +body_col0]`. See `scripts/build_libero_xvla_dataset.py:build_state_20d()`.

2. **Absolute action targets must come from `controller.goal_pos`/`goal_ori`, not from the robot's observed eef trajectory.** For LIBERO, the upstream X-VLA team computed these via sim replay and published the result as `2toINF/Libero-XVLA-format` on HuggingFace. For a new embodiment, either replay your demos and capture the controller's goal state at each step, or record the goal state during teleoperation.

### LIBERO reference converter

```bash
# 1. Download the precomputed upstream dataset (~31 GB)
.venv/bin/python -c "
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ThreadPoolExecutor
api = HfApi()
files = [f for f in api.list_repo_files('2toINF/Libero-XVLA-format', repo_type='dataset')
         if f.startswith(('libero_spatial/', 'libero_object/', 'libero_goal/', 'libero_10/'))]
with ThreadPoolExecutor(16) as ex:
    list(ex.map(lambda f: hf_hub_download('2toINF/Libero-XVLA-format', f, repo_type='dataset',
                local_dir='/mnt/localssd/sunlingfeng/datasets/libero_xvla_format'), files))
"

# 2. Convert to lerobot parquet (1692 episodes, ~2 hours with image stats computation)
.venv/bin/python scripts/build_libero_xvla_dataset.py
```

## Pretrained starting weights

X-VLA-Pt lives on the Hub as `2toINF/X-VLA-Pt`. lerobot's factory expects the
config to carry `action_mode`, `max_state_dim`, `max_action_dim`, and a normalized
feature schema; `2toINF/X-VLA-Pt` ships with values tuned for their default `ee6d`
mode, which doesn't match our 10-D single-arm setup. LOBE keeps a pre-adjusted
version at `/mnt/localssd/sunlingfeng/checkpoints/xvla-pt-v8` with:

- `output_features.action.shape = [10]`
- `action_mode = "auto"`
- `max_state_dim = 20`, `max_action_dim = 20`
- `normalization_mapping.ACTION = "IDENTITY"` (matches our upstream-precomputed targets)
- `policy_preprocessor.json` has state shape `[20]`

You can regenerate it from `2toINF/X-VLA-Pt` by applying those edits. For LOBE's
own experiments, just use the existing `xvla-pt-v8` directory.

## Recommended recipe (v1.2 — 90.5% LIBERO avg)

The best result LOBE produced on LIBERO uses **two-stage training**:

1. **Stage 1** (60k steps, ~3h40m on 8×H100): broad pretrain on `local/libero_all_v15`, which is the full `2toINF/Libero-XVLA-format` dataset (5,525 episodes across all 5 suites including libero_90). Uses the V14 recipe below. Output: `xvla-libero-v15`.
2. **Stage 2** (30k steps, ~1h50m on 8×H100): continue from `xvla-libero-v15/checkpoints/060000/pretrained_model` on the narrower `local/libero_xvla_v12` dataset (1,692 episodes, only the 4 eval fine-tune suites). Output: `xvla-libero-v16`.

Stage 1 learns long-horizon skills from libero_90 (libero_10 score goes 69 → 86). Stage 2 re-aligns the goal-conditioned policy on the 4 eval suites (goal score recovers 81 → 91), while libero_10 retains most of stage 1's gains because the model has already internalized the long-horizon motor skills. See [Benchmarks](../benchmarks.md) for V10→V16 progression and per-suite deltas.

Stage 2 uses the same launch command as stage 1 except for `--policy.path` (points at the stage 1 60k checkpoint), `--steps=30000`, `--policy.scheduler_decay_steps=30000`, `--policy.scheduler_warmup_steps=500` (shorter warmup for a warm start from the already-trained model), and a different `--output_dir`.

## Stage 1 launch command (same as v1.0 single-stage recipe — 85.75% if used alone)

```bash
MUJOCO_GL=osmesa .venv/bin/accelerate-launch --num_processes=8 --mixed_precision=bf16 \
  scripts/_lobe_train_entry.py \
  --dataset.repo_id=local/libero_xvla_v12 \
  --dataset.root=/mnt/localssd/sunlingfeng/datasets/local/libero_xvla_v12 \
  --dataset.image_transforms.enable=true \
  --policy.path=/mnt/localssd/sunlingfeng/checkpoints/xvla-pt-v8 \
  --policy.action_mode=auto \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.dtype=bfloat16 \
  --policy.use_amp=false \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_weight_decay=0.01 \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=60000 \
  --policy.scheduler_decay_lr=1e-4 \
  --batch_size=16 \
  --num_workers=4 \
  --steps=60000 \
  --save_freq=10000 \
  --log_freq=100 \
  --eval_freq=60000 \
  --output_dir=checkpoints/xvla-libero-v14 \
  --job_name=xvla-libero-v14
```

Key points:

- **`scripts/_lobe_train_entry.py`** instead of `.venv/bin/lerobot-train` — this imports `lobe` first, which applies the two load-bearing patches before `lerobot_train.main()` runs.
- **`--policy.*` (not `--optimizer.*` / `--scheduler.*`) for optimizer and scheduler flags**. This is non-obvious but load-bearing. `TrainPipelineConfig.validate()` in `lerobot/configs/train.py:135` calls `self.optimizer = self.policy.get_optimizer_preset()` and `self.scheduler = self.policy.get_scheduler_preset()` when `use_policy_training_preset=True` (the default), which **overwrites any top-level `--optimizer.*` / `--scheduler.*` CLI flags you passed**. The presets read from `XVLAConfig.optimizer_*` and `XVLAConfig.scheduler_*` attributes, so you have to set them via `--policy.optimizer_*` and `--policy.scheduler_*`. Pre-v1.0 runs silently trained with the lerobot defaults (`scheduler_decay_steps=30000, scheduler_decay_lr=2.5e-6`) regardless of what the launch script said.
- **`decay_lr == peak_lr == 1e-4`** makes `cosine_decay_with_warmup`'s decay multiplier a constant 1.0, so after the 1k-step warmup the LR stays flat. This matches the X-VLA paper's "constant LR after warmup" default (their `--use_cosine_decay` flag is off). With the lerobot default cosine decay to `decay_lr=2.5e-6`, the LR is 40× lower by step 30k and the second half of training essentially stops learning.
- **`batch_size=16 × 8 GPUs = 128 effective`** matches the paper.
- **`action_mode=auto`** — correct for single-arm LIBERO's native 10-D action. The 20-D `ee6d` mode is for dual-arm or for embodiment-mixed pretraining; zero-padding single-arm to 20-D to use it would just add noise.
- **`optimizer_betas` can't be set from CLI** due to a draccus tuple-parsing issue; the v1.0 recipe uses lerobot default `(0.9, 0.99)` instead of paper `(0.9, 0.95)`. Small difference, not a priority. If you need it, edit `XVLAConfig.optimizer_betas` in the checkpoint's `config.json` before launching.

## Inference knobs

Two eval-time policy config values move accuracy without retraining. Both can be set in the checkpoint's `config.json` before loading (or via `--policy.*` at `lobe-eval` time if the CLI parsing accepts the field). Measured on V16 checkpoint against 4-suite LIBERO eval:

| Knob | Default | Alternative | Effect vs default |
|---|---|---|---|
| `num_denoising_steps` | 10 | **20** | **+0.75 avg, +7 on libero_10, −2 on spatial/goal**. More ODE refinement per chunk prediction reduces per-step action error, which matters most for long-horizon multi-step sequences where errors compound. This is our current best (V17b, 91.25% avg). |
| `n_action_steps` | 30 | 10 | −0.75 avg, +4 on goal, −5 on libero_10. Shorter open-loop execution re-predicts more often, helping precise target matching (goal) but introducing incoherence on long multi-step sequences (libero_10). Net loss. |

**Rule of thumb**: bump `num_denoising_steps` to 20 if inference latency is not a concern (2× forward passes per chunk). Leave `n_action_steps` at chunk_size=30 unless your task is mostly single-goal precision work.

## Evaluation

```bash
MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES=0 .venv/bin/lobe-eval \
  --policy.path=checkpoints/xvla-libero-v14/checkpoints/060000/pretrained_model \
  --env.type=libero --env.task=libero_object \
  --env.control_mode=absolute \
  --eval.n_episodes=10 --eval.batch_size=1
```

Key points:

- **`lobe-eval`** (not `lerobot-eval`) so the lobe patches fire. With raw `lerobot-eval` you silently get 0% success.
- **`--env.control_mode=absolute`** — X-VLA emits absolute targets, not deltas. The env must set `controller.use_delta=False`.
- **GPUs 0–3 in parallel** for the four LIBERO suites. One suite at `n_episodes=10` takes ~25–60 min depending on the episode length cap (280 for spatial/object, 300 for goal, 520 for libero_10).

## Adapting to a new dataset

The pipeline is dataset-agnostic at the lerobot-train level. To fine-tune X-VLA on
your own lerobot-format single-arm dataset:

1. **Write a converter** to produce 20-D grip-site state and 10-D absolute EE6D
   action, following the two subtleties above. `scripts/build_libero_xvla_dataset.py`
   is the reference; `build_state_20d()` and the state_20d/abs_action_6d column
   write are the parts you'll copy.
2. **Swap dataset paths** in the launch command (`--dataset.repo_id`, `--dataset.root`).
3. **If your camera layout differs**, pass `--rename_map='{"observation.images.yourcam": "observation.images.image"}'` so the policy sees the expected keys.
4. **If your robot base frame differs from LIBERO's Panda**, the grip-site vs
   hand-body offset may not be exactly `R_z(-90°)` — measure it empirically with
   the method in `/tmp/compute_body_site_offset.py` (kept in git history) and
   update the column permutation.

The `lobe/patches.py` patches are generic to any `XVLAConfig` fine-tune — they
don't assume anything LIBERO-specific.
