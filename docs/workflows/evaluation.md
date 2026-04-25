# Evaluation

## Standard eval command

```bash
MUJOCO_GL=egl lobe-eval \
  --policy.path=<checkpoint_dir> \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10
```

The checkpoint dir should be `<output>/checkpoints/<step>/pretrained_model/` (the `pretrained_model` subdirectory).

## LIBERO suites

The SmolVLA paper averages four suites for the headline number:

| Suite key | Description | Difficulty |
|---|---|---|
| `libero_spatial` | Same objects, different layouts | Easy |
| `libero_object` | Different objects, same layout | Easy-Med |
| `libero_goal` | Different goals | Med |
| `libero_10` | Long-horizon (called "Long" in the paper) | Hard |
| `libero_90` | 90 small tasks | Mixed |

Always use `libero_spatial,libero_object,libero_goal,libero_10` for paper-comparable numbers.

## Per-policy eval flags

| Policy | Required flags |
|---|---|
| SmolVLA | `--policy.n_action_steps=10` and `--rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'` |
| Diffusion | nothing extra |
| Flow Matching (ours) | nothing extra |
| pi0 | `--policy.n_action_steps=10` |

## Rendering backend

LIBERO uses MuJoCo for rendering. On a headless server you have two options:

- **EGL** (`MUJOCO_GL=egl`): GPU-accelerated, ~21 it/s, ~2h for 400 episodes. Requires `libnvidia-gl-580-server` matching your driver version.
- **OSMesa** (`MUJOCO_GL=osmesa`): CPU software rendering, ~7 it/s, ~6h for 400 episodes. Always works, slower.

Prefer EGL when available.

## Reading results

The eval log prints per-task success rates as it goes:

```
Stepping through eval batches: 100%|██████████| 10/10 [09:22<00:00, 56.21s/it, running_success_rate=90.0%]
```

The aggregated number is logged at the end:

```
Aggregated Metrics for overall:
{'avg_sum_reward': 0.82, 'pc_success': 82.0, 'n_episodes': 400, ...}
```

`pc_success` is the percentage of successful episodes across all 40 tasks.

## Eval the official checkpoint

You can compare against the published SmolVLA checkpoint directly:

```bash
MUJOCO_GL=egl lobe-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 --eval.n_episodes=10 \
  --policy.n_action_steps=10
```

In our setup, the official checkpoint scores 41–62.8% (vs. paper's 87.3%) — a known reproduction gap reported by many community members.

---

## YAM (real robot, no sim)

YAM has no simulator. Evaluation uses replay-based MSE on held-out episodes.

### Replay MSE

```bash
# DP / FM (camera names match dataset directly):
uv run python scripts/eval_replay.py \
    --policy.path=checkpoints/yam-grey-cube-fm-v1/checkpoints/050000/pretrained_model \
    --dataset.repo_id=ttotmoon/yam_pick_up_grey_cube \
    --eval_episodes 8 9

# X-VLA (needs camera rename):
uv run python scripts/eval_replay.py \
    --policy.path=checkpoints/yam-grey-cube-xvla-v0/checkpoints/020000/pretrained_model \
    --dataset.repo_id=ttotmoon/yam_pick_up_grey_cube \
    --eval_episodes 8 9 \
    --rename_map='{"observation.images.head_camera": "observation.images.image", "observation.images.left_wrist_camera": "observation.images.image2", "observation.images.right_wrist_camera": "observation.images.image3"}'

# SmolVLA (camera1/2/3):
uv run python scripts/eval_replay.py \
    --policy.path=checkpoints/yam-grey-cube-smolvla-v0/checkpoints/020000/pretrained_model \
    --dataset.repo_id=ttotmoon/yam_pick_up_grey_cube \
    --eval_episodes 8 9 \
    --rename_map='{"observation.images.head_camera": "observation.images.camera1", "observation.images.left_wrist_camera": "observation.images.camera2", "observation.images.right_wrist_camera": "observation.images.camera3"}'
```

The script bulk-decodes episode videos (O(n)), runs `policy.select_action`
frame-by-frame, and compares predicted actions to ground-truth. Reports
per-joint MSE, L∞, and aggregate. Takes ~7 min for 2 episodes.

### Serving test

Verifies end-to-end: starts `lobe-serve` with each checkpoint, connects a
WebSocket client, sends synthetic observations, confirms action shape is
`(horizon, 14)`.

```bash
uv run python scripts/test_serve_all.py
```

### Inference benchmark

Raw forward-pass latency (no deployment patterns). Proper CUDA sync + warmup.

```bash
uv run python scripts/bench_inference.py --both     # compiled vs uncompiled
```
