# CLI Reference

## `lobe-train`

Wrapper around `lerobot-train`. Identical CLI surface, plus LOBE custom policies are auto-registered.

```bash
lobe-train [OPTIONS]
```

Common options:

| Option | Description |
|---|---|
| `--policy.type=<name>` | Policy type (`diffusion`, `act`, `flow_matching`, ...). Mutually exclusive with `--policy.path`. |
| `--policy.path=<repo_or_dir>` | Pretrained checkpoint to fine-tune |
| `--policy.repo_id=<name>` | Required for HF Hub publishing (any string works for local) |
| `--dataset.repo_id=<id>` | LeRobot-format dataset (HF Hub repo or local path) |
| `--batch_size=N` | Per-process (per-GPU) batch size |
| `--steps=N` | Total training steps |
| `--optimizer.lr=F` | Learning rate (overrides preset) |
| `--num_workers=N` | DataLoader workers (8 is good for LIBERO) |
| `--output_dir=<path>` | Where to save checkpoints |
| `--save_freq=N` | Checkpoint every N steps |
| `--save_checkpoint=true` | Enable checkpointing |
| `--wandb.enable=true` | Enable Weights & Biases logging |
| `--wandb.project=<name>` | W&B project |

Run `lobe-train --help` for the full list (it's long).

### Multi-GPU

```bash
uv run python -m accelerate.commands.launch \
  --num_processes 8 --multi_gpu --mixed_precision bf16 \
  $(which lobe-train) \
  --policy.type=diffusion \
  --dataset.repo_id=... \
  ...
```

## `lobe-eval`

Wrapper around `lerobot-eval`. Loads any pretrained policy and evaluates on a sim env.

```bash
lobe-eval [OPTIONS]
```

| Option | Description |
|---|---|
| `--policy.path=<dir_or_repo>` | Checkpoint dir (with `pretrained_model/`) or HF Hub repo |
| `--env.type=<name>` | Sim env (`libero`, `pusht`, `metaworld`, `aloha`, ...) |
| `--env.task=<task>` | Task or comma-separated list (e.g. `libero_spatial,libero_object,libero_goal,libero_10`) |
| `--eval.batch_size=N` | Parallel envs (use 1 for safest) |
| `--eval.n_episodes=N` | Episodes per task |
| `--policy.n_action_steps=N` | Override action chunk steps (e.g. 10 for SmolVLA) |
| `--rename_map='{...}'` | Remap env observation keys to policy expected keys |

### LIBERO eval template

```bash
MUJOCO_GL=egl lobe-eval \
  --policy.path=<checkpoint>/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.n_action_steps=10
```

## `lobe-serve`

Serve a trained policy over WebSocket.

```bash
lobe-serve [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--checkpoint <path>` | (required) | Checkpoint dir or HF Hub repo |
| `--host <host>` | `0.0.0.0` | Bind address |
| `--port <port>` | `8000` | Bind port |
| `--device <device>` | `cuda` | torch device |
| `--num-inference-steps N` | (from ckpt) | Override denoising/ODE steps for faster inference |
| `--noise-scheduler-type TYPE` | (from ckpt) | e.g. `DDIM` for DP (default DDPM is 100 steps = 450ms) |
| `--compile` | off | Enable torch.compile (~1.3-10× speedup, adds warmup) |
| `--chunk-mode` / `--no-chunk-mode` | on | Return full action chunk vs single action |
| `--rtc` | off | Enable Real-Time Chunking (VLAs only) |

Example:

```bash
# Fast DP serving (DDIM-10 + compile):
lobe-serve --checkpoint=checkpoints/.../pretrained_model --noise-scheduler-type=DDIM --num-inference-steps=10 --compile

# Fast FM serving (5-step + compile):
lobe-serve --checkpoint=checkpoints/.../pretrained_model --num-inference-steps=5 --compile
```

See [Serving workflow](../workflows/serving.md) for the protocol, test client, and benchmarks.

## Utility scripts

| Script | Purpose |
|---|---|
| `scripts/validate_yam_dataset.py` | Validate limb-exported YAM dataset (schema, stats, video alignment) |
| `scripts/convert_yam_video_to_image.py` | Convert video→image format for 20× faster training |
| `scripts/eval_replay.py` | Replay-based MSE eval on held-out episodes (no sim needed) |
| `scripts/test_serve_all.py` | End-to-end serving test (start server → send obs → verify action shape) |
| `scripts/bench_inference.py` | Raw forward-pass latency benchmark (compiled vs uncompiled) |
