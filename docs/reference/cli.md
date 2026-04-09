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

Example:

```bash
lobe-serve --checkpoint=/mnt/localssd/$USER/checkpoints/.../pretrained_model --port 8000
```

See [Serving workflow](../workflows/serving.md) for the protocol and a test client.
