# Configuration

LOBE itself has very little configuration — it inherits everything from lerobot's [draccus](https://github.com/dlwh/draccus) CLI system.

## CLI overrides

Every field of the policy/dataset/env/optim/eval configs is exposed as a CLI flag:

```bash
--policy.type=diffusion
--policy.optimizer_lr=1e-4
--policy.n_action_steps=10
--dataset.repo_id=HuggingFaceVLA/libero
--env.type=libero
--env.task=libero_spatial,libero_object,libero_goal,libero_10
--eval.batch_size=1 --eval.n_episodes=10
--optimizer.lr=4e-4
--scheduler.warmup_steps=500
```

Tuple/list values use bracket syntax:

```bash
--policy.down_dims=[512,1024,2048]
```

Dict values use JSON syntax (quote the whole arg):

```bash
'--rename_map={"observation.images.image": "observation.images.camera1"}'
```

## Environment variables

LOBE respects standard HuggingFace and PyTorch env vars:

| Variable | Purpose |
|---|---|
| `HF_HOME` | HuggingFace cache root |
| `HF_DATASETS_CACHE` | HF datasets cache (subdirectory of HF_HOME by default) |
| `TRANSFORMERS_CACHE` | Transformers model cache |
| `WANDB_DIR` | Weights & Biases logs dir |
| `WANDB_API_KEY` | W&B authentication |
| `TMPDIR` | Temp directory (set to SSD if root disk is small) |
| `MUJOCO_GL` | MuJoCo rendering backend (`egl`, `osmesa`, or `glfw`) |
| `PYOPENGL_PLATFORM` | PyOpenGL backend (match `MUJOCO_GL`) |
| `CUDA_VISIBLE_DEVICES` | Restrict to specific GPUs |

We strongly recommend pointing all caches to a fast SSD with plenty of space:

```bash
export HF_HOME=/mnt/localssd/$USER/cache/huggingface
export WANDB_DIR=/mnt/localssd/$USER/wandb
export TMPDIR=/mnt/localssd/$USER/tmp
```

Add these to `~/.bashrc` to make them persistent.

## Pinning lerobot

For reproducibility, pin lerobot to a specific commit in `pyproject.toml`:

```toml
[tool.uv.sources]
lerobot = { git = "https://github.com/huggingface/lerobot.git", rev = "<specific-sha>" }
```

This prevents auto-upgrades that might break our patches or change behavior.

## Override torch and huggingface-hub versions

lerobot often pins to versions that conflict with PyTorch nightly. We override them in `pyproject.toml`:

```toml
[tool.uv]
override-dependencies = [
    "torch>=2.2",
    "torchvision>=0.20",
    "huggingface-hub>=0.34,<2",
]
index-strategy = "unsafe-best-match"
```
