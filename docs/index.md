# LOBE

> The motor cortex (frontal **lobe**) learns to control **limb**s.

**LOBE** is a lightweight package for training, evaluating, and serving robot policies. It is the policy training and serving companion to [limb](https://github.com/TToTMooN/limb).

## What it does

- Train any [lerobot](https://github.com/huggingface/lerobot) policy (Diffusion, ACT, SmolVLA, pi0, XVLA, ...) **and** custom policies registered as plugins
- Evaluate trained checkpoints on [LIBERO](https://libero-project.github.io/) and other simulation benchmarks
- Serve trained policies over WebSocket to real robots controlled by limb
- All three operations work for any policy type with one CLI

## The standard workflow

Given a dataset collected with limb (or any LeRobot-format dataset):

```bash
# Train (any policy type)
lobe-train --policy.type=diffusion --dataset.repo_id=yourname/yam-cube
lobe-train --policy.type=flow_matching --dataset.repo_id=yourname/yam-cube
lobe-train --policy.path=lerobot/smolvla_base --dataset.repo_id=yourname/yam-cube

# Evaluate
lobe-eval --policy.path=checkpoints/.../pretrained_model --env.type=libero

# Serve to robot
lobe-serve --checkpoint=checkpoints/.../pretrained_model --port 8000
```

## Why LOBE on top of lerobot?

lerobot has all the building blocks (policies, training, eval, datasets) but they are sprawled across many CLIs and need glue code for custom policies. LOBE provides:

| | bare lerobot | LOBE |
|---|---|---|
| Custom policies (e.g. Flow Matching) | manual registration boilerplate | `lobe/__init__.py` auto-registers |
| Data loading optimization | edit installed files (lost on `uv sync`) | `lobe/patches.py` monkey-patch on import |
| Single CLI surface | `lerobot-train`, `lerobot-eval`, separate setup | `lobe-train`, `lobe-eval`, `lobe-serve` (all preconfigured) |
| Serving | not provided | `lobe-serve` works with any policy |

LOBE is a thin wrapper. The actual policies, training loops, and eval harnesses come from lerobot.

## Verified results (LIBERO 4-suite)

| Model | Source | Avg | Train Time |
|---|---|---|---|
| SmolVLA (ours, paper config) | 100k steps, batch=64 | **82.0%** | 4h on 8×H100 |
| SmolVLA (ours, scaled config) | 25k steps, batch=256 | **80.5%** | 1.5h on 8×H100 |
| SmolVLA (official HF checkpoint) | — | 41-62.8% | known repro gap |
| Diffusion (ours, h2h) | 25k steps, batch=256 | 36.5% | 50min on 8×H100 |
| FM (ours, h2h) | 25k steps, batch=256 | 33.75% | 52min on 8×H100 |

See [Benchmarks](benchmarks.md) for the full table and per-suite breakdown.

## Get started

Read the [Quick Start](quickstart.md) to install and run your first training in five minutes.
