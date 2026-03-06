# LOBE — Learning Orchestration, Brain-to-Embodiment

> The motor cortex (frontal **lobe**) learns to control **limb**s.

## Project Goal

Policy training and serving companion repo for [limb](https://github.com/TToTMooN/limb).
limb handles robot control + data collection; LOBE handles policy training + serving.
Primary use cases: fine-tuning VLA models (X-VLA, pi0.5, WALL-OSS) and serving them over WebSocket/OpenPI.

---

## Architecture

```
configs/
  train_xvla.yaml           # X-VLA fine-tuning config
  train_pi0.yaml             # pi0/pi0.5 fine-tuning config
  train_walloss.yaml         # WALL-OSS fine-tuning config
scripts/
  train.sh                   # Training launcher
  serve_policy.py            # WebSocket policy server (limb connects here)
datasets/                    # Local LeRobot v2.1 datasets (or use HF Hub)
checkpoints/                 # Training outputs
pyproject.toml
```

### End-to-End Workflow

1. **Collect data** — on robot machine with limb (teleop + episode recording)
2. **Convert** — limb's `convert_to_lerobot.py` → LeRobot v2.1 format
3. **Train** — on GPU machine with lobe (`lerobot-train`)
4. **Serve** — on GPU machine with lobe (OpenPI or WebSocket server)
5. **Deploy** — on robot machine with limb (connects to policy server)

### Model Matrix

| Model | Params | Training Speed | Best For |
|-------|--------|---------------|----------|
| X-VLA | 0.9B | Fast (~2h on 1xA100) | Small datasets (50-200 episodes), quick iteration |
| pi0.5 | 3B | Medium (~8h on 1xA100) | Best zero-shot transfer from ALOHA, robust |
| WALL-OSS | ~3B | Medium | MoE architecture, chain-of-thought reasoning |

### Action Space

- All models expect **224x224 RGB images** and **14-dim actions** (6 joints + 1 gripper per arm)
- YAM bimanual matches ALOHA's action space exactly (2x 6-DOF + gripper = 14 dims)
- pi0.5 requires a custom OpenPI transform class mapping YAM obs keys -> ALOHA keys

---

## Commands

```bash
# Train X-VLA (recommended for first experiments)
uv run lerobot-train \
  --dataset.repo_id=yourname/yam-red-cube \
  --policy.path=lerobot/xvla-base \
  --policy.dtype=bfloat16 \
  --batch_size=8 \
  --steps=20000 \
  --output_dir=checkpoints/xvla-yam-red-cube

# Train pi0.5
uv run lerobot-train \
  --dataset.repo_id=yourname/yam-red-cube \
  --policy.path=physical-intelligence/pi0.5 \
  --policy.dtype=bfloat16 \
  --steps=20000 \
  --output_dir=checkpoints/pi05-yam-red-cube

# Train WALL-OSS
uv run lerobot-train \
  --dataset.repo_id=yourname/yam-red-cube \
  --policy.type=wall_x \
  --policy.path=x-square-robot/wall-oss-flow \
  --steps=20000 \
  --output_dir=checkpoints/walloss-yam-red-cube

# Serve (OpenPI for pi0/pi0.5)
openpi serve --checkpoint checkpoints/pi05-yam-red-cube --port 8111

# Serve (WebSocket for X-VLA/WALL-OSS)
uv run python scripts/serve_policy.py \
  --checkpoint checkpoints/xvla-yam-red-cube \
  --host 0.0.0.0 --port 8000
```

---

## Development Conventions

- **Package manager**: `uv` (not pip). Run everything with `uv run --index-strategy unsafe-best-match` (needed for PyTorch nightly index).
- **Python**: 3.11 exactly
- **Logging**: `loguru` everywhere — `from loguru import logger`. Never use `print()` or `logging`.
- **Linter**: `ruff` (line length 119, config in pyproject.toml)
- **Training framework**: LeRobot (HuggingFace)
- **Datasets**: LeRobot v2.1 format
- **Model hub**: HuggingFace Hub for datasets and checkpoints

### Lint

```bash
uv run ruff check .
uv run ruff format .
```

---

## Key Dependencies

```
lerobot[pi,xvla,wallx]  # Training framework with VLA policy support
huggingface-hub          # Dataset/checkpoint management
wandb                    # Experiment tracking
websockets               # Policy server (serve extra)
msgpack                  # Wire serialization (serve extra)
msgpack-numpy            # NumPy array serialization (serve extra)
```

Install: `uv sync`
Install with serving: `uv sync --extra serve`
