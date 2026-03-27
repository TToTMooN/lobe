# LOBE — Learning Orchestration, Brain-to-Embodiment

> **First principles: clean, essential, nothing more.** This is a unified library to train and serve policies for [limb](https://github.com/TToTMooN/limb). Check existing implementations before reinventing. Focus on what downstream robot deployment actually needs.

> The motor cortex (frontal **lobe**) learns to control **limb**s.

## Project Goal

Policy training and serving companion repo for [limb](https://github.com/TToTMooN/limb).
limb handles robot control + data collection; LOBE handles policy training + serving.
Primary use cases: training visuomotor baselines (FM, Diffusion), fine-tuning VLAs (SmolVLA, pi0), and serving them over WebSocket.

---

## Architecture

```
lobe/
  configs/                  # Training configs (tyro dataclasses + named presets)
    base.py                 # Dataclass definitions
    pusht.py                # PushT presets
    aloha.py                # ALOHA presets
  envs/                     # Per-environment constants, data loading, eval
    pusht.py
    aloha.py
    yam_bimanual.py
  policies/
    flow_matching/          # Flow Matching policy (transformer + U-Net backbones)
    diffusion_wrapper.py    # Diffusion Policy (LeRobot wrapper with normalization)
    normalize.py            # Normalize/Unnormalize (MEAN_STD, MIN_MAX)
    factory.py              # create_policy(), load_checkpoint()
  data/
    fast_dataset.py         # GPU-resident .pt tensor cache
    loading.py              # LeRobot dataset loading
  serve.py                  # WebSocket policy server
  client.py                 # WebSocket client for testing
scripts/
  train.py                  # Main training script (preset subcommands)
  train_vla.py              # VLA fine-tuning via lerobot-train
  prepare_dataset.py        # Pre-resize + cache datasets as .pt
  validate_fm.py            # FM validation across configs
  eval_pusht_web.py         # Browser-based PushT eval viewer
```

### End-to-End Workflow

1. **Collect data** — on robot machine with limb (teleop + episode recording)
2. **Convert** — limb's `convert_to_lerobot.py` → LeRobot v2.1 format
3. **Train** — on GPU machine with lobe (`scripts/train.py` or `scripts/train_vla.py`)
4. **Serve** — on GPU machine with lobe (WebSocket server)
5. **Deploy** — on robot machine with limb (connects to policy server)

---

## Commands

```bash
# ── Visuomotor baselines (FM / Diffusion) ──

# Train with named presets (recommended):
uv run python scripts/train.py pusht-fm                         # PushT + Flow Matching
uv run python scripts/train.py pusht-fm --train.steps 25000     # override any field
uv run python scripts/train.py pusht-fm --policy.backbone unet  # switch backbone
uv run python scripts/train.py aloha-fm-fast                    # ALOHA + pre-cached data
uv run python scripts/train.py --help                           # list all presets

# ── VLA fine-tuning (via lerobot-train) ──

uv run python scripts/train_vla.py --model smolvla --dataset lerobot/aloha_sim_transfer_cube_human_image
uv run python scripts/train_vla.py --model smolvla --dataset yourname/yam-data --steps 50000

# ── Data preparation ──

uv run python scripts/prepare_dataset.py lerobot/aloha_sim_insertion_human_image --resize 224

# ── Serving ──

uv run python -m lobe.serve --checkpoint checkpoints/pusht_fm/flow_matching_50000 --host 0.0.0.0

# ── Validation ──

uv run python scripts/validate_fm.py --tests 1,2,3 --steps 10000   # PushT regression
uv run python scripts/validate_fm.py --steps 25000                  # full validation

# ── Lint ──

uv run ruff check .
uv run ruff format .
```

---

## Config System

Pure Python dataclasses with `tyro.extras.overridable_config_cli` (nerfstudio pattern). Presets are subcommands, every field overridable from CLI.

```
lobe/configs/
    __init__.py   # merges all env presets into PRESETS dict
    base.py       # dataclass definitions (EnvConfig, FMPolicyConfig, etc.)
    pusht.py      # PushT presets
    aloha.py      # ALOHA presets
```

**Adding a new env:** create `lobe/configs/myenv.py` with a `PRESETS` dict, add one import to `__init__.py`.

**Adding a new policy:** add dataclass to `base.py`, add to `TrainPipelineConfig.policy` union.

---

## Development Conventions

- **Package manager**: `uv` (not pip)
- **Python**: >=3.12
- **Logging**: `loguru` everywhere — `from loguru import logger`. Never `print()` or `logging`.
- **Linter**: `ruff` (line length 119, config in pyproject.toml)
- **Training framework**: LeRobot (HuggingFace)
- **Datasets**: LeRobot v2.1 format (image preferred over video for speed)
- **Configs**: Python dataclasses + tyro (no YAML)
- **Shared machine**: always check `nvidia-smi` before launching training

---

## Key Dependencies

```
lerobot[pi,smolvla]      # Training framework with VLA policy support
huggingface-hub          # Dataset/checkpoint management
wandb                    # Experiment tracking
tyro                     # CLI config parsing
websockets               # Policy server (serve extra)
msgpack                  # Wire serialization (serve extra)
```

Install: `uv sync`
Install with serving: `uv sync --extra serve`
