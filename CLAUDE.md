# LOBE — Learning Orchestration, Brain-to-Embodiment

> The motor cortex (frontal **lobe**) learns to control **limb**s.

## First Principles

1. **Check existing implementations before reinventing.** PushT, LIBERO, robomimic are well-researched — search papers and codebases for what works before writing new code.
2. **Verify with evidence, not assumptions.** Always check reported success rates on the EXACT dataset/env you're using. Different dataset variants (e.g. standard ALOHA vs AV-ALOHA) give wildly different results.
3. **Clean, essential, nothing more.** No premature abstractions. Three similar lines > a helper nobody needs yet.
4. **Shared machine etiquette.** Always `nvidia-smi` before launching. Don't hog GPU when others need it.
5. **Data loading is always the bottleneck.** Use image-format datasets (not video). Use more workers. For small datasets, pre-cache as .pt tensors.
6. **Benchmark on tasks where your policy type is KNOWN to work.** Diffusion/FM are proven on PushT, LIBERO, robomimic. Don't waste time debugging a policy on a benchmark where it was never shown to work.

## Key References

- **Diffusion Policy** (Chi et al. 2023): 95-100% on robomimic, ~0.91 IoU on PushT. Gold standard for visuomotor baselines.
- **Flow Matching** (Lipman et al. 2023): Conditional OT flow matching. Simpler than DDPM, same or better quality.
- **LeRobot** (HuggingFace): Training framework. Native support for PushT, LIBERO, MetaWorld.
- **LIBERO** (Liu et al. 2023): 130 tasks, 5 suites. 7-DOF manipulation. Our primary sim benchmark.
- **SmolVLA** (HuggingFace 2025): 450M VLA, flow matching action expert. 87.3% on LIBERO (batch=64, 100k steps).

## Current Status & Next Steps

See `.claude/projects/-home-lingfeng-playground-lobe/memory/MEMORY.md` for detailed agent memory (project state, lessons learned, user preferences).

### Verified results
- FM on PushT: 40% (transformer, 10k steps), 60% (unet, 10k steps) — both backbones work
- FM on PushT: needs full 50k run for final numbers

### In progress / next
1. **SmolVLA on LIBERO** — reproduce 87.3% published result
   - Dataset: `HuggingFaceVLA/libero` (273k examples, 22.4k episodes, 2 cameras)
   - Config: batch=64, 100k steps, `lerobot/smolvla_base` pretrained
   - Camera remap: image→camera1, image2→camera2, 1 empty
   - Command: `uv run python scripts/train_vla.py --model smolvla --dataset HuggingFaceVLA/libero --steps 100000 --batch-size 64`
2. **FM on LIBERO** — test if our FM policy works on real manipulation
   - Command: `uv run python scripts/train.py libero-fm --performance.no-compile`
   - Diffusion Policy baseline: 72.4% on LIBERO (published)
   - If FM fails here too, compare against LeRobot's built-in diffusion
3. **Eval**: use `lerobot-eval --env.type=libero --env.task=libero_10` after training

### Published LIBERO baselines (target numbers)
| Model | Params | Avg Success | Config |
|-------|--------|-------------|--------|
| Diffusion Policy | - | 72.4% | - |
| SmolVLA | 0.45B | 87.3% | batch=64, 100k steps |
| pi0-FAST | 3B | 82.5% | batch=32, 20k steps |
| pi0.5 | 3B | 97.5% | batch=32x8GPU, 6k steps |
| X-VLA | 0.9B | 98.1% | ~30k steps |

---

## Project Goal

Policy training and serving companion for [limb](https://github.com/TToTMooN/limb).
limb handles robot control + data collection; LOBE handles policy training + serving.

---

## Architecture

```
lobe/
  configs/                  # Training configs (tyro dataclasses + named presets)
    base.py                 # Dataclass definitions (FMPolicyConfig, DiffusionPolicyConfig, etc.)
    pusht.py                # PushT presets
    libero.py               # LIBERO presets
  envs/                     # Per-environment constants, data loading, eval
    pusht.py                # PushT (2D, quick sanity check)
    libero.py               # LIBERO (7-DOF manipulation, primary benchmark)
    yam_bimanual.py         # YAM (real robot, future)
  policies/
    flow_matching/          # Flow Matching policy
      modeling_flow_matching.py   # Core FM policy (train + inference)
      configuration_flow_matching.py
      flow_transformer.py         # DiT-style Transformer+AdaLN backbone
      vision_encoder.py           # ResNet18+GlobalAvgPool (512-d)
    diffusion_wrapper.py    # Diffusion Policy (LeRobot wrapper)
    normalize.py            # Normalize/Unnormalize (MEAN_STD, MIN_MAX)
    factory.py              # create_policy(), load_checkpoint()
  data/
    fast_dataset.py         # GPU-resident .pt tensor cache
    loading.py              # LeRobot dataset loading
  serve.py                  # WebSocket policy server
  experiment_log.py         # Append-only experiments.tsv logger
scripts/
  train.py                  # Main training (preset subcommands via tyro)
  train_vla.py              # VLA fine-tuning via lerobot-train
  prepare_dataset.py        # Pre-resize + cache datasets as .pt
  validate_fm.py            # FM validation across configs
```

---

## Commands

```bash
# ── Visuomotor baselines (FM / Diffusion) ──
uv run python scripts/train.py pusht-fm                              # PushT + Flow Matching
uv run python scripts/train.py libero-fm --performance.no-compile    # LIBERO + Flow Matching
uv run python scripts/train.py --help                                # list all presets

# ── VLA fine-tuning (reproducing published results) ──
uv run python scripts/train_vla.py --model smolvla --dataset HuggingFaceVLA/libero --steps 100000 --batch-size 64

# ── Evaluation ──
# SmolVLA/pi0 on LIBERO (via lerobot-eval):
lerobot-eval --policy.path=checkpoints/smolvla-libero-100k/checkpoints/100000/pretrained_model \
  --env.type=libero --env.task=libero_10 --eval.n_episodes=10 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'

# ── Lint ──
uv run ruff check .
uv run ruff format .
```

---

## Config System

Pure Python dataclasses + `tyro.extras.overridable_config_cli` (nerfstudio pattern).
Presets are subcommands. Every field overridable from CLI.

**Adding a new env:** create `lobe/configs/myenv.py` with a `PRESETS` dict, add one import to `__init__.py`.

**Adding a new policy:** add dataclass to `base.py`, add to `TrainPipelineConfig.policy` union.

---

## Setup

```bash
uv sync
# Install LIBERO benchmark
pip install robosuite==1.4.1 bddl easydict matplotlib gym
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /tmp/LIBERO
echo "/tmp/LIBERO" > .venv/lib/python3.*/site-packages/libero_path.pth
```

## Development Conventions

- **Package manager**: `uv` (not pip)
- **Python**: >=3.12
- **Logging**: `loguru` — `from loguru import logger`. Never `print()` or `logging`.
- **Linter**: `ruff` (line length 119)
- **Configs**: Python dataclasses + tyro. No YAML.
- **Experiment tracking**: wandb + experiments.tsv
- **Datasets**: LeRobot v2.1 format. Use `_image` variants for speed.
