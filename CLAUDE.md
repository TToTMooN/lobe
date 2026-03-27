# LOBE — Learning Orchestration, Brain-to-Embodiment

> The motor cortex (frontal **lobe**) learns to control **limb**s.

## First Principles

1. **Check existing implementations before reinventing.** PushT, LIBERO, robomimic are well-researched — search papers and codebases for what works before writing new code.
2. **Verify with evidence, not assumptions.** Always check reported success rates on the EXACT dataset/env you're using. Different dataset variants (e.g. standard ALOHA vs AV-ALOHA) give wildly different results.
3. **Clean, essential, nothing more.** No premature abstractions. Three similar lines > a helper nobody needs yet.
4. **Shared machine etiquette.** Always `nvidia-smi` before launching. Don't hog GPU when others need it.
5. **Data loading is always the bottleneck.** Use image-format datasets (not video). Use more workers. For small datasets, pre-cache as .pt tensors.
6. **Benchmark on tasks where your policy type is KNOWN to work.** Diffusion/FM are proven on PushT, LIBERO, robomimic. ACT is the only thing proven on ALOHA sim. Don't waste time debugging a policy on a benchmark where it was never shown to work.

## Key References

- **Diffusion Policy** (Chi et al. 2023): 95-100% on robomimic (Lift/Can/Square/Transport), ~0.91 IoU on PushT. The gold standard for visuomotor baselines.
- **Flow Matching for Generative Modeling** (Lipman et al. 2023): Conditional OT flow matching. Simpler than DDPM, same or better quality.
- **VITA** (ICLR 2026): Flow matching with action autoencoder. Claims 100% on CubeTransfer but uses custom AV-ALOHA dataset (21-dim), NOT standard ALOHA. Their FM baseline uses Transformer+AdaLN, global avg pool, MEAN_STD normalization.
- **LeRobot** (HuggingFace): Training framework. Native support for PushT, LIBERO, MetaWorld. Diffusion, ACT, pi0, SmolVLA policies.
- **LIBERO** (Liu et al. 2023): 130 tasks, 5 suites. 7-DOF manipulation. pi0.5 gets 97.5%. This is our primary sim benchmark.
- **SmolVLA** (HuggingFace 2025): 450M VLA. Designed for SO100 real-world, not sim. Paper recommends batch_size=64, 100k steps.

## Project Goal

Policy training and serving companion for [limb](https://github.com/TToTMooN/limb).
limb handles robot control + data collection; LOBE handles policy training + serving.
Primary use cases: training visuomotor baselines (FM, Diffusion), fine-tuning VLAs (SmolVLA, pi0), and serving over WebSocket.

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
      configuration_flow_matching.py
      modeling_flow_matching.py
      flow_transformer.py   # DiT-style Transformer+AdaLN backbone
      vision_encoder.py     # ResNet18+GlobalAvgPool (512-d, VITA-style)
    diffusion_wrapper.py    # Diffusion Policy (LeRobot wrapper)
    normalize.py            # Normalize/Unnormalize (MEAN_STD, MIN_MAX)
    factory.py              # create_policy(), load_checkpoint()
  data/
    fast_dataset.py         # GPU-resident .pt tensor cache
    loading.py              # LeRobot dataset loading
  serve.py                  # WebSocket policy server
  client.py                 # WebSocket client
  experiment_log.py         # Append-only experiments.tsv logger
scripts/
  train.py                  # Main training (preset subcommands)
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

uv run python scripts/train.py pusht-fm                              # PushT + Flow Matching
uv run python scripts/train.py libero-fm --performance.no-compile    # LIBERO + Flow Matching
uv run python scripts/train.py pusht-fm --policy.backbone unet       # switch backbone
uv run python scripts/train.py --help                                # list all presets

# ── VLA fine-tuning (via lerobot-train) ──

uv run python scripts/train_vla.py --model smolvla --dataset lerobot/libero_10_image
uv run python scripts/train_vla.py --model smolvla --dataset yourname/yam-data --steps 100000

# ── Serving ──

uv run python -m lobe.serve --checkpoint checkpoints/pusht_fm/flow_matching_50000

# ── Lint ──

uv run ruff check .
uv run ruff format .
```

---

## Config System

Pure Python dataclasses + `tyro.extras.overridable_config_cli` (nerfstudio pattern).
Presets are subcommands. Every field overridable from CLI.

```
lobe/configs/
    __init__.py   # merges all env presets
    base.py       # dataclass definitions
    pusht.py      # PushT presets
    libero.py     # LIBERO presets
```

**Adding a new env:** create `lobe/configs/myenv.py` with a `PRESETS` dict, add one import to `__init__.py`.

**Adding a new policy:** add dataclass to `base.py`, add to `TrainPipelineConfig.policy` union.

---

## Development Conventions

- **Package manager**: `uv` (not pip)
- **Python**: >=3.12
- **Logging**: `loguru` — `from loguru import logger`. Never `print()` or `logging`.
- **Linter**: `ruff` (line length 119)
- **Training framework**: LeRobot (HuggingFace)
- **Datasets**: LeRobot v2.1 format. Use `_image` variants (not video) for speed.
- **Configs**: Python dataclasses + tyro. No YAML.
- **Experiment tracking**: wandb + experiments.tsv (append-only local log)
- **Shared machine**: always check `nvidia-smi` before launching training

---

## Key Dependencies

```
lerobot[pi,smolvla]      # Training framework with VLA policy support
huggingface-hub          # Dataset/checkpoint management
wandb                    # Experiment tracking
tyro                     # CLI config parsing
libero                   # LIBERO benchmark (pip install from github)
```

Install: `uv sync`
