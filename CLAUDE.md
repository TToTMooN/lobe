# LOBE — Learning Orchestration, Brain-to-Embodiment

> The motor cortex (frontal **lobe**) learns to control **limb**s.

## First Principles

1. **Check existing implementations before reinventing.** PushT, LIBERO, robomimic are well-researched — search papers and codebases for what works before writing new code.
2. **Verify with evidence, not assumptions.** Always check reported success rates on the EXACT dataset/env you're using. Different dataset variants (e.g. standard ALOHA vs AV-ALOHA) give wildly different results.
3. **Clean, essential, nothing more.** No premature abstractions. Three similar lines > a helper nobody needs yet.
4. **Shared machine etiquette.** Always `nvidia-smi` before launching. Don't hog GPU when others need it.
5. **Data loading is always the bottleneck.** Use image-format datasets (not video). Use more workers. For small datasets, pre-cache as .pt tensors.
6. **Benchmark on tasks where your policy type is KNOWN to work.** Diffusion/FM are proven on PushT, LIBERO, robomimic. Don't waste time debugging a policy on a benchmark where it was never shown to work.
7. **Log everything to experiments.tsv — even failures.** Every training run, eval, and debug attempt gets a row. This is the audit trail of how we got to the final result. Failed experiments are as valuable as successes — they record what we tried and why it didn't work. Think of it as auto-research: the tsv tells the full story.
8. **Use existing tools, don't reinvent.** For multi-GPU, use `accelerate`. For VLA training, use `lerobot-train`. Search other codebases first. Only build custom when you're certain it's cleaner.
9. **Maximize GPU utilization.** Use all available GPUs for a single training run (distributed), not one GPU per experiment. `accelerate launch --num_processes=N` for lerobot-based training.
10. **NEVER STOP.** Always have a training run, eval, or experiment running. When one finishes, immediately start the next item on the plan. Commit progress, log results, update the plan, and launch the next experiment. Idle GPUs are wasted money. If blocked on one task, start another in parallel.

## Key References

- **Diffusion Policy** (Chi et al. 2023): 95-100% on robomimic, ~0.91 IoU on PushT. Gold standard for visuomotor baselines.
- **Flow Matching** (Lipman et al. 2023): Conditional OT flow matching. Simpler than DDPM, same or better quality.
- **LeRobot** (HuggingFace): Training framework. Native support for PushT, LIBERO, MetaWorld.
- **LIBERO** (Liu et al. 2023): 130 tasks, 5 suites. 7-DOF manipulation. Our primary sim benchmark.
- **SmolVLA** (HuggingFace 2025): 450M VLA, flow matching action expert. 87.3% on LIBERO (batch=64, 100k steps).

## Current Status & Next Steps

See `.claude/projects/-home-lingfeng-playground-lobe/memory/MEMORY.md` for detailed agent memory (project state, lessons learned, user preferences).

### LIBERO Benchmark Results

#### Eval Results (osmesa rendering, `--eval.n_episodes=10` per task)
| Model | Params | All-Suite (40 tasks) | LIBERO-10 | Training Time | Steps/s | Notes |
|-------|--------|---------------------|-----------|---------------|---------|-------|
| **SmolVLA (ours)** | 450M (100M learnable) | **82.0%** | 51% | 4h (8×H100) | 7.0 | Paper config, bf16 |
| SmolVLA (official HF) | 450M | — | 41% | — | — | Known reproduction gap |
| SmolVLA (published) | 450M | 87.3% | — | — | — | Paper reference |
| Diffusion v2 (ours) | — | training... | — | ~11h (8×H100) | ~8 | batch=256, 200k steps |
| Diffusion v1 (ours) | — | 25.9% | — | 3.5h | 4.0 | FAILED: batch=512 too big |
| Diffusion (published) | — | 72.4% | — | — | — | Paper reference |
| FM (ours) | 16M | needs eval | — | 1.9h (1×H100) | 7.3 | loss=0.177, 50k steps |
| pi0-FAST (published) | 3B | 82.5% | — | — | — | batch=32, 20k steps |
| pi0.5 (published) | 3B | 97.5% | — | — | — | batch=32×8GPU, 6k steps |
| X-VLA (published) | 0.9B | 98.1% | — | — | — | ~30k steps |
| **X-VLA v10 (ours)** | 0.9B | 69.75% | 42% | 55min (8×H100) | 6.2 | Fine-tune of 2toINF/X-VLA-Pt on rel2abs-converted HuggingFaceVLA/libero (20k steps, batch 64). Per-suite: spatial 72% / object 90% / goal 75% / libero_10 42%. |
| **X-VLA v11 (ours)** | 0.9B | 72.25% | 50% | 3h40m (8×H100) | 4.6 | V10 + paper recipe match: 60k steps (3× more), batch 128 (2× larger), data aug (ColorJitter + RandomAffine + SharpnessJitter). Per-suite: spatial 80 / object 84 / goal 75 / libero_10 50. Small +2.5 avg — libero_10 gains offset by object regression. |
| **X-VLA v12 (ours)** | 0.9B | 84.00% | 72% | 3h40m (8×H100) | 4.6 | V11 + train on `2toINF/Libero-XVLA-format` (upstream OpenVLA-regenerated LIBERO with precomputed abs_action_6d — **eliminates 5-30 mm nearest-neighbor offset** in training targets). Per-suite: spatial 83 / object 93 / goal 88 / libero_10 72. +11.75 over V11 (including +22 on libero_10). |
| **X-VLA v14 (ours) — v1.0 recipe** | 0.9B | 85.75% | 69% | 3h40m (8×H100) | 4.6 | V12 + **constant LR** (`--policy.scheduler_decay_lr=1e-4` = peak, via `policy.*` flag to bypass the preset-overwrite in `TrainPipelineConfig.validate()` that had silently ignored all our `--scheduler.*` / `--optimizer.*` args since V11) + `weight_decay=0.01` + `grad_clip_norm=1.0`. Per-suite: **spatial 86 / object 95 / goal 93** / libero_10 69. +1.75 over V12. Gap to paper: 12.35 pp. |
| **X-VLA v15 (ours)** | 0.9B | **87.00%** | **86%** | 3h40m (8×H100) | 4.6 | V14 + libero_90 auxiliary training data via `2toINF/Libero-XVLA-format` (5525 total eps, 3.3× V14). Per-suite: **spatial 88** / object 93 / goal 81 / **libero_10 86**. +1.25 over V14 avg but **+17 on libero_10** — long-horizon tasks benefit most from libero_90's diverse scenes. Goal regressed -12 (side effect of more diverse training data diluting goal-conditioned alignment). Gap to paper: 11.1 pp. **Current best.** |

#### Training Speed Benchmarks (8×H100, LIBERO dataset)
| Model | Batch/GPU | Eff. Batch | updt_s | data_s | Steps/s | Samples/s | GPU Mem | Notes |
|-------|-----------|-----------|--------|--------|---------|-----------|---------|-------|
| SmolVLA (no opt) | 8 | 64 | 0.503 | 0.004 | 2.0 | 128 | 6GB | No bf16, no data fix |
| SmolVLA (bf16) | 8 | 64 | 0.163 | 0.134 | 2.0 | 128 | 6GB | bf16, data bottleneck |
| SmolVLA (bf16+fix) | 32 | 256 | 0.212 | 0.011 | 4.4 | 1126 | 12GB | **Best: bf16+data fix** |
| SmolVLA (bf16+fix) | 64 | 512 | 0.369 | 0.019 | 2.6 | 1311 | 22GB | Max throughput |
| Diffusion (bf16+fix) | 32 | 256 | 0.093 | 0.035 | ~8 | ~2048 | 14GB | Currently training |

#### Key Optimizations Applied
1. **Data loading fix (12×)**: Bypass HF datasets `set_transform` for non-image columns — avoids decoding 100 throwaway PNGs per action-chunk query
2. **bf16 mixed precision (3×)**: `--mixed_precision bf16` via accelerate
3. **persistent_workers + prefetch_factor=4**: Eliminates worker respawn overhead
4. **Lesson learned**: Always match published effective batch size, or scale LR proportionally (linear rule)

### Current Plan (in priority order — ALWAYS have something running)
1. ~~FM on LIBERO~~ ✓ trained (loss=0.177), needs eval implementation
2. ~~SmolVLA full benchmark~~ ✓ 82.0% across all suites
3. **Diffusion v2 on LIBERO** — running now (batch=256, 200k steps, ~11h)
4. **FM eval on LIBERO** — implement sim evaluate() for our FM policy
5. **Mujoco downgrade + re-eval** — try mujoco==3.3.2 to close SmolVLA eval gap
6. **More methods** — pi0, XVLA, ACT on LIBERO

### Eval notes
- Use `MUJOCO_GL=osmesa` (EGL broken due to NVIDIA driver version mismatch)
- Use `--policy.n_action_steps=10` for SmolVLA (1 gave worse results)
- Use `--eval.batch_size=1 --eval.n_episodes=10` per task
- Output dir on SSD: `/mnt/localssd/sunlingfeng/checkpoints/`
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
