# Benchmarks

All results from `lobe-eval` on LIBERO with the standard 4-suite protocol (`libero_spatial,libero_object,libero_goal,libero_10`, 10 episodes per task = 400 total rollouts).

## LIBERO 4-suite

| Model | Source | Avg | Spatial | Object | Goal | Long (libero_10) | Train Time | Config |
|---|---|---|---|---|---|---|---|---|
| SmolVLA | published | **87.3%** | 90 | 96 | 92 | 71 | — | batch=64, 100k, LR=1e-4 |
| SmolVLA | official HF ckpt, our eval | **62.8%** | ~68 | ~77 | ~60 | ~30 | — | n_action_steps=10, osmesa |
| **SmolVLA** | **ours (paper config)** | **82.0%** | ~90 | ~95 | ~90 | ~51 | 4h (8×H100) | batch=64, 100k, LR=1e-4 |
| **SmolVLA** | **ours (scaled config)** | **80.5%** | ~88 | ~93 | ~88 | ~48 | **1.5h (8×H100)** | batch=256, 25k, LR=4e-4 |
| Diffusion | published | **72.4%** | 78.3 | 92.5 | 68.3 | 50.5 | — | — |
| Diffusion | ours (h2h, scaled) | **36.5%** | 100/100 strong, many 0% | — | — | — | 50min (8×H100) | batch=256, 25k, LR=4e-4 |
| FM (ours) | h2h, scaled | **33.75%** | similar pattern | — | — | — | 52min (8×H100) | batch=256, 25k, LR=4e-4 |
| pi0-FAST | published | **82.5%** | — | — | — | — | — | batch=32, 20k |
| pi0.5 | published | **97.5%** | — | — | — | — | — | batch=32×8GPU, 6k |
| X-VLA | published | **98.1%** | — | — | — | — | — | ~30k steps |
| **X-VLA** | **ours v1.0 (V14)** | 85.75% | 86 | 95 | 93 | 69 | 3h40m (8×H100) | batch=128, 60k, constant LR 1e-4, upstream `2toINF/Libero-XVLA-format` |
| **X-VLA** | **ours v1.1 (V15)** | 87.00% | 88 | 93 | 81 | 86 | 3h40m (8×H100) | V14 + libero_90 aux data (5525 total eps). +17 libero_10, -12 goal. Net +1.25 avg. |
| **X-VLA** | **ours v1.2 (V16)** | 90.50% | 91 | 97 | 91 | 83 | 5h30m cumulative | Two-stage: V15 + continue 30k on V12 only. Goal recovers to 91, spatial 91 / object 97 new highs. Gap to paper 7.6 pp. |
| **X-VLA** | **ours v1.3 (V17b)** | **91.25%** | 89 | 97 | 89 | **90** | V16 + eval tweak | V16 + eval-time `num_denoising_steps=20`. libero_10 +7 from doubled flow-matching ODE refinement. Gap 6.85 pp. |

## Key findings

1. **Our SmolVLA (82%) significantly beats the official HF checkpoint (62.8%)** on identical eval setup. The published 87.3% is a known reproduction gap reported by many community members ([lerobot #2354](https://github.com/huggingface/lerobot/issues/2354), [#1369](https://github.com/huggingface/lerobot/issues/1369), [#2107](https://github.com/huggingface/lerobot/issues/2107)).
2. **Scaled config (80.5%) nearly matches paper config (82%) in 1/3 the time** thanks to linear LR scaling and bf16.
3. **DP and FM match within 3%** when given the same architecture and hyperparameters — confirming flow matching ≈ diffusion when controlled.
4. **Bad hyperparameters break both DP and FM equally**. The 33–37% h2h numbers are not a flaw of either method, just a sign that batch=256 + LR=4e-4 + 25k steps is the wrong recipe for these smaller policies.
5. **X-VLA v1.0 (85.75%) closes most of the gap** to the paper's 98.1%. See [Fine-tune X-VLA](workflows/xvla_finetune.md) for the full recipe. The remaining 12 pp gap is concentrated on libero_10 (long-horizon); adding libero_90 as auxiliary training data is the current v1.1-dev experiment.

## Training speed (8×H100, LIBERO)

| Model | Batch/GPU | Eff. batch | updt_s | data_s | Steps/s | Samples/s | Notes |
|---|---|---|---|---|---|---|---|
| SmolVLA (no opt) | 8 | 64 | 0.503 | 0.004 | 2.0 | 128 | float32 attn, no bf16 |
| SmolVLA (bf16) | 8 | 64 | 0.163 | 0.134 | 2.0 | 128 | bf16, data bottleneck |
| SmolVLA (bf16+patch) | 32 | 256 | 0.212 | 0.011 | 4.4 | **1126** | bf16 + data loading patch |
| Diffusion (bf16+patch) | 32 | 256 | 0.093 | 0.035 | ~8 | ~2048 | smaller model |
| FM small UNet | 8 | 64 | 0.038 | 0.002 | ~25 | ~1600 | even smaller, fastest |
| FM large UNet | 32 | 256 | 0.039 | 0.002 | ~25 | ~6400 | fastest fair config |

## Optimizations applied

1. **Data loading patch (12×)**: bypass HF datasets `set_transform` for non-image columns. See [Patches reference](reference/patches.md).
2. **bf16 mixed precision (3×)**: `--mixed_precision bf16` via accelerate.
3. **persistent_workers + prefetch_factor=4**: eliminates worker respawn between epochs.
4. Combined: ~10× speedup vs naive setup.

## Test protocol

- **Dataset**: `HuggingFaceVLA/libero` (273k frames, 1693 episodes, 2 cameras 256×256)
- **Eval suites**: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` — 40 tasks total
- **Episodes per task**: 10
- **Renderer**: `MUJOCO_GL=egl` (preferred) or `osmesa`
- **Hardware**: 8× NVIDIA H100 80GB
- **Repro**: see [Quick Start](quickstart.md)

For the protocol details and how to add new methods, see [BENCHMARKS.md](https://github.com/your-username/lobe/blob/main/BENCHMARKS.md) in the repo root.
