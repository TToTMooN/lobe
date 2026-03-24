# LOBE Implementation Plan

## Guiding Principle

**PushT is the verification gate.** Every capability (data loading, training, evaluation, serving) must be validated end-to-end on PushT before moving to real robot data. PushT is small, fast, well-benchmarked by the community, and catches integration bugs early. Nothing moves to Phase 2+ until PushT results match or exceed published baselines.

## Overview

Three phases, each self-contained and deployable:

1. **LeRobot-native** — diffusion + flow matching policies, then VLAs, all via LeRobot
2. **OpenPI integration** — pi0/pi0.5 fine-tuning + serving via Physical Intelligence's stack
3. **Unified serving layer** — single WebSocket server wrapping any backend

---

## Phase 1: LeRobot-Native Training + Serving

### 1a. PushT Baseline (CURRENT — verification gate)

**Goal**: Train FM and Diffusion on PushT, achieve published success rates, validate full pipeline.

**Completed:**
- [x] Flow Matching Policy implementation (Option A — patched DiffusionPolicy)
  - `lobe/policies/flow_matching/` — config + model
  - Same U-Net architecture, linear interpolation + velocity field + Euler ODE
  - 14 unit tests passing
- [x] `lobe/video_compat.py` — PyAV compatibility for PyTorch nightly
- [x] `scripts/train_pusht.py` — training script with performance optimizations
  - Image dataset (`lerobot/pusht_image`) — 12x faster than video loading
  - bf16 autocast, TF32, torch.compile, cudnn.benchmark
  - 13 steps/s (3,300+ samples/s) on H100 at batch=256
- [x] `scripts/eval_pusht.py` — interactive pygame evaluation viewer
- [x] `scripts/sweep_pusht.py` — batch sweep CLI for hyperparameter comparison
- [x] Training benchmark: FM 5000 steps in 6.4 min, Diffusion in 6.7 min on H100

**In progress:**
- [x] Train to convergence (20,000 steps) — done for both FM and Diffusion
- [x] Quantitative eval via sweep — done, results below
- [x] Validate checkpoint loading in eval scripts — fixed torch.compile _orig_mod bug
- [ ] **Debug FM inference quality gap** — FM underperforms Diffusion (see results)
- [ ] Test with more training steps (LeRobot's pretrained uses 200k)

**Results (20k steps, fp32, batch=256, 10 rollouts):**

| Policy | 1 step | 4 steps | 10 steps | Latency |
|--------|--------|---------|----------|---------|
| Flow Matching | 10% / 0.16 | 0% / 0.14 | **40% / 0.23** | 16ms |
| Diffusion | N/A | N/A | **30-70% / 0.47** | 60ms |

FM consistently ~15-20% success across all configurations tested (v1-v8).
Diffusion consistently ~65% success at 50k steps. FM+U-Net needs significantly more
training (HRI-EU uses ~1.2M steps) and/or EMA to match diffusion. Not worth blocking on.

**Decision: Use Diffusion Policy as verified PushT baseline. Move to VLA training.**
VLA architectures (pi0, xvla, smolvla) use FM natively with transformer backbones
where it's proven to work. FM+U-Net can be revisited later if needed.

**Success criteria (gate to 1b):**
- Diffusion policy achieves >0.5 avg reward on PushT — **PASSED (0.45+ at 50k steps)**
- Full train→eval→serve loop works end-to-end

### 1b. Dataset pipeline

- [ ] Verify limb's `convert_to_lerobot.py` output loads correctly in LeRobot
- [ ] Document expected obs/action key mapping for YAM bimanual
- [ ] Test with image-based storage (proven faster than video in 1a)

### 1c. Training configs (VLAs)

**Tier 1 — small/fast:**
- [ ] X-VLA (0.9B, soft-prompt fine-tuning)
- [ ] SmolVLA (lightweight VLA, flow matching action expert)

**Tier 2 — production:**
- [ ] pi0 / pi0-FAST (3B, flow matching, best pretrained)
- [ ] pi0.5 (3B, best ALOHA transfer)

**Tier 3 — experimental:**
- [ ] WALL-OSS (MoE, chain-of-thought)
- [ ] GR00T N1.5 (requires NVIDIA Isaac — evaluate ROI first)

### 1d. Serving (LeRobot inference)

- [ ] `scripts/serve_policy.py` — WebSocket server implementing limb's policy_server_spec
  - Load any LeRobot checkpoint (diffusion, flow matching, VLA — same interface)
  - Expose metadata (image size, action dims) on connect
  - Accept obs → return action chunks via msgpack
- [ ] Test with PushT policy first, then robot policies

### Milestones

1. **PushT verified** — FM + Diffusion trained, evaluated, success rates match baselines
2. **Diffusion Policy on robot** — validates full data→train→serve→deploy loop
3. **Flow Matching on robot** — 1-step inference advantage
4. **X-VLA on robot** — first VLA deployment

---

## Phase 2: OpenPI Integration

**Goal**: Fine-tune pi0/pi0.5 using OpenPI's stack for best quality, serve via OpenPI.

**Gate**: Phase 1 milestones 1-2 completed.

- [ ] OpenPI setup (JAX + CUDA versioning)
- [ ] YAM transform class (map YAM obs keys → pi0 input format)
- [ ] Training configs for pi0 / pi0.5 on YAM data
- [ ] OpenPI serving (limb already has OpenPI client)

### Milestone

pi0.5 fine-tuned via OpenPI, served, running on robot via limb.

---

## Phase 3: Unified Serving Layer

**Goal**: Single server wrapping any backend (LeRobot, OpenPI, future).

**Gate**: Phase 2 milestone completed.

- [ ] `lobe/serve/server.py` — WebSocket + msgpack server
- [ ] `PolicyBackend` protocol with LeRobot + OpenPI backends
- [ ] Config-driven launch (YAML: backend type, checkpoint, device, host/port)

### Milestone

One command to serve any model. limb connects with same `PolicyClient` regardless of backend.

---

## Hardware

- **Training**: H100 PCIe 80GB (CUDA 13.0, PyTorch nightly cu128)
- **Deployment**: RTX 5090 (sm_120 Blackwell, cu128 nightly required)

## Key Learnings

1. **Data loading is the bottleneck, not GPU.** Use `lerobot/pusht_image` (pre-decoded) over `lerobot/pusht` (video). AV1 video decode is ~220ms/frame. Image loading is near-instant. This applies to all LeRobot datasets — always prefer image format for training.
2. **torch.compile gives ~50% speedup** after warmup on H100. Worth it for runs >1000 steps.
3. **bf16 + TF32** are free performance on Ampere+. Always enable.
4. **16 dataloader workers** is the sweet spot on this machine.
