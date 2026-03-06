# LOBE Implementation Plan

## Overview

Three phases, each self-contained and deployable:

1. **LeRobot-native** — diffusion + flow matching policies, then VLAs, all via LeRobot
2. **OpenPI integration** — pi0/pi0.5 fine-tuning + serving via Physical Intelligence's stack
3. **Unified serving layer** — single WebSocket server wrapping any backend

---

## Policy Landscape

### What's in LeRobot (official, `src/lerobot/policies/`)

| Policy | Type | Flow Matching? | Notes |
|--------|------|---------------|-------|
| `diffusion` | Diffusion Policy | No (DDPM denoising) | Strong baseline, well-tested |
| `pi0` | VLA | **Yes** (flow matching action expert) | 3B, pretrained on 8 robot platforms |
| `pi0_fast` | VLA | **Yes** + FAST tokenizer | Faster inference than pi0 |
| `pi05` | VLA | **Yes** | Upgraded pi0 with knowledge insulation |
| `smolvla` | VLA | **Yes** (flow matching action expert) | Lightweight VLA, community data |
| `xvla` | VLA | **Yes** | 0.9B, soft-prompt fine-tuning |
| `wall_x` | VLA | Diffusion or next-token | MoE architecture |
| `act` | Attention cloning | No | Simple baseline |
| `vqbet` | VQ-BeT | No | Discretized actions |
| `groot` | VLA | Unknown | NVIDIA GR00T N1.5 |
| `tdmpc` | Model-based RL | No | |
| `sac` | RL | No | |

### Flow matching outside LeRobot

| Project | What | LeRobot compat? |
|---------|------|-----------------|
| [HRI-EU/flow_matching](https://github.com/HRI-EU/flow_matching) | Affordance-based flow matching policy. Invited to integrate into LeRobot PushT. | Partial — same data format, not a LeRobot policy class |
| [VITA](https://github.com/ucd-dare/VITA) (ICLR 2026) | Noise-free flow matching in latent space. 1-step ODE inference. | Uses LeRobot datasets (HF format → Zarr), separate training framework |
| [Streaming Flow Policy](https://streaming-flow-policy.github.io/) (ICRA 2025) | Flows from narrow Gaussian around last action instead of noise. Faster inference. | Research paper, no LeRobot integration yet |
| [FlowPolicy](https://github.com/zql-kk/FlowPolicy) (AAAI 2025 Oral) | 3D flow-based policy via consistency flow matching. | Standalone, not LeRobot |

### Key insight

**Flow matching is already the training objective for pi0, pi0_fast, pi0.5, SmolVLA, and X-VLA in LeRobot.** These are all flow matching policies — they just happen to also be VLAs. For a pure flow matching policy *without* a VLM backbone (closer to diffusion policy but with flow matching loss), the options are:

1. **Modify LeRobot's diffusion policy** to use flow matching instead of DDPM (relatively straightforward — same U-Net, different noise schedule + loss)
2. **Port HRI-EU/flow_matching** into a LeRobot policy class
3. **Use pi0 with `train_expert_only=true`** — this trains only the flow matching action expert, effectively using it as a flow matching policy with a frozen VLM encoder
4. **SmolVLA** — smallest VLA (uses flow matching), fast to train

---

## Phase 1: LeRobot-Native Training + Serving

**Goal**: End-to-end fine-tuning and deployment using only LeRobot.

### 1a. Dataset pipeline

- [ ] Script: `scripts/download_dataset.py` — pull LeRobot datasets from HF Hub
- [ ] Script: `scripts/inspect_dataset.py` — print dataset stats (episodes, shapes, tasks)
- [ ] Verify limb's `convert_to_lerobot.py` output loads correctly in LeRobot
- [ ] Document expected obs/action key mapping for YAM bimanual

### 1b. Training configs

**Tier 1 — baselines (no pretrained VLM needed):**
- [ ] `configs/diffusion.yaml` — Diffusion Policy (DDPM, strong baseline, well-understood)
- [ ] `configs/flow_matching.yaml` — Flow Matching Policy (option: adapt diffusion policy with FM loss, or wrap HRI-EU implementation)

**Tier 2 — flow matching VLAs (small → large):**
- [ ] `configs/smolvla.yaml` — SmolVLA (smallest VLA, flow matching action expert, fast iteration)
- [ ] `configs/xvla.yaml` — X-VLA (0.9B, flow matching, soft-prompt fine-tuning)
- [ ] `configs/pi0.yaml` — pi0 via LeRobot (3B, flow matching, `--policy.type=pi0`)
- [ ] `configs/pi0fast.yaml` — pi0-FAST (flow matching + FAST tokenizer, 5x faster training)

**Tier 3 — large models:**
- [ ] `configs/pi05.yaml` — pi0.5 via LeRobot (3B, best ALOHA transfer)
- [ ] `configs/wallx.yaml` — WALL-OSS (MoE, diffusion or next-token modes)
- [ ] `configs/groot.yaml` — GR00T N1.5 (NVIDIA, requires Isaac setup)

### 1c. Flow Matching Policy implementation (non-VLA)

Two options to discuss:

**Option A: Patch LeRobot's diffusion policy**
- Fork/modify `lerobot/policies/diffusion/` to swap DDPM for conditional flow matching
- Same 1D U-Net architecture, replace noise schedule with linear interpolation + vector field regression
- Minimal code change (~50 lines in the noise/loss logic)
- Pro: stays in LeRobot ecosystem, same config system
- Con: maintaining a fork

**Option B: Wrap HRI-EU/flow_matching as a LeRobot policy**
- Implement `lobe/policies/flow_matching_policy.py` conforming to LeRobot's policy interface
- Load pretrained weights or train from scratch
- Pro: proven implementation, published results
- Con: more integration work, different codebase style

**Option C: Just use SmolVLA / pi0 with `train_expert_only=true`**
- These ARE flow matching policies. The VLM is just an encoder.
- `train_expert_only=true` freezes VLM, trains only the flow matching action head
- Pro: zero custom code, best pretrained representations
- Con: larger model, slower inference than pure flow matching

### 1d. Training launcher

- [ ] `scripts/train.sh` — wrapper around `lerobot-train` with sensible defaults
  - Sets output dir, enables wandb logging, supports resume
- [ ] Update `.claude/commands/train.md`

### 1e. Evaluation

- [ ] `scripts/eval_checkpoint.py` — run `lerobot-eval` or log action statistics

### 1f. Serving (LeRobot inference)

- [ ] `scripts/serve_policy.py` — WebSocket server implementing limb's policy_server_spec
  - Load any LeRobot checkpoint (diffusion, flow matching, VLA — all the same interface)
  - Expose metadata (image size, action dims) on connect
  - Accept obs → return action chunks via msgpack

### Milestones

1. **Diffusion Policy on robot** — validates full data→train→serve→deploy loop
2. **Flow matching policy on robot** — either custom or via SmolVLA/pi0
3. **X-VLA on robot** — first VLA deployment

---

## Phase 2: OpenPI Integration

**Goal**: Fine-tune pi0/pi0.5 using OpenPI's stack for best quality, serve via OpenPI.

### 2a. OpenPI setup

- [ ] Add openpi as a git submodule or document install steps
- [ ] Pin compatible JAX + CUDA versions (or use PyTorch path — OpenPI now supports both)

### 2b. YAM transform class

- [ ] `scripts/openpi/yam_transform.py` — custom OpenPI transform
  - `YamInputs`: map YAM obs keys (left/right joint_pos, wrist camera RGB) → pi0 input format
  - `YamOutputs`: map pi0 action output → YAM 14-dim actions
  - Handle image resizing + action (un)normalization

### 2c. OpenPI training config

- [ ] `configs/openpi/pi0_yam.py` — data + train config for YAM
- [ ] `configs/openpi/pi05_yam.py` — same for pi0.5

### 2d. OpenPI training scripts

- [ ] `scripts/openpi/compute_norm_stats.sh`
- [ ] `scripts/openpi/train.sh`

### 2e. OpenPI serving

- [ ] `scripts/openpi/serve.sh` — limb already has OpenPI client, so this just works

### Milestone

pi0.5 fine-tuned via OpenPI, served, running on robot via limb.

---

## Phase 3: Unified Serving Layer

**Goal**: Single server wrapping any backend (LeRobot, OpenPI, future).

### 3a. Server architecture

- [ ] `lobe/serve/server.py` — WebSocket + msgpack server (limb's policy_server_spec)
- [ ] `lobe/serve/base.py` — `PolicyBackend` protocol:
  ```python
  class PolicyBackend(Protocol):
      def load(self, checkpoint_path: str) -> None: ...
      def metadata(self) -> dict: ...
      def predict(self, obs: dict) -> np.ndarray: ...
  ```

### 3b. Backend implementations

- [ ] `lobe/serve/backends/lerobot_backend.py` — wraps any LeRobot policy
- [ ] `lobe/serve/backends/openpi_backend.py` — wraps OpenPI inference

### 3c. Config-driven launch

- [ ] YAML config: backend type, checkpoint path, device, host/port
- [ ] `scripts/serve.py` — unified entry point

### Milestone

One command to serve any model. limb connects with same `PolicyClient` regardless of backend.

---

## Roadmap

```
Phase 1 (LeRobot-native)
│
├─ 1a. Dataset pipeline                    ← START HERE
├─ 1b. Diffusion Policy config             ← first training run
├─ 1c. Flow Matching Policy (decide A/B/C)
├─ 1d. Training launcher
├─ 1f. WebSocket policy server
│      ↓
│   [Milestone 1: Diffusion Policy on robot]
│   [Milestone 2: Flow Matching on robot]
│
├─ 1b. SmolVLA / X-VLA configs             ← VLA fine-tuning
├─ 1b. pi0 / pi0-FAST configs
│      ↓
│   [Milestone 3: X-VLA on robot]
│
├─ 1b. pi0.5 / WALL-OSS / GR00T configs
│      ↓
│   [Milestone 4: pi0.5 (PyTorch) on robot]
│   [Milestone 5: WALL-OSS on robot]
│   [Milestone 6: GR00T on robot]
│
Phase 2 (OpenPI)
│
├─ 2a. OpenPI setup
├─ 2b. YAM transform class
├─ 2c-d. Training configs + scripts
├─ 2e. Serving
│      ↓
│   [Milestone 7: pi0.5 via OpenPI (JAX) on robot]
│
Phase 3 (Unified serving)
│
├─ 3a-c. Server + backends + config
│      ↓
│   [Milestone 8: any model, one serve command]
```

---

## Hardware

- **Training**: Multi-H100 on GCP (all models feasible, including large VLAs and multi-GPU training)
- **Deployment**: RTX 5090 (sufficient for inference on all models, including pi0.5 3B in bfloat16)

## Serving Strategy

Both PyTorch and JAX paths for pi0/pi0.5:
- **LeRobot (PyTorch)**: native `lerobot-eval` or custom WebSocket server — simpler, single framework
- **OpenPI (JAX)**: Physical Intelligence's optimized serving — potentially faster inference, JAX XLA compilation
- Phase 3 unified server wraps both, limb doesn't need to know which backend is running

## Open Questions

1. **Flow matching non-VLA**: Option A (patch diffusion), B (port HRI-EU), or C (just use SmolVLA/pi0)? Option C is fastest to get running. Option A gives the cleanest comparison to diffusion policy.
2. **Dataset size** — how many episodes we have or plan to collect?
3. **GR00T N1.5** — requires NVIDIA Isaac setup. Confirm we want to invest in this dependency.
4. **VITA / Streaming Flow Policy** — worth investigating later, or out of scope?
