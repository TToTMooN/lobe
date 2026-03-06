# LOBE

**Learning Orchestration — Brain-to-Embodiment**

> The motor cortex (frontal **lobe**) learns to control **limb**s.

Policy training and serving companion for [limb](https://github.com/TToTMooN/limb).
limb handles robot control + data collection; LOBE handles policy training + serving.

## Supported Models

| Model | Params | Training Speed | Best For |
|-------|--------|---------------|----------|
| **Flow Matching** | 262M | Fast | Drop-in upgrade over Diffusion Policy — 1-step inference |
| **Diffusion Policy** | 262M | Fast | Strong baseline, well-tested |
| **X-VLA** | 0.9B | ~2h on 1xA100 | Small datasets, quick iteration |
| **pi0.5** | 3B | ~8h on 1xA100 | Zero-shot transfer from ALOHA |
| **WALL-OSS** | ~3B | Medium | MoE, chain-of-thought reasoning |

All VLA models expect **224x224 RGB** images and **14-dim actions** (2x 6-DOF + gripper), matching YAM bimanual.

## Setup

```bash
git clone https://github.com/TToTMooN/lobe.git && cd lobe
uv sync
```

### GPU Compatibility

Different GPUs require different PyTorch + CUDA builds:

| GPU | CUDA Arch | PyTorch CUDA | Notes |
|-----|-----------|-------------|-------|
| **H100** | sm_90 | `cu121` or `cu124` | Training GPU — standard PyTorch pip works |
| **RTX 5090** | sm_120 (Blackwell) | `cu128` (nightly) | Deployment GPU — requires PyTorch nightly or >=2.8 with CUDA 12.8 |
| **A100** | sm_80 | `cu118` or `cu121` | Standard PyTorch pip works |

If you see `NVIDIA GeForce RTX 5090 ... is not compatible with the current PyTorch installation`, install the nightly:

```bash
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

For H100/A100, standard PyTorch works out of the box.

## Quick Start

### 1. Collect data (limb, on robot machine)

```bash
cd limb
uv run limb/envs/launch.py \
  --config_path configs/yam_gello_network_bimanual.yaml configs/collection.yaml
```

### 2. Convert to LeRobot format (limb, on robot machine)

```bash
uv run scripts/data/convert_to_lerobot.py \
  --input_dir recordings/pick_up_the_red_cube_20260305 \
  --output_dir datasets/yam_red_cube \
  --task "pick up the red cube and place it in the bowl" \
  --success_only \
  --push_to_hub yourname/yam-red-cube
```

### 3. Train (lobe, on GPU machine)

```bash
cd lobe

# X-VLA — recommended for first experiments
uv run lerobot-train \
  --dataset.repo_id=yourname/yam-red-cube \
  --policy.path=lerobot/xvla-base \
  --policy.dtype=bfloat16 \
  --batch_size=8 \
  --steps=20000 \
  --output_dir=checkpoints/xvla-yam-red-cube
```

### 4. Serve (lobe, on GPU machine)

```bash
# OpenPI server (pi0/pi0.5)
openpi serve --checkpoint checkpoints/pi05-yam-red-cube --port 8111

# WebSocket server (X-VLA/WALL-OSS)
uv run python scripts/serve_policy.py \
  --checkpoint checkpoints/xvla-yam-red-cube \
  --host 0.0.0.0 --port 8000
```

### 5. Deploy (limb, on robot machine)

```bash
cd limb
uv run limb/envs/launch.py --config_path configs/yam_pi0_bimanual.yaml
```

## Project Structure

```
lobe/
  configs/
    train_xvla.yaml           # X-VLA fine-tuning config
    train_pi0.yaml             # pi0/pi0.5 fine-tuning config
    train_walloss.yaml         # WALL-OSS fine-tuning config
  scripts/
    train.sh                   # Training launcher
    serve_policy.py            # WebSocket policy server
  datasets/                    # Local LeRobot v2.1 datasets
  checkpoints/                 # Training outputs
  pyproject.toml
```

## Notes

- Aim for **50-200 successful teleop episodes** for initial fine-tuning
- pi0.5 requires a custom OpenPI transform class mapping YAM obs keys -> ALOHA keys
- X-VLA is recommended for first experiments — fastest to train, smallest checkpoint
- limb's `convert_to_lerobot.py` outputs LeRobot v2.1 format (no lerobot dependency needed)
- limb's `PolicyClient` connects to any server over WebSocket — no lerobot needed on robot side
