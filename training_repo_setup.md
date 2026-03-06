# Setting Up LOBE — Training Repo for limb

> **LOBE** = Learning Orchestration — Brain-to-Embodiment
>
> The motor cortex (frontal **lobe**) learns to control **limb**s.

LOBE is the companion training repo for [limb](https://github.com/TToTMooN/limb).
limb handles robot control + data collection; LOBE handles policy training + serving.

---

## Quick Start

```bash
# Create repo
mkdir lobe && cd lobe
git init

# Create pyproject.toml
cat > pyproject.toml << 'PYPROJECT'
[project]
name = "lobe"
version = "0.1.0"
description = "Policy training and serving for limb robot control"
requires-python = ">=3.11,<3.12"
dependencies = [
    "lerobot[pi,xvla,wallx]",
    "huggingface-hub",
    "wandb",
]

[project.optional-dependencies]
serve = [
    "websockets",
    "msgpack",
    "msgpack-numpy",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
PYPROJECT

# Create directory structure
mkdir -p configs scripts datasets checkpoints

# Install
uv sync
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
    serve_policy.py            # WebSocket policy server (limb connects here)
  datasets/                    # Local LeRobot v2.1 datasets (or use HF Hub)
  checkpoints/                 # Training outputs
  pyproject.toml
```

## End-to-End Workflow

### 1. Collect data (limb, on robot machine)

```bash
cd limb
uv run limb/envs/launch.py \
  --config_path configs/yam_gello_network_bimanual.yaml configs/collection.yaml
```

### 2. Convert and upload (limb, on robot machine)

```bash
uv run scripts/data/convert_to_lerobot.py \
  --input_dir recordings/pick_up_the_red_cube_20260305_143000 \
  --output_dir datasets/yam_red_cube \
  --task "pick up the red cube and place it in the bowl" \
  --success_only \
  --push_to_hub yourname/yam-red-cube
```

### 3. Train (lobe, on GPU machine)

```bash
cd lobe

# X-VLA — smallest (0.9B), fastest to train, learns soft prompts for your embodiment
uv run lerobot-train \
  --dataset.repo_id=yourname/yam-red-cube \
  --policy.path=lerobot/xvla-base \
  --policy.dtype=bfloat16 \
  --batch_size=8 \
  --steps=20000 \
  --output_dir=checkpoints/xvla-yam-red-cube

# pi0.5 — use ALOHA checkpoint (bimanual 2x6DOF+gripper, same action dims as YAM)
uv run lerobot-train \
  --dataset.repo_id=yourname/yam-red-cube \
  --policy.path=physical-intelligence/pi0.5 \
  --policy.dtype=bfloat16 \
  --steps=20000 \
  --output_dir=checkpoints/pi05-yam-red-cube

# WALL-OSS — supports both diffusion and next-token prediction modes
uv run lerobot-train \
  --dataset.repo_id=yourname/yam-red-cube \
  --policy.type=wall_x \
  --policy.path=x-square-robot/wall-oss-flow \
  --steps=20000 \
  --output_dir=checkpoints/walloss-yam-red-cube
```

### 4. Serve policy (lobe, on GPU machine or robot machine)

```bash
# Option A: OpenPI server (for pi0/pi0.5)
# See https://github.com/Physical-Intelligence/openpi
openpi serve --checkpoint checkpoints/pi05-yam-red-cube --port 8111

# Option B: WebSocket server matching limb's policy_server_spec.md
uv run python scripts/serve_policy.py \
  --checkpoint checkpoints/xvla-yam-red-cube \
  --host 0.0.0.0 --port 8000
```

### 5. Deploy (limb, on robot machine)

```bash
cd limb

# Connect to OpenPI server
uv run limb/envs/launch.py --config_path configs/yam_pi0_bimanual.yaml

# Connect to generic WebSocket server
uv run limb/envs/launch.py --config_path configs/yam_policy_bimanual.yaml
```

## Model Comparison

| Model | Params | Training Speed | Best For |
|-------|--------|---------------|----------|
| X-VLA | 0.9B | Fast (~2h on 1xA100) | Small datasets (50-200 episodes), quick iteration |
| pi0.5 | 3B | Medium (~8h on 1xA100) | Best zero-shot transfer from ALOHA, robust |
| WALL-OSS | ~3B | Medium | MoE architecture, chain-of-thought reasoning |

## Notes

- All models expect **224x224 RGB images** and **14-dim actions** (6 joints + 1 gripper per arm)
- YAM bimanual matches ALOHA's action space exactly (2x 6-DOF + gripper = 14 dims)
- pi0.5 requires a custom OpenPI transform class mapping YAM obs keys → ALOHA keys
- X-VLA is recommended for first experiments — fastest to train, smallest checkpoint
- Aim for **50-200 successful teleop episodes** for initial fine-tuning
- limb's `convert_to_lerobot.py` outputs LeRobot v2.1 format (no lerobot dependency needed)
- limb's `PolicyClient` connects to any server over WebSocket — no lerobot needed on robot side
