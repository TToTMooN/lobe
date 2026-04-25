# Fine-tune on a YAM (bimanual limb) dataset

LOBE's YAM pipeline takes a LeRobot v3.0 dataset collected by
[limb](https://github.com/TToTMooN/limb) and trains a policy that can be
(a) replay-evaluated offline and (b) deployed on the robot via
`lobe-serve`. Reference dataset: [`ttotmoon/yam_pick_up_grey_cube`](https://huggingface.co/datasets/ttotmoon/yam_pick_up_grey_cube).

Four backbones are supported (presets in `lobe/configs/yam.py`). All
produce 14-D joint-space actions for the YAM bimanual robot.

## Results (yam_pick_up_grey_cube, held-out episodes 8-9)

| Backbone | Params | Replay MSE | Inference (compiled) | Train time |
|----------|--------|------------|----------------------|------------|
| **FM** | 275M | **0.00155** | **18 ms** | 2.6h |
| **X-VLA** | 879M | 0.00247 | 78 ms | 2.0h |
| **DP** | 271M | 0.01068 | 35 ms | 2h* |
| **SmolVLA** | 450M (100M learnable) | 0.02785 | 24 ms | 1.0h |

*DP trained on image-format dataset. Original video-format run took 43h.

## Prerequisites

### 1. Validate the dataset

```bash
uv run python scripts/validate_yam_dataset.py ttotmoon/yam_pick_up_grey_cube
```

If `overall: FAIL`, use the fixer flags (`--create-tag`, `--rebuild-stats`).

### 2. Convert to image format (recommended)

Video-format datasets are ~20× slower to train on due to per-frame HEVC decode.

```bash
uv run python scripts/convert_yam_video_to_image.py \
    --repo_id ttotmoon/yam_pick_up_grey_cube \
    --output local/yam_pick_up_grey_cube_image \
    --resize 240 320
```

Takes ~4 min. All launch commands below use the image-format dataset.

## Diffusion Policy

Preset: `yam_grey_cube_diffusion`. ImageNet-pretrained ResNet18 (shared
across 3 cameras), UNet1d backbone, horizon 16, DDPM training / DDIM
inference.

```bash
.venv/bin/accelerate-launch --num_processes=8 --mixed_precision=bf16 \
  scripts/_lobe_train_entry.py \
  --dataset.repo_id=local/yam_pick_up_grey_cube_image \
  --dataset.root=$HOME/.cache/huggingface/lerobot/local/yam_pick_up_grey_cube_image \
  --policy.type=diffusion \
  --policy.horizon=16 --policy.n_obs_steps=2 --policy.n_action_steps=8 \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --policy.resize_shape=[240,320] --policy.crop_ratio=1.0 \
  --policy.use_group_norm=false \
  --policy.optimizer_lr=1e-4 --policy.optimizer_weight_decay=1e-6 \
  --policy.scheduler_warmup_steps=500 --policy.push_to_hub=false \
  --batch_size=8 --num_workers=4 --steps=50000 \
  --save_freq=10000 --log_freq=100 --eval_freq=0 \
  --output_dir=checkpoints/yam-grey-cube-dp-v0 --job_name=yam-grey-cube-dp-v0
```

## Flow Matching

Preset: `yam_grey_cube_flow_matching`. Same encoder as DP, UNet1d backbone
with matching dims, Euler ODE solver (5 steps at inference).

```bash
.venv/bin/accelerate-launch --num_processes=8 --mixed_precision=bf16 \
  scripts/_lobe_train_entry.py \
  --dataset.repo_id=local/yam_pick_up_grey_cube_image \
  --dataset.root=$HOME/.cache/huggingface/lerobot/local/yam_pick_up_grey_cube_image \
  --policy.type=flow_matching \
  --policy.horizon=16 --policy.n_obs_steps=2 --policy.n_action_steps=8 \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --policy.resize_shape=[240,320] --policy.crop_ratio=1.0 \
  --policy.use_group_norm=false \
  --policy.backbone=unet1d --policy.down_dims=[512,1024,2048] \
  --policy.num_inference_steps=10 \
  --policy.optimizer_lr=1e-4 --policy.optimizer_weight_decay=1e-6 \
  --policy.scheduler_warmup_steps=500 --policy.push_to_hub=false \
  --batch_size=9 --num_workers=4 --steps=50000 \
  --save_freq=10000 --log_freq=100 --eval_freq=0 \
  --output_dir=checkpoints/yam-grey-cube-fm-v1 --job_name=yam-grey-cube-fm-v1
```

## X-VLA

Preset: `yam_grey_cube_xvla`. Fine-tunes from `2toINF/X-VLA-Pt` (0.9B).
`action_mode=auto` auto-pads 14-D actions to 20-D internally. Camera
names must be remapped to X-VLA's expected `image/image2/image3`.

```bash
.venv/bin/accelerate-launch --num_processes=8 --mixed_precision=bf16 \
  scripts/_lobe_train_entry.py \
  --dataset.repo_id=local/yam_pick_up_grey_cube_image \
  --dataset.root=$HOME/.cache/huggingface/lerobot/local/yam_pick_up_grey_cube_image \
  --dataset.image_transforms.enable=true \
  --rename_map='{"observation.images.head_camera": "observation.images.image", "observation.images.left_wrist_camera": "observation.images.image2", "observation.images.right_wrist_camera": "observation.images.image3"}' \
  --policy.path=/mnt/localssd/sunlingfeng/checkpoints/xvla-pt-yam14 \
  --policy.action_mode=auto \
  --policy.chunk_size=30 --policy.n_action_steps=30 \
  --policy.dtype=bfloat16 --policy.use_amp=false --policy.push_to_hub=false \
  --policy.optimizer_lr=1e-4 --policy.optimizer_weight_decay=0.01 \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=20000 --policy.scheduler_decay_lr=1e-4 \
  --batch_size=16 --num_workers=4 --steps=20000 \
  --save_freq=5000 --log_freq=100 --eval_freq=0 \
  --output_dir=checkpoints/yam-grey-cube-xvla-v0 --job_name=yam-grey-cube-xvla-v0
```

The `xvla-pt-yam14` checkpoint is a copy of `xvla-pt-v8` with
`output_features.action.shape=[14]`. See
[`xvla_finetune.md`](./xvla_finetune.md) for the V14 recipe details.

## SmolVLA

Preset: `yam_grey_cube_smolvla`. Fine-tunes from `lerobot/smolvla_base`
(450M, 100M learnable). Frozen VLM encoder, trains action expert only.
14-D action auto-padded to 32 internally. Camera names → `camera1/2/3`.

```bash
.venv/bin/accelerate-launch --num_processes=4 --mixed_precision=bf16 \
  scripts/_lobe_train_entry.py \
  --dataset.repo_id=local/yam_pick_up_grey_cube_image \
  --dataset.root=$HOME/.cache/huggingface/lerobot/local/yam_pick_up_grey_cube_image \
  --dataset.image_transforms.enable=true \
  --rename_map='{"observation.images.head_camera": "observation.images.camera1", "observation.images.left_wrist_camera": "observation.images.camera2", "observation.images.right_wrist_camera": "observation.images.camera3"}' \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=1e-5 --policy.scheduler_warmup_steps=500 \
  --batch_size=8 --num_workers=4 --steps=20000 \
  --save_freq=5000 --log_freq=100 --eval_freq=0 \
  --output_dir=checkpoints/yam-grey-cube-smolvla-v0 --job_name=yam-grey-cube-smolvla-v0
```

## Evaluation

### Replay MSE (offline, no robot needed)

```bash
uv run python scripts/eval_replay.py \
    --policy.path=checkpoints/yam-grey-cube-fm-v1/checkpoints/050000/pretrained_model \
    --dataset.repo_id=ttotmoon/yam_pick_up_grey_cube \
    --eval_episodes 8 9

# For X-VLA/SmolVLA, add the camera rename map:
uv run python scripts/eval_replay.py \
    --policy.path=checkpoints/yam-grey-cube-xvla-v0/checkpoints/020000/pretrained_model \
    --dataset.repo_id=ttotmoon/yam_pick_up_grey_cube \
    --eval_episodes 8 9 \
    --rename_map='{"observation.images.head_camera": "observation.images.image", ...}'
```

### Serving test (verify action shape)

```bash
uv run python scripts/test_serve_all.py
```

See [`serving.md`](./serving.md) for deployment and speed optimization.

## Notes

- **`--policy.use_group_norm=false`** is required for DP/FM when using
  pretrained ImageNet weights (GroupNorm conversion corrupts BN stats).
- **`--policy.optimizer_*`** not `--optimizer.*` — lerobot's
  `TrainPipelineConfig.validate()` silently overwrites top-level optimizer
  flags with policy presets.
- **Image dataset is ~20× faster** than video for training. Always convert
  before launching.
