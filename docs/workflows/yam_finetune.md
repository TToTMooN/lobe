# Fine-tune on a YAM (bimanual limb) dataset

LOBE's YAM pipeline takes a LeRobot v3.0 dataset collected by
[limb](https://github.com/TToTMooN/limb) and trains a policy that can be
(a) replay-evaluated offline and (b) deployed on the robot via
`lobe-serve`. Reference dataset: [`ttotmoon/yam_pick_up_grey_cube`](https://huggingface.co/datasets/ttotmoon/yam_pick_up_grey_cube).

The four backbones roll out in phases (see
[`docs/milestones/yam_multibackbone.md`](../milestones/yam_multibackbone.md)).
**This doc currently covers Phase 1: Diffusion Policy**. FM, SmolVLA, X-VLA
sections will land as the respective phases complete.

## Prerequisite: validate the dataset

```bash
uv run python scripts/validate_yam_dataset.py ttotmoon/yam_pick_up_grey_cube
```

Expected output: `overall: PASS` with 10 episodes / 10958 frames / 30 fps /
3 cameras. If the script reports a missing `v3.0` tag on HF, rerun with
`--create-tag` (requires write access to the HF repo).

## Phase 1: Diffusion Policy

Preset: `yam_grey_cube_diffusion` in `lobe/configs/yam.py`. ImageNet-pretrained
ResNet18 visual encoder shared across the 3 cameras, UNet1d temporal backbone,
horizon 16 / n_action_steps 8 / n_obs_steps 2. Images are resized to 240x320
(half the native 480x640, aspect ratio preserved) and not cropped — the YAM
camera framing is already tight.

### Launch (8x H100)

```bash
.venv/bin/accelerate-launch --num_processes=8 --mixed_precision=bf16 \
  scripts/_lobe_train_entry.py \
  --dataset.repo_id=ttotmoon/yam_pick_up_grey_cube \
  --policy.type=diffusion \
  --policy.horizon=16 \
  --policy.n_obs_steps=2 \
  --policy.n_action_steps=8 \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --policy.resize_shape=[240,320] \
  --policy.crop_ratio=1.0 \
  --policy.use_group_norm=false \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_weight_decay=1e-6 \
  --policy.scheduler_warmup_steps=500 \
  --policy.push_to_hub=false \
  --batch_size=8 \
  --num_workers=4 \
  --steps=50000 \
  --save_freq=10000 \
  --log_freq=100 \
  --eval_freq=0 \
  --output_dir=checkpoints/yam-grey-cube-dp-v0 \
  --job_name=yam-grey-cube-dp-v0
```

`batch_size=8 x 8 GPUs = 64` effective, matching the design-doc recipe.
`eval_freq=0` disables in-training sim eval — YAM has no sim; the eval path is
replay MSE (Phase 3) and on-robot (Phase 6).

### Why `--policy.optimizer_*` and not `--optimizer.*`

Same preset-overwrite trap documented in the X-VLA recipe — see the "Key points"
section of [`xvla_finetune.md`](./xvla_finetune.md) for the full root-cause write-up.
`TrainPipelineConfig.validate()` calls `policy.get_optimizer_preset()` when
`use_policy_training_preset=True` (the default), which silently overwrites any
`--optimizer.*` flags with the DP config's `optimizer_lr=1e-4` preset. Pass
optimizer settings via `--policy.optimizer_*` to actually take effect.

### Expected metrics

- Wall time: ~2h on 8x H100 (batch 64, 50k steps, bf16).
- Training loss monotonically decreasing; no stable floor known yet — Phase 1
  is the first DP run on real YAM data, so its final loss + Phase 3 replay MSE
  jointly define the baseline that later backbones (FM, SmolVLA, X-VLA) will
  be compared against.

## Phase 2 (coming): Flow Matching

Identical to the DP command above but with `--policy.type=flow_matching`. Head-
to-head comparison on the same encoder and schedule.

## Phases 3-6 (coming)

See [`docs/milestones/yam_multibackbone.md`](../milestones/yam_multibackbone.md)
for the full plan.
