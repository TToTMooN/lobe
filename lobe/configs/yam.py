"""YAM bimanual training presets.

Each preset is a dataclass that produces a list of CLI flags for
`scripts/_lobe_train_entry.py` via `to_launch_args()`.

The actual training runs through `lerobot-train.main()` which consumes the
draccus-style flags — the dataclass is just a named, typed container so
presets can be diffed and version-controlled.

Phase 6 will add `scripts/train_yam.py` that:

    import accelerate.commands.launch as al
    args = PRESETS["yam_grey_cube_diffusion"].to_launch_args()
    subprocess.run(["accelerate-launch", "--num_processes=8",
                    "--mixed_precision=bf16",
                    "scripts/_lobe_train_entry.py", *args])

Until then, copy-paste the launch command in docs/workflows/yam_finetune.md.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class YAMBaseConfig:
    """Shared config for all YAM backbones."""

    dataset_repo_id: str = "ttotmoon/yam_pick_up_grey_cube"
    dataset_root: str | None = None
    output_dir: str = "checkpoints/yam"
    job_name: str = "yam"
    batch_size: int = 8  # per-GPU; effective batch = batch_size * num_processes
    num_workers: int = 4
    steps: int = 50_000
    save_freq: int = 10_000
    log_freq: int = 100
    eval_freq: int = 0  # no in-training sim eval — YAM eval is replay/on-robot

    def base_args(self) -> list[str]:
        args = [
            f"--dataset.repo_id={self.dataset_repo_id}",
            f"--batch_size={self.batch_size}",
            f"--num_workers={self.num_workers}",
            f"--steps={self.steps}",
            f"--save_freq={self.save_freq}",
            f"--log_freq={self.log_freq}",
            f"--eval_freq={self.eval_freq}",
            f"--output_dir={self.output_dir}",
            f"--job_name={self.job_name}",
        ]
        if self.dataset_root is not None:
            args.append(f"--dataset.root={self.dataset_root}")
        return args


@dataclass
class YAMDiffusionConfig(YAMBaseConfig):
    """Diffusion Policy on YAM — 3 cameras, 14-D joint-space state/action."""

    output_dir: str = "checkpoints/yam-grey-cube-dp-v0"
    job_name: str = "yam-grey-cube-dp-v0"

    # DP architecture — lerobot defaults are good; keep explicit for auditability.
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str = "ResNet18_Weights.IMAGENET1K_V1"
    # GroupNorm conversion corrupts pretrained BN stats, so keep BN when using
    # pretrained ImageNet weights. Lerobot's `use_group_norm=True` default
    # assumes random-init, hence the conflict.
    use_group_norm: bool = False
    # YAM cameras are 480x640. Half-resolution preserves aspect ratio.
    resize_shape: tuple[int, int] = (240, 320)
    crop_ratio: float = 1.0

    # Optimizer — DP's default lerobot recipe. Bypasses the LR preset-overwrite
    # trap by setting via --policy.optimizer_* (see xvla_finetune.md for why).
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-6
    scheduler_warmup_steps: int = 500

    def to_launch_args(self) -> list[str]:
        return [
            *self.base_args(),
            "--policy.type=diffusion",
            f"--policy.horizon={self.horizon}",
            f"--policy.n_obs_steps={self.n_obs_steps}",
            f"--policy.n_action_steps={self.n_action_steps}",
            f"--policy.vision_backbone={self.vision_backbone}",
            f"--policy.pretrained_backbone_weights={self.pretrained_backbone_weights}",
            f"--policy.resize_shape=[{self.resize_shape[0]},{self.resize_shape[1]}]",
            f"--policy.crop_ratio={self.crop_ratio}",
            f"--policy.use_group_norm={str(self.use_group_norm).lower()}",
            f"--policy.optimizer_lr={self.optimizer_lr}",
            f"--policy.optimizer_weight_decay={self.optimizer_weight_decay}",
            f"--policy.scheduler_warmup_steps={self.scheduler_warmup_steps}",
            "--policy.push_to_hub=false",
        ]


@dataclass
class YAMFlowMatchingConfig(YAMBaseConfig):
    """Flow Matching on YAM — same architecture as DP, ODE head instead of DDPM."""

    output_dir: str = "checkpoints/yam-grey-cube-fm-v0"
    job_name: str = "yam-grey-cube-fm-v0"

    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    resize_shape: tuple[int, int] = (240, 320)
    crop_ratio: float = 1.0
    # Match DP's UNet backbone and channel dims for a fair comparison.
    backbone: str = "unet1d"
    down_dims: str = "512,1024,2048"
    num_inference_steps: int = 10
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-6
    scheduler_warmup_steps: int = 500

    def to_launch_args(self) -> list[str]:
        return [
            *self.base_args(),
            "--policy.type=flow_matching",
            f"--policy.horizon={self.horizon}",
            f"--policy.n_obs_steps={self.n_obs_steps}",
            f"--policy.n_action_steps={self.n_action_steps}",
            f"--policy.vision_backbone={self.vision_backbone}",
            f"--policy.pretrained_backbone_weights={self.pretrained_backbone_weights}",
            f"--policy.resize_shape=[{self.resize_shape[0]},{self.resize_shape[1]}]",
            f"--policy.crop_ratio={self.crop_ratio}",
            f"--policy.use_group_norm={str(self.use_group_norm).lower()}",
            f"--policy.backbone={self.backbone}",
            f"--policy.down_dims=[{self.down_dims}]",
            f"--policy.num_inference_steps={self.num_inference_steps}",
            f"--policy.optimizer_lr={self.optimizer_lr}",
            f"--policy.optimizer_weight_decay={self.optimizer_weight_decay}",
            f"--policy.scheduler_warmup_steps={self.scheduler_warmup_steps}",
            "--policy.push_to_hub=false",
        ]


@dataclass
class YAMXVLAConfig(YAMBaseConfig):
    """X-VLA fine-tune on YAM — V14 LIBERO recipe adapted for 14-D joint space."""

    dataset_repo_id: str = "local/yam_pick_up_grey_cube_image"
    dataset_root: str = "/home/sunlingfeng/.cache/huggingface/lerobot/local/yam_pick_up_grey_cube_image"
    output_dir: str = "checkpoints/yam-grey-cube-xvla-v0"
    job_name: str = "yam-grey-cube-xvla-v0"
    batch_size: int = 16
    steps: int = 20_000
    save_freq: int = 5_000

    policy_path: str = "/mnt/localssd/sunlingfeng/checkpoints/xvla-pt-yam14"
    chunk_size: int = 30
    n_action_steps: int = 30
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 20_000
    scheduler_decay_lr: float = 1e-4

    def to_launch_args(self) -> list[str]:
        return [
            *self.base_args(),
            "--dataset.image_transforms.enable=true",
            f"--policy.path={self.policy_path}",
            "--policy.action_mode=auto",
            f"--policy.chunk_size={self.chunk_size}",
            f"--policy.n_action_steps={self.n_action_steps}",
            "--policy.dtype=bfloat16",
            "--policy.use_amp=false",
            "--policy.push_to_hub=false",
            f"--policy.optimizer_lr={self.optimizer_lr}",
            f"--policy.optimizer_weight_decay={self.optimizer_weight_decay}",
            f"--policy.optimizer_grad_clip_norm={self.grad_clip_norm}",
            f"--policy.scheduler_warmup_steps={self.scheduler_warmup_steps}",
            f"--policy.scheduler_decay_steps={self.scheduler_decay_steps}",
            f"--policy.scheduler_decay_lr={self.scheduler_decay_lr}",
        ]


@dataclass
class YAMSmolVLAConfig(YAMBaseConfig):
    """SmolVLA fine-tune on YAM — frozen VLM encoder, trains action expert only."""

    dataset_repo_id: str = "local/yam_pick_up_grey_cube_image"
    dataset_root: str = "/home/sunlingfeng/.cache/huggingface/lerobot/local/yam_pick_up_grey_cube_image"
    output_dir: str = "checkpoints/yam-grey-cube-smolvla-v0"
    job_name: str = "yam-grey-cube-smolvla-v0"
    batch_size: int = 32
    steps: int = 20_000
    save_freq: int = 5_000

    policy_path: str = "lerobot/smolvla_base"
    optimizer_lr: float = 1e-5
    scheduler_warmup_steps: int = 500

    def to_launch_args(self) -> list[str]:
        return [
            *self.base_args(),
            "--dataset.image_transforms.enable=true",
            f"--policy.path={self.policy_path}",
            "--policy.push_to_hub=false",
            f"--policy.optimizer_lr={self.optimizer_lr}",
            f"--policy.scheduler_warmup_steps={self.scheduler_warmup_steps}",
            # SmolVLA defaults: train_expert_only=True, freeze_vision_encoder=True
            # max_state_dim=32, max_action_dim=32 (auto-pads 14-D YAM state/action)
        ]


PRESETS: dict[str, YAMBaseConfig] = {
    "yam_grey_cube_diffusion": YAMDiffusionConfig(),
    "yam_grey_cube_flow_matching": YAMFlowMatchingConfig(),
    "yam_grey_cube_xvla": YAMXVLAConfig(),
    "yam_grey_cube_smolvla": YAMSmolVLAConfig(),
}
