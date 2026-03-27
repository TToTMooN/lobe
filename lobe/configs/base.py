"""Base config dataclasses — shared across all env presets."""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Environment ──────────────────────────────────────────────────────────────


@dataclass
class EnvConfig:
    """Environment / dataset configuration."""

    name: str = "pusht"  # pusht | aloha | yam
    dataset_repo_id: str = "lerobot/pusht_image"
    fps: float = 10.0
    n_obs_steps: int = 0  # 0 = use env default
    horizon: int = 0  # 0 = use env default
    n_action_steps: int = 0  # 0 = use env default
    max_steps: int = 0  # 0 = use env default


# ── Policy (union: each type has only its own fields) ────────────────────────


@dataclass
class FMPolicyConfig:
    """Flow Matching policy — transformer or U-Net backbone."""

    backbone: str = "transformer"  # transformer | unet
    vision_encoder: str = "spatial_softmax"  # spatial_softmax (64-d) | global_pool (512-d, VITA-style)
    num_inference_steps: int = 10
    ode_solver: str = "euler"  # euler | midpoint
    delta_actions: bool = False  # predict action[t] - action[0] per chunk (better for position control)
    down_dims: str = "256,512,1024"  # U-Net channel dims (ignored for transformer)
    embed_dim: int = 256
    resize_shape: str = ""  # e.g. "224,224" for ALOHA
    crop_ratio: float = 0.8


@dataclass
class DiffusionPolicyConfig:
    """Diffusion Policy (LeRobot wrapper)."""

    num_inference_steps: int = 100


# ── Training ─────────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    steps: int = 50000
    batch_size: int = 256
    lr: float = 1e-4
    warmup_steps: int = 500
    use_cosine_schedule: bool = True
    ema_power: float = 0.75  # 0 = disable
    seed: int = 42


@dataclass
class PerformanceConfig:
    """GPU / data loading performance."""

    bf16: bool = True
    tf32: bool = True
    compile: bool = True
    compile_mode: str = "reduce-overhead"
    num_workers: int = 16
    prefetch_factor: int = 4
    gradient_accumulation: int = 1


@dataclass
class LoggingConfig:
    """Logging and checkpointing."""

    log_every: int = 100
    save_every: int = 10000
    eval_every: int = 0  # 0 = disable
    eval_rollouts: int = 10
    output_dir: str = "checkpoints"


@dataclass
class WandbConfig:
    """Weights & Biases integration."""

    enable: bool = True
    project: str = "lobe-train"
    name: str = ""  # auto-generated if empty
    group: str = ""
    tags: str = ""  # comma-separated


# ── Root config ──────────────────────────────────────────────────────────────


@dataclass
class TrainPipelineConfig:
    """Full training pipeline configuration."""

    env: EnvConfig = field(default_factory=EnvConfig)
    policy: FMPolicyConfig | DiffusionPolicyConfig = field(default_factory=FMPolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    device: str = "cuda"
