"""Training configuration — nested dataclasses following LeRobot convention.

Usage with tyro CLI:
    uv run python scripts/train_pusht.py --env.dataset-repo-id lerobot/pusht_image --steps 50000
    uv run python scripts/train_pusht.py --policy-type diffusion --wandb.enable
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Environment / dataset configuration."""

    name: str = "pusht"  # pusht | yam (see lobe/envs/)
    dataset_repo_id: str = "lerobot/pusht_image"
    fps: float = 10.0
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    max_steps: int = 300  # max steps per eval rollout


@dataclass
class PolicyConfig:
    """Policy architecture configuration."""

    type: str = "flow_matching"  # flow_matching | diffusion
    num_inference_steps: int = 10
    ode_solver: str = "euler"  # euler | midpoint
    # FM-specific
    down_dims: str = "256,512,1024"
    embed_dim: int = 256
    crop_ratio: float = 0.8


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
    gpu_preload: bool = False  # preload entire dataset to GPU memory


@dataclass
class LoggingConfig:
    """Logging and checkpointing."""

    log_every: int = 100
    save_every: int = 10000
    eval_every: int = 0  # 0 = disable eval during training
    eval_rollouts: int = 10
    output_dir: str = "checkpoints/pusht"


@dataclass
class WandbConfig:
    """Weights & Biases integration."""

    enable: bool = True
    project: str = "lobe-train"
    name: str = ""  # auto-generated if empty
    group: str = ""  # group related runs (e.g. "pusht-fm-sweep")
    tags: str = ""  # comma-separated tags


@dataclass
class TrainPipelineConfig:
    """Full training pipeline configuration — pass to tyro.cli()."""

    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    device: str = "cuda"
