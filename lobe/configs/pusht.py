"""PushT presets — verified baselines for the PushT benchmark."""

from lobe.configs.base import (
    DiffusionPolicyConfig,
    EnvConfig,
    FMPolicyConfig,
    LoggingConfig,
    PerformanceConfig,
    TrainConfig,
    TrainPipelineConfig,
)

PRESETS = {
    "pusht-fm": (
        "PushT — Flow Matching (verified 60-80%)",
        TrainPipelineConfig(
            env=EnvConfig(name="pusht", dataset_repo_id="lerobot/pusht_image"),
            policy=FMPolicyConfig(backbone="transformer"),
            train=TrainConfig(steps=50000, batch_size=256),
            performance=PerformanceConfig(num_workers=4),
            logging=LoggingConfig(output_dir="checkpoints/pusht_fm", eval_every=10000),
        ),
    ),
    "pusht-diffusion": (
        "PushT — Diffusion Policy baseline",
        TrainPipelineConfig(
            env=EnvConfig(name="pusht", dataset_repo_id="lerobot/pusht_image"),
            policy=DiffusionPolicyConfig(),
            train=TrainConfig(steps=50000, batch_size=256),
            performance=PerformanceConfig(num_workers=4),
            logging=LoggingConfig(output_dir="checkpoints/pusht_diffusion", eval_every=10000),
        ),
    ),
}
