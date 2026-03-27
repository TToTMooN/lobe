"""LIBERO presets — 7-DOF manipulation benchmarks (130 tasks, 5 suites)."""

from lobe.configs.base import (
    EnvConfig,
    FMPolicyConfig,
    LoggingConfig,
    PerformanceConfig,
    TrainConfig,
    TrainPipelineConfig,
)

PRESETS = {
    "libero-fm": (
        "LIBERO-10 — Flow Matching (7-DOF, 2 cameras, 256x256)",
        TrainPipelineConfig(
            env=EnvConfig(
                name="libero",
                dataset_repo_id="lerobot/libero_10_image",
            ),
            policy=FMPolicyConfig(
                backbone="transformer",
                vision_encoder="global_pool",
                num_inference_steps=10,
                resize_shape="224,224",
            ),
            train=TrainConfig(steps=50000, batch_size=64, lr=1e-4),
            performance=PerformanceConfig(num_workers=4),
            logging=LoggingConfig(output_dir="checkpoints/libero_fm", save_every=10000),
        ),
    ),
}
