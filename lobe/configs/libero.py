"""LIBERO presets — 7-DOF manipulation benchmarks (130 tasks, 5 suites).

Published results on HuggingFaceVLA/libero dataset:
  Diffusion Policy: 72.4% avg (SmolVLA paper)
  SmolVLA (0.45B):  87.3% avg (batch=64, 100k steps)
  pi0.5 (3B):       97.5% avg (batch=32x8GPU, 6k steps)
  X-VLA (0.9B):     98.1% avg
"""

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
        "LIBERO — Flow Matching (7-DOF, 2 cameras, 224x224)",
        TrainPipelineConfig(
            env=EnvConfig(
                name="libero",
                dataset_repo_id="HuggingFaceVLA/libero",
            ),
            policy=FMPolicyConfig(
                backbone="transformer",
                vision_encoder="global_pool",
                num_inference_steps=10,
                resize_shape="224,224",
            ),
            train=TrainConfig(steps=50000, batch_size=64, lr=1e-4),
            performance=PerformanceConfig(num_workers=8),
            logging=LoggingConfig(output_dir="checkpoints/libero_fm", save_every=10000),
        ),
    ),
}
