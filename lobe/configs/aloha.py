"""ALOHA sim presets — bimanual manipulation benchmarks."""

from lobe.configs.base import (
    EnvConfig,
    FMPolicyConfig,
    LoggingConfig,
    PerformanceConfig,
    TrainConfig,
    TrainPipelineConfig,
)

PRESETS = {
    "aloha-fm": (
        "ALOHA sim — Flow Matching (horizon=16, MEAN_STD norm)",
        TrainPipelineConfig(
            env=EnvConfig(
                name="aloha",
                dataset_repo_id="lerobot/aloha_sim_transfer_cube_human_image",
                horizon=16,
                n_action_steps=8,
                n_obs_steps=1,
            ),
            policy=FMPolicyConfig(
                backbone="transformer",
                vision_encoder="global_pool",
                num_inference_steps=6,
                resize_shape="224,224",
            ),
            train=TrainConfig(steps=25000, batch_size=256),
            performance=PerformanceConfig(num_workers=0),
            logging=LoggingConfig(output_dir="checkpoints/aloha_fm", save_every=5000),
        ),
    ),
    "aloha-fm-fast": (
        "ALOHA sim FM — pre-cached .pt dataset (zero data overhead)",
        TrainPipelineConfig(
            env=EnvConfig(
                name="aloha",
                dataset_repo_id="datasets/aloha_sim_transfer_cube_human_image_224.pt",
                horizon=16,
                n_action_steps=8,
                n_obs_steps=1,
            ),
            policy=FMPolicyConfig(backbone="transformer", vision_encoder="global_pool", num_inference_steps=6),
            train=TrainConfig(steps=25000, batch_size=256),
            performance=PerformanceConfig(num_workers=0),
            logging=LoggingConfig(output_dir="checkpoints/aloha_fm", save_every=5000),
        ),
    ),
    "aloha-fm-delta": (
        "ALOHA sim FM — delta actions (predict motion, not position)",
        TrainPipelineConfig(
            env=EnvConfig(
                name="aloha",
                dataset_repo_id="datasets/aloha_sim_transfer_cube_human_image_224.pt",
                horizon=16,
                n_action_steps=8,
                n_obs_steps=1,
            ),
            policy=FMPolicyConfig(
                backbone="transformer",
                vision_encoder="global_pool",
                num_inference_steps=6,
                delta_actions=True,
            ),
            train=TrainConfig(steps=25000, batch_size=256),
            performance=PerformanceConfig(num_workers=0),
            logging=LoggingConfig(output_dir="checkpoints/aloha_fm_delta", save_every=5000),
        ),
    ),
}
