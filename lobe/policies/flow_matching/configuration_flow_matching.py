"""Configuration for Flow Matching Policy.

Identical architecture to DiffusionPolicy (same 1D U-Net, vision encoder, FiLM conditioning),
but replaces the DDPM/DDIM noise process with conditional flow matching:
- Linear interpolation instead of beta noise schedule
- Velocity field prediction instead of noise prediction
- Euler ODE integration instead of reverse diffusion

Reference: "Flow Matching for Generative Modeling" (Lipman et al., 2023)
Cherry-picked from HRI-EU/flow_matching: optimal transport coupling option.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("flow_matching")
@dataclass
class FlowMatchingConfig(PreTrainedConfig):
    """Configuration class for FlowMatchingPolicy.

    Uses the same U-Net architecture and vision encoder as DiffusionPolicy.
    The only differences are in the noise process (flow matching vs DDPM) and inference
    (Euler ODE vs reverse diffusion).

    Args:
        n_obs_steps: Number of observation steps to condition on.
        horizon: Action prediction horizon.
        n_action_steps: Number of action steps to execute per policy call.
        sigma: Noise level for conditional flow matching. 0.0 = deterministic OT path.
        num_inference_steps: Number of Euler steps at inference time. 1 = single-step.
        use_optimal_transport: Use minibatch optimal transport coupling (from torchcfm).
            Improves sample diversity by finding better noise-to-data pairings.
        vision_backbone: Torchvision ResNet backbone name.
        down_dims: U-Net channel dimensions per downsampling stage.
        diffusion_step_embed_dim: Timestep embedding dimension (reused name for U-Net compat).
    """

    # Inputs / output structure (same as diffusion)
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    drop_n_last_frames: int = 7

    # Vision backbone
    vision_backbone: str = "resnet18"
    # spatial_softmax (64-d) | global_pool (512-d) | dinov2_small (384-d) | dinov2_base (768-d) | siglip_base (768-d)
    vision_encoder: str = "spatial_softmax"
    vision_encoder_frozen: bool = True  # freeze pretrained vision encoder (DINOv2/SigLIP)
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 0.8
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # Backbone: "unet" (DiffusionConditionalUnet1d) or "transformer" (DiT-style AdaLN)
    backbone: str = "transformer"

    # U-Net params (used when backbone="unet")
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 256
    use_film_scale_modulation: bool = True

    # Transformer params (used when backbone="transformer")
    transformer_d_model: int = 256
    transformer_n_heads: int = 4
    transformer_n_layers: int = 4
    transformer_dropout: float = 0.1

    # Flow matching specific
    sigma: float = 0.0
    num_inference_steps: int = 10
    ode_solver: str = "euler"  # euler (pi0 standard, fast) | midpoint (2nd-order, more accurate)
    delta_actions: bool = False  # predict action[t] - action[0] per chunk (better for position control)
    use_optimal_transport: bool = False
    clip_sample: bool = False
    clip_sample_range: float = 1.0

    # Inference / serving
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # Loss
    do_mask_loss_for_padding: bool = False

    # Training presets (same as diffusion)
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        # vision_backbone is only used when vision_encoder is spatial_softmax or global_pool
        if self.vision_encoder in ("spatial_softmax", "global_pool"):
            if not self.vision_backbone.startswith("resnet"):
                raise ValueError(
                    f"`vision_backbone` must be a ResNet variant when using {self.vision_encoder}. "
                    f"Got {self.vision_backbone}."
                )

        if self.resize_shape is not None and (len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)):
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

        if self.backbone == "unet":
            downsampling_factor = 2 ** len(self.down_dims)
            if self.horizon % downsampling_factor != 0:
                raise ValueError(
                    "The horizon should be an integer multiple of the downsampling factor (which is determined "
                    f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
                )

        if self.num_inference_steps < 1:
            raise ValueError(f"`num_inference_steps` must be >= 1. Got {self.num_inference_steps}.")

        if self.sigma < 0:
            raise ValueError(f"`sigma` must be >= 0. Got {self.sigma}.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        # Compute crop_shape from crop_ratio and image size if not explicitly set
        if self.crop_shape is None and self.crop_ratio < 1.0 and len(self.image_features) > 0:
            first_image_ft = next(iter(self.image_features.values()))
            h, w = first_image_ft.shape[1], first_image_ft.shape[2]
            self.crop_shape = (int(h * self.crop_ratio), int(w * self.crop_ratio))

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
