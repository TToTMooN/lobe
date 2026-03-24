"""Flow Matching Policy — drop-in replacement for DiffusionPolicy.

Same architecture (1D conditional U-Net, ResNet vision encoder, FiLM conditioning, EMA).
Only the noise process, loss, and inference loop differ:

Diffusion (DDPM):
    train:  x_noisy = schedule.add_noise(x, eps, t);  loss = MSE(net(x_noisy, t), eps)
    infer:  for t in reversed(timesteps): x = scheduler.step(net(x, t), t, x)

Flow Matching:
    train:  x_t = (1-t)*x_0 + t*x_1;  loss = MSE(net(x_t, t), x_1 - x_0)
    infer:  for i in range(steps): x += net(x, t) * dt   (Euler integration)

The U-Net predicts a velocity field v(x_t, t) instead of noise eps.
"""

from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionRgbEncoder,
)
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from torch import Tensor, nn

from lobe.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig


class FlowMatchingPolicy(PreTrainedPolicy):
    """Flow Matching Policy with the same interface as DiffusionPolicy.

    Reuses DiffusionPolicy's U-Net, vision encoder, and action chunking logic.
    Replaces DDPM forward/reverse diffusion with conditional flow matching.
    """

    config_class = FlowMatchingConfig
    name = "flow_matching"

    def __init__(
        self,
        config: FlowMatchingConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        self._queues = None
        self.flow_matching = FlowMatchingModel(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.flow_matching.parameters()

    def reset(self):
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.flow_matching.generate_actions(batch)
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        if ACTION in batch:
            batch.pop(ACTION)

        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        batch = self.normalize_targets(batch)
        loss = self.flow_matching.compute_loss(batch)
        return loss, None


class FlowMatchingModel(nn.Module):
    """Core flow matching model. Same architecture as DiffusionModel, different training/inference."""

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config

        # Build observation encoders — identical to DiffusionModel
        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        # Same U-Net — predicts velocity fields instead of noise, architecture is identical
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)

        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

        # Flow matching parameters
        self.sigma = config.sigma
        self.num_inference_steps = config.num_inference_steps

        # Optional: optimal transport coupling from torchcfm
        self._ot_sampler = None
        if config.use_optimal_transport:
            try:
                from torchcfm.optimal_transport import OTPlanSampler

                self._ot_sampler = OTPlanSampler(method="exact")
            except ImportError:
                raise ImportError(
                    "torchcfm is required for optimal transport coupling. Install with: pip install torchcfm"
                )

    def _sample_flow_matching(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sample timestep, interpolated point, and target velocity for flow matching.

        Implements conditional flow matching (Lipman et al., 2023):
            x_t = (1 - (1 - sigma) * t) * x_0 + t * x_1
            u_t = x_1 - (1 - sigma) * x_0

        Args:
            x0: Noise samples, shape (B, T, D)
            x1: Target trajectories (clean actions), shape (B, T, D)

        Returns:
            t: Sampled timesteps, shape (B, 1) for broadcasting
            x_t: Interpolated points, shape (B, T, D)
            u_t: Target velocity field, shape (B, T, D)
        """
        batch_size = x0.shape[0]
        t = torch.rand(batch_size, 1, 1, device=x0.device, dtype=x0.dtype)

        # Linear interpolation with optional sigma smoothing
        # For sigma=0: x_t = (1-t)*x0 + t*x1, u_t = x1 - x0
        mu_t = t * x1 + (1 - (1 - self.sigma) * t) * x0
        u_t = x1 - (1 - self.sigma) * x0

        return t.squeeze(-1), mu_t, u_t

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate with state vector. Identical to DiffusionModel."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]

        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [encoder(images) for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ..."))
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate actions via Euler integration of the learned velocity field.

        Instead of reverse diffusion (DDPM: x_t -> x_{t-1} for t=T..0),
        we integrate forward (FM: x_0 -> x_1 via dx = v(x,t)*dt for t=0..1).
        """
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Start from noise (x_0 in flow matching notation)
        sample = torch.randn(
            size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        dt = 1.0 / self.num_inference_steps

        for i in range(self.num_inference_steps):
            t_cur = i / self.num_inference_steps
            t = torch.full((batch_size,), t_cur, dtype=dtype, device=device)

            # Midpoint method (2nd-order) — much more accurate than Euler for same step count
            v1 = self.unet(sample, t, global_cond=global_cond)
            t_mid = torch.full((batch_size,), t_cur + 0.5 * dt, dtype=dtype, device=device)
            v2 = self.unet(sample + v1 * (0.5 * dt), t_mid, global_cond=global_cond)
            sample = sample + v2 * dt

        return sample

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)
        actions = self.conditional_sample(batch_size, global_cond=global_cond)

        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute flow matching loss.

        Instead of:
            DDPM: noisy = schedule.add_noise(clean, eps, t); loss = MSE(net(noisy, t), eps)
        We do:
            FM: x_t = (1-t)*x_0 + t*x_1; loss = MSE(net(x_t, t), x_1 - x_0)
        """
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        # x1 = target trajectory (clean actions)
        x1 = batch[ACTION]
        # x0 = noise
        x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype)

        # Optional: optimal transport coupling — reorder x0 to minimize transport cost to x1
        if self._ot_sampler is not None:
            b, t_dim, d = x0.shape
            x0_flat = x0.reshape(b, t_dim * d)
            x1_flat = x1.reshape(b, t_dim * d)
            x0_flat, x1_flat = self._ot_sampler.sample_plan(x0_flat, x1_flat)
            x0 = x0_flat.reshape(b, t_dim, d)
            x1 = x1_flat.reshape(b, t_dim, d)

        # Sample t, interpolated point x_t, and target velocity u_t
        t, x_t, u_t = self._sample_flow_matching(x0, x1)

        # The U-Net expects timestep as (B,) — t is (B, 1) after squeeze
        t_flat = t.squeeze(-1)

        # Predict velocity field
        v_t = self.unet(x_t, t_flat, global_cond=global_cond)

        # Flow matching loss: MSE between predicted and target velocity
        loss = F.mse_loss(v_t, u_t, reduction="none")

        # Mask padded actions
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    f"You need to provide 'action_is_pad' in the batch when {self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()
