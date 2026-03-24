"""Policy creation and checkpoint loading — generic across all environments.

Supports FlowMatching and Diffusion policies. Environment-specific configs
(obs dims, action dims, image shapes) come from the dataset features.
"""

from __future__ import annotations

from pathlib import Path

import torch
from lerobot.configs.types import FeatureType
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from loguru import logger
from torch import nn

from lobe.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lobe.policies.flow_matching.modeling_flow_matching import FlowMatchingPolicy


def split_features(features: dict):
    """Split dataset features into input (obs) and output (action) features."""
    input_features = {k: v for k, v in features.items() if v.type != FeatureType.ACTION}
    output_features = {k: v for k, v in features.items() if v.type == FeatureType.ACTION}
    return input_features, output_features


def create_policy(
    policy_type: str,
    features: dict,
    stats: dict,
    *,
    n_obs_steps: int = 2,
    horizon: int = 16,
    n_action_steps: int = 8,
    num_inference_steps: int = 10,
    compile_model: bool = False,
    compile_mode: str = "reduce-overhead",
    # FM-specific
    fm_down_dims: tuple[int, ...] = (256, 512, 1024),
    fm_embed_dim: int = 256,
) -> nn.Module:
    """Create a policy from dataset features. Works for any environment."""
    input_features, output_features = split_features(features)

    if policy_type == "flow_matching":
        config = FlowMatchingConfig(
            n_obs_steps=n_obs_steps,
            horizon=horizon,
            n_action_steps=n_action_steps,
            num_inference_steps=num_inference_steps,
            compile_model=compile_model,
            compile_mode=compile_mode,
            down_dims=fm_down_dims,
            diffusion_step_embed_dim=fm_embed_dim,
        )
        config.input_features = input_features
        config.output_features = output_features
        return FlowMatchingPolicy(config, dataset_stats=stats)
    elif policy_type == "diffusion":
        config = DiffusionConfig(
            n_obs_steps=n_obs_steps,
            horizon=horizon,
            n_action_steps=n_action_steps,
        )
        config.input_features = input_features
        config.output_features = output_features
        return DiffusionPolicy(config, dataset_stats=stats)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def load_checkpoint(policy: nn.Module, checkpoint: str | Path, device: str = "cuda") -> bool:
    """Load checkpoint into policy, handling torch.compile _orig_mod prefix.

    Returns True if checkpoint was loaded, False if not found.
    """
    if not checkpoint or not Path(checkpoint).exists():
        return False

    ckpt_path = Path(checkpoint)
    if ckpt_path.is_dir():
        for name in ["model.pt", "model.safetensors", "pytorch_model.bin"]:
            if (ckpt_path / name).exists():
                ckpt_path = ckpt_path / name
                break

    if not ckpt_path.is_file():
        return False

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)
    logger.info(f"Loaded checkpoint: {ckpt_path}")
    return True
