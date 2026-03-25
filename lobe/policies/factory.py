"""Policy creation and checkpoint loading — generic across all environments.

Supports FlowMatching, Diffusion, and VLA policies (pi0, SmolVLA via LeRobot).
Environment-specific configs (obs dims, action dims) come from the dataset features.
"""

from __future__ import annotations

from pathlib import Path

import torch
from lerobot.configs.types import FeatureType
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from loguru import logger
from torch import nn

from lobe.policies.diffusion_wrapper import NormalizedDiffusionPolicy
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
    resize_shape: tuple[int, int] | None = None,
) -> nn.Module:
    """Create a policy from dataset features. Works for any environment.

    For VLA policies (pi0, smolvla), use lerobot-train directly or scripts/train_vla.py.
    This factory handles FM and Diffusion baselines.
    """
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
            resize_shape=resize_shape,
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
        return NormalizedDiffusionPolicy(config, dataset_stats=stats)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}. Use 'flow_matching' or 'diffusion'.")


def load_checkpoint(policy: nn.Module, checkpoint: str | Path, device: str = "cuda") -> bool:
    """Load checkpoint into policy, handling torch.compile _orig_mod prefix.

    Works with checkpoints from:
    - lobe's training scripts (model.pt)
    - LeRobot's lerobot-train (model.safetensors)
    - HuggingFace Hub (pytorch_model.bin)

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


def load_pretrained_policy(pretrained_path: str, device: str = "cuda") -> nn.Module:
    """Load a pretrained LeRobot policy from HuggingFace Hub.

    Works with any registered LeRobot policy (pi0, smolvla, diffusion, etc.).
    The policy config and weights are loaded from the pretrained path.

    Args:
        pretrained_path: HuggingFace model path (e.g. "lerobot/smolvla_base", "lerobot/pi0").
        device: Target device.

    Returns:
        Loaded policy ready for inference or fine-tuning.
    """
    from lerobot.configs.policies import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(pretrained_path)
    policy_class = config.get_choice_class(config.type)
    policy = policy_class(config)
    policy.to(device)
    logger.info(f"Loaded pretrained policy: {pretrained_path} ({type(policy).__name__})")
    return policy
