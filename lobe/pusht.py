"""PushT environment utilities — shared across train, eval, and sweep scripts.

Centralizes dataset loading, policy creation, checkpoint handling, and evaluation
to eliminate duplication and ensure consistency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from loguru import logger
from torch import nn

from lobe.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lobe.policies.flow_matching.modeling_flow_matching import FlowMatchingPolicy

# PushT constants — single source of truth
FPS = 10.0
N_OBS_STEPS = 2
HORIZON = 16
N_ACTION_STEPS = 8
MAX_STEPS = 300
DEFAULT_DATASET = "lerobot/pusht_image"


def load_dataset(repo_id: str = DEFAULT_DATASET):
    """Load PushT LeRobot dataset with standard observation/action timestamps."""
    obs_timestamps = [i / FPS for i in range(1 - N_OBS_STEPS, 1)]
    action_timestamps = [i / FPS for i in range(1 - N_OBS_STEPS, 1 - N_OBS_STEPS + HORIZON)]
    delta_timestamps = {
        "observation.image": obs_timestamps,
        "observation.state": obs_timestamps,
        "action": action_timestamps,
    }
    is_video = "image" not in repo_id
    kwargs = {"video_backend": "torchcodec"} if is_video else {}
    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, **kwargs)
    features = dataset_to_policy_features(dataset.meta.features)
    return dataset, features


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
    horizon: int = HORIZON,
    n_action_steps: int = N_ACTION_STEPS,
    num_inference_steps: int = 10,
    compile_model: bool = False,
    compile_mode: str = "reduce-overhead",
    fm_down_dims: tuple[int, ...] = (256, 512, 1024),
    fm_embed_dim: int = 256,
) -> nn.Module:
    """Create a FlowMatching or Diffusion policy with consistent config."""
    input_features, output_features = split_features(features)

    if policy_type == "flow_matching":
        config = FlowMatchingConfig(
            n_obs_steps=N_OBS_STEPS,
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
        config = DiffusionConfig(n_obs_steps=N_OBS_STEPS, horizon=horizon, n_action_steps=n_action_steps)
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


def obs_to_batch(obs: dict, device: str) -> dict[str, torch.Tensor]:
    """Preprocess a gym observation into a policy batch on device."""
    processed = preprocess_observation(obs)
    return {k: v.to(device) for k, v in processed.items()}


def run_rollout(policy, device: str, seed: int = 0, max_steps: int = MAX_STEPS) -> dict:
    """Run a single PushT rollout and return metrics."""
    import gym_pusht  # noqa: F401
    import gymnasium

    env = gymnasium.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels_agent_pos")
    obs, _ = env.reset(seed=seed)
    policy.reset()

    rewards, latencies = [], []
    for _ in range(max_steps):
        import time

        batch = obs_to_batch(obs, device)
        t0 = time.perf_counter()
        with torch.no_grad():
            action = policy.select_action(batch)
        latencies.append(time.perf_counter() - t0)
        obs, reward, terminated, truncated, info = env.step(action[0].cpu().numpy().clip(0, 512))
        rewards.append(reward)
        if terminated or truncated:
            break

    env.close()
    return {
        "avg_reward": float(np.mean(rewards)),
        "max_reward": float(np.max(rewards)),
        "success": bool(info.get("is_success", False)),
        "steps": len(rewards),
        "avg_latency_ms": float(np.mean(latencies) * 1000),
    }


def evaluate(policy, device: str, n_rollouts: int = 10, seed: int = 0) -> tuple[float, float]:
    """Run multiple rollouts and return (success_rate, avg_reward)."""
    policy.eval()
    successes, rewards = [], []
    for i in range(n_rollouts):
        result = run_rollout(policy, device, seed=seed + i)
        successes.append(result["success"])
        rewards.append(result["avg_reward"])
    return float(np.mean(successes)), float(np.mean(rewards))
