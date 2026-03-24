"""PushT environment config — dataset parameters, constants, gym helpers."""

from __future__ import annotations

import numpy as np
import torch
from lerobot.envs.utils import preprocess_observation

from lobe.data.loading import load_lerobot_dataset

# PushT constants
FPS = 10.0
N_OBS_STEPS = 2
HORIZON = 16
N_ACTION_STEPS = 8
MAX_STEPS = 300
DEFAULT_DATASET = "lerobot/pusht_image"


def delta_timestamps():
    """Standard PushT observation/action timestamps."""
    obs_ts = [i / FPS for i in range(1 - N_OBS_STEPS, 1)]
    act_ts = [i / FPS for i in range(1 - N_OBS_STEPS, 1 - N_OBS_STEPS + HORIZON)]
    return {
        "observation.image": obs_ts,
        "observation.state": obs_ts,
        "action": act_ts,
    }


def load_dataset(repo_id: str = DEFAULT_DATASET):
    """Load PushT dataset with standard timestamps."""
    return load_lerobot_dataset(repo_id, delta_timestamps())


def obs_to_batch(obs: dict, device: str) -> dict[str, torch.Tensor]:
    """Preprocess a gym-pusht observation into a policy batch."""
    processed = preprocess_observation(obs)
    return {k: v.to(device) for k, v in processed.items()}


def run_rollout(policy, device: str, seed: int = 0, max_steps: int = MAX_STEPS) -> dict:
    """Run a single PushT rollout and return metrics."""
    import time

    import gym_pusht  # noqa: F401
    import gymnasium

    env = gymnasium.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels_agent_pos")
    obs, _ = env.reset(seed=seed)
    policy.reset()

    rewards, latencies = [], []
    for _ in range(max_steps):
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
        "coverage": float(info.get("coverage", 0)),
        "steps": len(rewards),
        "avg_latency_ms": float(np.mean(latencies) * 1000),
    }


def evaluate(policy, device: str, n_rollouts: int = 10, seed: int = 0) -> tuple[float, float]:
    """Run multiple PushT rollouts and return (success_rate, avg_reward)."""
    policy.eval()
    successes, rewards = [], []
    for i in range(n_rollouts):
        result = run_rollout(policy, device, seed=seed + i)
        successes.append(result["success"])
        rewards.append(result["avg_reward"])
    return float(np.mean(successes)), float(np.mean(rewards))
