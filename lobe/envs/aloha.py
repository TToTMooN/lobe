"""ALOHA sim environment config -- dataset parameters, constants, gym helpers.

Supports both AlohaInsertion-v0 and AlohaTransferCube-v0.
ALOHA is a bimanual manipulation platform with 14-dim action space (2x 6-DOF + gripper).
"""

from __future__ import annotations

import os

import numpy as np
import torch
from lerobot.envs.utils import preprocess_observation

from lobe.data.loading import load_lerobot_dataset

# ALOHA constants
FPS = 50.0
N_OBS_STEPS = 1
HORIZON = 96  # ALOHA uses longer horizons (must be divisible by 8 for U-Net)
N_ACTION_STEPS = 96
ACTION_DIM = 14  # 6 joints + 1 gripper per arm x 2
MAX_STEPS = 400
DEFAULT_DATASET = "lerobot/aloha_sim_insertion_human"

# Task -> gym env mapping
TASK_ENVS = {
    "insertion": "gym_aloha/AlohaInsertion-v0",
    "transfer_cube": "gym_aloha/AlohaTransferCube-v0",
}


def delta_timestamps():
    """Standard ALOHA observation/action timestamps."""
    obs_ts = [i / FPS for i in range(1 - N_OBS_STEPS, 1)]
    act_ts = [i / FPS for i in range(1 - N_OBS_STEPS, 1 - N_OBS_STEPS + HORIZON)]
    return {
        "observation.images.top": obs_ts,
        "observation.state": obs_ts,
        "action": act_ts,
    }


def load_dataset(repo_id: str = DEFAULT_DATASET):
    """Load ALOHA dataset with standard timestamps."""
    return load_lerobot_dataset(repo_id, delta_timestamps())


def obs_to_batch(obs: dict, device: str) -> dict[str, torch.Tensor]:
    """Preprocess a gym_aloha observation into a policy batch."""
    processed = preprocess_observation(obs)
    return {k: v.to(device) for k, v in processed.items()}


def _detect_task(dataset_repo_id: str = "") -> str:
    """Detect ALOHA task from dataset name or env var."""
    task = os.environ.get("ALOHA_TASK", "")
    if not task:
        if "transfer_cube" in dataset_repo_id:
            task = "transfer_cube"
        else:
            task = "insertion"
    return task


def run_rollout(policy, device: str, seed: int = 0, max_steps: int = MAX_STEPS, task: str = "") -> dict:
    """Run a single ALOHA rollout and return metrics."""
    import time

    os.environ.setdefault("MUJOCO_GL", "egl")

    import gym_aloha  # noqa: F401
    import gymnasium

    if not task:
        task = _detect_task()
    env_id = TASK_ENVS.get(task, TASK_ENVS["insertion"])
    env = gymnasium.make(env_id, obs_type="pixels_agent_pos")
    obs, _ = env.reset(seed=seed)
    policy.reset()

    rewards, latencies = [], []
    for _ in range(max_steps):
        batch = obs_to_batch(obs, device)
        t0 = time.perf_counter()
        with torch.no_grad():
            action = policy.select_action(batch)
        latencies.append(time.perf_counter() - t0)
        action_np = action[0].cpu().numpy() if action.dim() > 1 else action.cpu().numpy()
        action_np = np.clip(action_np, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action_np)
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


def evaluate(policy, device: str, n_rollouts: int = 10, seed: int = 0, task: str = "") -> tuple[float, float]:
    """Run multiple ALOHA rollouts and return (success_rate, avg_reward)."""
    policy.eval()
    successes, rewards = [], []
    for i in range(n_rollouts):
        result = run_rollout(policy, device, seed=seed + i, task=task)
        successes.append(result["success"])
        rewards.append(result["avg_reward"])
    return float(np.mean(successes)), float(np.mean(rewards))
