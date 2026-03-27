"""LIBERO environment config — 7-DOF manipulation benchmarks.

LIBERO provides 130 tasks across 5 suites (spatial, object, goal, 10, 90/100).
7-dim action space (6 DOF + gripper), 256x256 images (front + wrist cameras).
Diffusion/FM policies achieve 80-99% on these tasks.
"""

from __future__ import annotations

from lobe.data.loading import load_lerobot_dataset

# LIBERO constants
FPS = 10.0
N_OBS_STEPS = 1
HORIZON = 16
N_ACTION_STEPS = 8
ACTION_DIM = 7  # 6 DOF + gripper
STATE_DIM = 8  # 7 + gripper state
MAX_STEPS = 300
DEFAULT_DATASET = "HuggingFaceVLA/libero"


def delta_timestamps():
    """Standard LIBERO observation/action timestamps."""
    obs_ts = [i / FPS for i in range(1 - N_OBS_STEPS, 1)]
    act_ts = [i / FPS for i in range(1 - N_OBS_STEPS, 1 - N_OBS_STEPS + HORIZON)]
    return {
        "observation.images.image": obs_ts,
        "observation.images.image2": obs_ts,
        "observation.state": obs_ts,
        "action": act_ts,
    }


def load_dataset(repo_id: str = DEFAULT_DATASET):
    """Load LIBERO dataset with standard timestamps."""
    return load_lerobot_dataset(repo_id, delta_timestamps())
