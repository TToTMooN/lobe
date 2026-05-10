"""YAM bimanual robot config — for limb data and real robot deployment.

Data format: LeRobot v2.1/v3.0 from limb's convert_to_lerobot.py.
Action space: 14-dim (2x 6-DOF + gripper), matches ALOHA exactly.
Observations: 14-dim state (joint positions) + 2 wrist camera images.
"""

from __future__ import annotations

from lobe.data.loading import load_lerobot_dataset

# YAM bimanual constants (matches ALOHA action space)
FPS = 30.0  # limb data collection frequency
N_OBS_STEPS = 1  # VLAs typically use 1 obs step
HORIZON = 16  # action prediction horizon (for FM/Diffusion)
N_ACTION_STEPS = 8
ACTION_DIM = 14  # 6 joints + 1 gripper per arm x 2
STATE_DIM = 14
NUM_CAMERAS = 3  # head_camera, left_wrist_camera, right_wrist_camera
IMAGE_SHAPE = (480, 640, 3)  # raw capture from RealSense
MODEL_IMAGE_SHAPE = (240, 320, 3)  # resized for training (half resolution)

# VLA action chunk settings (pi0/smolvla convention)
VLA_CHUNK_SIZE = 50
VLA_N_ACTION_STEPS = 50

# Camera names (from limb's data collection)
CAMERA_NAMES = ["head_camera", "left_wrist_camera", "right_wrist_camera"]

# Default dataset (placeholder — user provides their own)
DEFAULT_DATASET = ""


def delta_timestamps(fps: float = FPS, n_obs_steps: int = N_OBS_STEPS, horizon: int = HORIZON):
    """Generate observation/action timestamps for YAM data."""
    obs_ts = [i / fps for i in range(1 - n_obs_steps, 1)]
    act_ts = [i / fps for i in range(1 - n_obs_steps, 1 - n_obs_steps + horizon)]

    timestamps = {
        "observation.state": obs_ts,
        "action": act_ts,
    }
    for cam in CAMERA_NAMES:
        timestamps[f"observation.images.{cam}"] = obs_ts

    return timestamps


def load_dataset(repo_id: str, fps: float = FPS, n_obs_steps: int = N_OBS_STEPS, horizon: int = HORIZON):
    """Load a YAM bimanual dataset from HuggingFace Hub or local path."""
    if not repo_id:
        raise ValueError("Must provide dataset repo_id for YAM (e.g. yourname/yam-red-cube)")
    return load_lerobot_dataset(repo_id, delta_timestamps(fps, n_obs_steps, horizon))


# No gym env for real robot — eval is done on hardware via limb
# evaluate() intentionally not defined
