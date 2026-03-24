"""YAM bimanual robot config — for limb data and real robot deployment.

Data format: LeRobot v2.1 from limb's convert_to_lerobot.py.
Action space: 14-dim (2x 6-DOF + gripper), matches ALOHA exactly.
Observations: 14-dim state (joint positions) + 2 wrist camera images (480x640).
"""

from __future__ import annotations

from lobe.data.loading import load_lerobot_dataset

# YAM bimanual constants (matches ALOHA action space)
FPS = 50.0  # limb control frequency (may vary per config)
N_OBS_STEPS = 2
HORIZON = 16
N_ACTION_STEPS = 8
ACTION_DIM = 14  # 6 joints + 1 gripper per arm x 2
STATE_DIM = 14  # same as action dim
IMAGE_SHAPE = (480, 640, 3)  # raw capture resolution
MODEL_IMAGE_SHAPE = (224, 224, 3)  # model input resolution (resized during training)
NUM_CAMERAS = 2  # left_wrist_camera, right_wrist_camera


def delta_timestamps():
    """Standard YAM observation/action timestamps."""
    obs_ts = [i / FPS for i in range(1 - N_OBS_STEPS, 1)]
    act_ts = [i / FPS for i in range(1 - N_OBS_STEPS, 1 - N_OBS_STEPS + HORIZON)]
    return {
        "observation.images.left_wrist_camera": obs_ts,
        "observation.images.right_wrist_camera": obs_ts,
        "observation.state": obs_ts,
        "action": act_ts,
    }


def load_dataset(repo_id: str):
    """Load a YAM bimanual dataset from HuggingFace Hub."""
    return load_lerobot_dataset(repo_id, delta_timestamps())
