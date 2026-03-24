"""Generic LeRobot dataset loading utilities.

Works with any LeRobot v2.1 dataset — PushT, limb YAM data, or HuggingFace Hub datasets.
Environment-specific timestamp configs live in lobe/envs/.
"""

from __future__ import annotations

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features


def load_lerobot_dataset(
    repo_id: str,
    delta_timestamps: dict[str, list[float]],
    video_backend: str | None = None,
):
    """Load a LeRobot dataset with given timestamp config.

    Args:
        repo_id: HuggingFace dataset repo ID (e.g. "lerobot/pusht_image", "yourname/yam-red-cube").
        delta_timestamps: Per-feature timestamp offsets (relative to current frame).
        video_backend: "torchcodec", "pyav", or None (auto-detect: image datasets skip backend).

    Returns:
        (dataset, features) tuple.
    """
    if video_backend is None:
        # Image datasets don't need a video backend
        is_video = "image" not in repo_id
        kwargs = {"video_backend": "torchcodec"} if is_video else {}
    else:
        kwargs = {"video_backend": video_backend}

    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, **kwargs)
    features = dataset_to_policy_features(dataset.meta.features)
    return dataset, features
