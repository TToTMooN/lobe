"""Generic LeRobot dataset loading utilities.

Works with any LeRobot dataset (v2.1, v3.0) — PushT, limb YAM data, HuggingFace Hub datasets.
Environment-specific timestamp configs live in lobe/envs/.
"""

from __future__ import annotations

from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from loguru import logger


def load_lerobot_dataset(
    repo_id: str,
    delta_timestamps: dict[str, list[float]],
    video_backend: str | None = None,
    root: str | None = None,
    episodes: list[int] | None = None,
):
    """Load a LeRobot dataset with given timestamp config.

    Args:
        repo_id: HuggingFace dataset repo ID or local path.
        delta_timestamps: Per-feature timestamp offsets (relative to current frame).
        video_backend: "torchcodec", "pyav", or None (auto-detect).
        root: Local dataset root directory (overrides HF download).
        episodes: Subset of episode indices to load.

    Returns:
        (dataset, features) tuple.
    """
    if video_backend is None:
        # Image datasets (with "image" in name) don't need a video backend
        is_video = "image" not in repo_id
        kwargs = {"video_backend": "torchcodec"} if is_video else {}
    else:
        kwargs = {"video_backend": video_backend}

    if root is not None:
        kwargs["root"] = root
    if episodes is not None:
        kwargs["episodes"] = episodes

    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, **kwargs)
    features = dataset_to_policy_features(dataset.meta.features)

    # Log dataset info
    n_episodes = len(dataset.meta.episodes) if hasattr(dataset.meta, "episodes") else "?"
    logger.info(f"Dataset: {repo_id} | {len(dataset)} frames | {n_episodes} episodes")
    for key, ft in features.items():
        logger.debug(f"  {key}: shape={ft.shape}, type={ft.type}")

    return dataset, features


def get_dataset_info(repo_id: str, root: str | None = None) -> dict:
    """Get dataset metadata without loading the full dataset.

    Useful for inspecting a dataset before training.
    """
    dataset = LeRobotDataset(repo_id, root=root)
    info = {
        "repo_id": repo_id,
        "n_frames": len(dataset),
        "features": {
            k: {"shape": v.get("shape", v.get("names", "?")), "dtype": v.get("dtype", "?")}
            for k, v in dataset.meta.features.items()
        },
    }
    if hasattr(dataset.meta, "info"):
        info["fps"] = dataset.meta.info.get("fps")
        info["video"] = dataset.meta.info.get("video", False)
    if hasattr(dataset.meta, "episodes"):
        info["n_episodes"] = len(dataset.meta.episodes)
    return info
