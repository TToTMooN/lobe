"""Build a new LeRobot dataset from HuggingFaceVLA/libero observations + converted abs_action_6d.

Reads:
  - HuggingFaceVLA/libero (original) — provides images, state, task, timestamps
  - cache_dir/episode_{i:06d}.npz — abs_action_6d per episode (10-D) from rel2abs conversion

Writes a new LeRobot v2.1 dataset where:
  - action shape [10] = [abs_xyz(3), rot6d(6), gripper(1)]   (was [7] = delta)
  - all other fields copied verbatim from the source
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import torch
import tyro
from loguru import logger

import lobe  # noqa: F401

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def build_dataset(
    cache_dir: Path,
    output_root: Path,
    repo_id: str,
    fps: int = 10,
) -> None:
    """Build the new dataset by iterating through HuggingFaceVLA/libero and swapping action."""
    logger.info("Loading source dataset HuggingFaceVLA/libero...")
    src = LeRobotDataset("HuggingFaceVLA/libero")
    n_episodes = src.num_episodes
    logger.info(f"Source: {n_episodes} episodes, {src.num_frames} frames")

    # Build new dataset features: same as src but action=(10,)
    src_features = dict(src.meta.features)
    src_features["action"] = {
        **src_features["action"],
        "shape": (10,),
        "names": ["abs_xyz_x", "abs_xyz_y", "abs_xyz_z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5", "gripper"],
    }

    # Only include the features we actually write; LeRobotDataset manages episode_index, frame_index, etc.
    feature_keys_for_create = {}
    for k, v in src_features.items():
        if k in {"episode_index", "frame_index", "index", "task_index", "timestamp"}:
            continue
        feature_keys_for_create[k] = v

    logger.info(f"Creating new dataset at {output_root}/{repo_id} with features: {list(feature_keys_for_create.keys())}")

    output_root.mkdir(parents=True, exist_ok=True)
    dst = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=feature_keys_for_create,
        root=output_root / repo_id,
        use_videos=False,  # store images raw for reproducibility
    )

    # Iterate episodes
    ep_dataidx = src.episode_data_index
    skipped = 0
    for ep_idx in range(n_episodes):
        cache_file = cache_dir / f"episode_{ep_idx:06d}.npz"
        if not cache_file.exists():
            logger.warning(f"Episode {ep_idx}: no cache file, skipping")
            skipped += 1
            continue

        cache = np.load(cache_file, allow_pickle=True)
        abs_action_6d = cache["abs_action_6d"]  # (T, 10)

        # Get source episode frames
        frame_start = int(ep_dataidx["from"][ep_idx])
        frame_end = int(ep_dataidx["to"][ep_idx])
        n_frames = frame_end - frame_start

        if n_frames != abs_action_6d.shape[0]:
            logger.warning(
                f"Episode {ep_idx}: frame count mismatch (src={n_frames}, cache={abs_action_6d.shape[0]}), skipping"
            )
            skipped += 1
            continue

        # Add frames
        for local_t in range(n_frames):
            src_frame = src[frame_start + local_t]
            new_action = torch.from_numpy(abs_action_6d[local_t]).float()
            frame = {
                "observation.images.image": src_frame["observation.images.image"],
                "observation.images.image2": src_frame["observation.images.image2"],
                "observation.state": src_frame["observation.state"],
                "action": new_action,
                "task": src_frame["task"],
            }
            dst.add_frame(frame)

        dst.save_episode()
        if (ep_idx + 1) % 50 == 0:
            logger.info(f"[{ep_idx+1}/{n_episodes}] episodes written (skipped={skipped})")

    logger.success(f"Dataset built: {dst.num_episodes} episodes, {dst.num_frames} frames, skipped {skipped}")
    logger.info(f"Output: {output_root / repo_id}")


def main(
    cache_dir: str = "/mnt/localssd/sunlingfeng/datasets/libero_abs_action_6d_cache",
    output_root: str = "/mnt/localssd/sunlingfeng/datasets",
    repo_id: str = "local/libero_abs_action_6d",
):
    """Build the abs_action_6d LeRobot dataset."""
    build_dataset(Path(cache_dir), Path(output_root), repo_id)


if __name__ == "__main__":
    tyro.cli(main)
