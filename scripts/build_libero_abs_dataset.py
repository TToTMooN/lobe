"""Build a new LeRobot dataset by copying HuggingFaceVLA/libero's parquet files
and swapping the `action` column with the converted `abs_action_6d` values.

This is a direct parquet-level rewriter — no LeRobotDataset.create, no video re-encoding,
no per-frame image writes. It reuses the source dataset's already-encoded images and only
replaces the action column.

Input:
  - Source LeRobot cache: /mnt/localssd/sunlingfeng/cache/huggingface/lerobot/HuggingFaceVLA/libero/
  - Converted actions:    /mnt/localssd/sunlingfeng/datasets/libero_abs_action_6d_cache/episode_*.npz

Output:
  - Local dataset rooted at /mnt/localssd/sunlingfeng/datasets/local/libero_abs_action_6d/
    with same layout as source but action column swapped to shape [10]
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro
from loguru import logger


def build_dataset(
    source_root: Path,
    cache_dir: Path,
    output_root: Path,
) -> None:
    source_data = source_root / "data" / "chunk-000"
    source_meta = source_root / "meta"
    output_data = output_root / "data" / "chunk-000"
    output_meta = output_root / "meta"

    output_data.mkdir(parents=True, exist_ok=True)
    output_meta.mkdir(parents=True, exist_ok=True)

    # Load all per-episode abs_action_6d into a single dict keyed by (episode_index, frame_index)
    logger.info("Loading cached abs_action_6d...")
    abs_actions: dict[int, np.ndarray] = {}
    for ep_idx in range(1693):
        cache_file = cache_dir / f"episode_{ep_idx:06d}.npz"
        if not cache_file.exists():
            logger.warning(f"Missing episode {ep_idx}")
            continue
        abs_actions[ep_idx] = np.load(cache_file, allow_pickle=True)["abs_action_6d"].astype(np.float32)
    logger.info(f"Loaded {len(abs_actions)} cached episode actions")

    # Process parquet files
    parquet_files = sorted(source_data.glob("file-*.parquet"))
    logger.info(f"Rewriting {len(parquet_files)} parquet files...")

    total_frames = 0
    missing_episodes = 0
    for i, src_file in enumerate(parquet_files):
        dst_file = output_data / src_file.name
        table = pq.read_table(src_file)

        # Group rows by episode_index and build the new action column
        ep_col = table.column("episode_index").to_pylist()
        frame_col = table.column("frame_index").to_pylist()

        new_action_list: list[list[float]] = [None] * len(table)
        for row_idx, (ep_idx, frame_idx) in enumerate(zip(ep_col, frame_col)):
            if ep_idx not in abs_actions:
                # Episode was not converted — fallback to zeros to keep schema consistent
                new_action_list[row_idx] = [0.0] * 10
                missing_episodes += 1
                continue
            ep_actions = abs_actions[ep_idx]
            if frame_idx >= ep_actions.shape[0]:
                new_action_list[row_idx] = [0.0] * 10
                missing_episodes += 1
                continue
            new_action_list[row_idx] = ep_actions[frame_idx].tolist()

        # Replace the action column (pyarrow list<float32>)
        new_action_array = pa.array(new_action_list, type=pa.list_(pa.float32()))
        new_table = table.set_column(
            table.schema.get_field_index("action"),
            "action",
            new_action_array,
        )

        pq.write_table(new_table, dst_file, compression="snappy")
        total_frames += len(table)
        if (i + 1) % 50 == 0:
            logger.info(f"[{i+1}/{len(parquet_files)}] files written, {total_frames} frames, missing={missing_episodes}")

    logger.success(
        f"Wrote {len(parquet_files)} parquet files ({total_frames} frames, {missing_episodes} fallback frames)"
    )

    # Copy meta files: info.json (with updated action feature shape), tasks.parquet,
    # episodes directory, and stats.json. The schema in info.json MUST have action shape (10,)
    # or the loader will error.
    logger.info("Copying and updating meta files...")

    # info.json — update action shape
    src_info = json.loads((source_meta / "info.json").read_text())
    src_info["features"]["action"] = {
        **src_info["features"]["action"],
        "shape": [10],
        "names": [
            "abs_xyz_x",
            "abs_xyz_y",
            "abs_xyz_z",
            "rot6d_0",
            "rot6d_1",
            "rot6d_2",
            "rot6d_3",
            "rot6d_4",
            "rot6d_5",
            "gripper",
        ],
    }
    (output_meta / "info.json").write_text(json.dumps(src_info, indent=2))

    # tasks.parquet — copy as-is
    shutil.copy2(source_meta / "tasks.parquet", output_meta / "tasks.parquet")

    # episodes directory — copy recursively
    src_eps = source_meta / "episodes"
    dst_eps = output_meta / "episodes"
    if dst_eps.exists():
        shutil.rmtree(dst_eps)
    shutil.copytree(src_eps, dst_eps)

    # stats.json — copy as-is (we don't use it for IDENTITY normalization anyway)
    if (source_meta / "stats.json").exists():
        shutil.copy2(source_meta / "stats.json", output_meta / "stats.json")

    # Videos: symlink the source videos directory so we don't duplicate GBs of data
    src_videos = source_root / "videos"
    dst_videos = output_root / "videos"
    if src_videos.exists() and not dst_videos.exists():
        dst_videos.symlink_to(src_videos)
        logger.info(f"Symlinked videos: {dst_videos} → {src_videos}")

    logger.success(f"Dataset built at {output_root}")


def main(
    source_root: str = "/mnt/localssd/sunlingfeng/cache/huggingface/lerobot/HuggingFaceVLA/libero",
    cache_dir: str = "/mnt/localssd/sunlingfeng/datasets/libero_abs_action_6d_cache",
    output_root: str = "/mnt/localssd/sunlingfeng/datasets/local/libero_abs_action_6d",
):
    build_dataset(Path(source_root), Path(cache_dir), Path(output_root))


if __name__ == "__main__":
    tyro.cli(main)
