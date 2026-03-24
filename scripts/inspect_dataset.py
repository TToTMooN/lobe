"""Inspect a LeRobot dataset — show features, stats, episodes.

Usage:
    uv run python scripts/inspect_dataset.py lerobot/pusht
    uv run python scripts/inspect_dataset.py lerobot/pusht_image
    uv run python scripts/inspect_dataset.py yourname/yam-red-cube
"""

from __future__ import annotations

import sys

from loguru import logger

import lobe.video_compat  # noqa: F401
from lobe.data.loading import get_dataset_info


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_dataset.py <repo_id>")
        sys.exit(1)

    repo_id = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else None

    logger.info(f"Inspecting: {repo_id}")
    info = get_dataset_info(repo_id, root=root)

    print(f"\n{'=' * 60}")
    print(f"Dataset: {info['repo_id']}")
    print(f"Frames:  {info['n_frames']}")
    if "n_episodes" in info:
        print(f"Episodes: {info['n_episodes']}")
    if "fps" in info:
        print(f"FPS:     {info['fps']}")
    if "video" in info:
        print(f"Video:   {info['video']}")
    print(f"{'=' * 60}")
    print("\nFeatures:")
    for name, ft in info["features"].items():
        shape = str(ft.get("shape", "?"))
        dtype = str(ft.get("dtype", "?"))
        print(f"  {name:40s} shape={shape:<20s} dtype={dtype}")
    print()


if __name__ == "__main__":
    main()
