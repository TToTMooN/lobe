"""Convert a LeRobot video dataset to image format for faster training.

Video decode is the #1 training bottleneck for VLA models. Image datasets
load 5-10x faster. This script decodes all video frames and stores them
as images in parquet files.

Usage:
    uv run python scripts/convert_video_to_image.py lerobot/aloha_sim_insertion_human
    uv run python scripts/convert_video_to_image.py lerobot/pusht --output-dir datasets/pusht_image_v3
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

import lobe.video_compat  # noqa: F401


def convert(repo_id: str, output_dir: str | None = None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    logger.info(f"Loading dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id, video_backend="pyav")

    # Find video features
    video_keys = []
    for key, feat in dataset.meta.features.items():
        if feat.get("dtype") == "video":
            video_keys.append(key)

    if not video_keys:
        logger.info("No video features found — dataset is already image-based")
        return

    logger.info(f"Video features to convert: {video_keys}")
    logger.info(f"Total frames: {len(dataset)}")

    # Decode all frames
    logger.info("Decoding all video frames...")
    all_images = {key: [] for key in video_keys}

    for i in tqdm(range(len(dataset)), desc="Decoding"):
        item = dataset[i]
        for key in video_keys:
            if key in item:
                img = item[key]
                if isinstance(img, torch.Tensor):
                    img = img.numpy()
                all_images[key].append(img)

    # Save as local image dataset
    out_path = Path(output_dir) if output_dir else Path(f"datasets/{repo_id.split('/')[-1]}_image")
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving to: {out_path}")
    logger.info(f"Decoded {len(all_images[video_keys[0]])} frames per camera")

    # Save images as numpy arrays for simplicity
    for key in video_keys:
        images = np.stack(all_images[key])
        save_path = out_path / f"{key.replace('.', '_')}.npy"
        np.save(save_path, images)
        mb = images.nbytes / 1e6
        logger.info(f"  {key}: shape={images.shape}, {mb:.0f} MB")

    logger.info("Done. Use the image arrays for fast training.")
    logger.info(f"Total size: {sum(np.load(f).nbytes for f in out_path.glob('*.npy')) / 1e6:.0f} MB")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/convert_video_to_image.py <repo_id> [--output-dir <dir>]")
        sys.exit(1)

    repo_id = sys.argv[1]
    output_dir = None
    if "--output-dir" in sys.argv:
        idx = sys.argv.index("--output-dir")
        output_dir = sys.argv[idx + 1]

    convert(repo_id, output_dir)


if __name__ == "__main__":
    main()
