"""Prepare a LeRobot dataset for fast training — pre-resize and cache as tensors.

Converts any LeRobot dataset (image or video) into a single .pt file with
pre-resized images, ready to load to GPU in seconds.

Usage:
    # ALOHA 480x640 -> 224x224 (3.6GB, loads in 5s)
    uv run python scripts/prepare_dataset.py lerobot/aloha_sim_insertion_human_image --resize 224

    # PushT 96x96 -> no resize needed (already small)
    uv run python scripts/prepare_dataset.py lerobot/pusht_image

    # Custom output path
    uv run python scripts/prepare_dataset.py lerobot/aloha_sim_insertion_human_image \
      --resize 224 --output datasets/aloha_224.pt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from loguru import logger
from torchvision.transforms import functional as tvf
from tqdm import tqdm

import lobe.video_compat  # noqa: F401


def prepare(repo_id: str, resize: int | None = None, output: str | None = None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    logger.info(f"Loading dataset: {repo_id}")
    t0 = time.time()
    dataset = LeRobotDataset(repo_id)
    n = len(dataset)
    logger.info(f"Dataset: {n} frames, {len(dataset.meta.episodes)} episodes")

    # Identify feature types
    image_keys = []
    tensor_keys = []
    for key, feat in dataset.meta.features.items():
        dtype = feat.get("dtype", "")
        if dtype in ("image", "video"):
            image_keys.append(key)
        elif dtype in ("float32", "float64", "int64", "bool"):
            tensor_keys.append(key)

    logger.info(f"Image features: {image_keys}")
    logger.info(f"Tensor features: {tensor_keys}")

    if resize:
        logger.info(f"Resize images to: {resize}x{resize}")

    # Pre-allocate tensors
    logger.info("Reading all frames...")
    cache = {}

    # Read first item to get shapes
    item = dataset[0]
    for key in image_keys:
        img = item[key]
        if isinstance(img, torch.Tensor):
            if resize:
                img = tvf.resize(img, [resize, resize], antialias=True)
            cache[key] = torch.empty(n, *img.shape, dtype=torch.float32)
            cache[key][0] = img.float() if img.dtype != torch.float32 else img
        else:
            logger.warning(f"Skipping non-tensor image: {key} ({type(img)})")

    for key in tensor_keys:
        val = item[key]
        if isinstance(val, torch.Tensor):
            cache[key] = torch.empty(n, *val.shape, dtype=val.dtype)
            cache[key][0] = val

    # Fill the rest
    for i in tqdm(range(1, n), desc="Loading"):
        item = dataset[i]
        for key in image_keys:
            if key in cache:
                img = item[key]
                if resize:
                    img = tvf.resize(img, [resize, resize], antialias=True)
                cache[key][i] = img.float() if img.dtype != torch.float32 else img
        for key in tensor_keys:
            if key in cache:
                cache[key][i] = item[key]

    # Save
    if output is None:
        name = repo_id.split("/")[-1]
        suffix = f"_{resize}" if resize else ""
        output = f"datasets/{name}{suffix}.pt"

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Also save metadata
    cache["__meta__"] = {
        "repo_id": repo_id,
        "n_frames": n,
        "resize": resize,
        "features": {k: list(v.shape) for k, v in cache.items() if isinstance(v, torch.Tensor)},
        "stats": {k: dict(v) for k, v in dataset.meta.stats.items()} if hasattr(dataset.meta, "stats") else {},
    }

    torch.save(cache, out_path)
    mb = out_path.stat().st_size / 1e6
    elapsed = time.time() - t0
    logger.info(f"Saved: {out_path} ({mb:.0f} MB)")
    logger.info(f"Preparation time: {elapsed:.0f}s")
    logger.info(f"Load with: cache = torch.load('{out_path}')")

    # Print summary
    for key, val in cache.items():
        if isinstance(val, torch.Tensor):
            logger.info(f"  {key}: {val.shape} {val.dtype} ({val.nbytes / 1e6:.0f} MB)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/prepare_dataset.py <repo_id> [--resize N] [--output path]")
        sys.exit(1)

    repo_id = sys.argv[1]
    resize = None
    output = None

    if "--resize" in sys.argv:
        idx = sys.argv.index("--resize")
        resize = int(sys.argv[idx + 1])
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        output = sys.argv[idx + 1]

    prepare(repo_id, resize, output)


if __name__ == "__main__":
    main()
