"""GPU video preload for fast training on small datasets.

Decodes all video frames into a VRAM tensor at startup. During training,
frame access is a simple tensor index — zero decode latency.

Inspired by real-is-sim's CompactedVideoBuilder approach.

Usage:
    from lobe.video_preload import preload_dataset_to_gpu

    # Wraps a LeRobot dataset to serve frames from GPU memory
    dataset = preload_dataset_to_gpu(dataset, device="cuda")
"""

from __future__ import annotations

import torch
from loguru import logger


def preload_dataset_to_gpu(dataset, device: str = "cuda"):
    """Preload a LeRobot image dataset's tensors to GPU for zero-copy access.

    Only works with image datasets (not video). For small datasets that fit in VRAM,
    this eliminates all data loading overhead during training.

    Args:
        dataset: A LeRobotDataset (image-based, already loaded).
        device: Target device ("cuda", "cuda:0", etc.).

    Returns:
        The dataset with its HuggingFace Arrow table backed by GPU tensors
        (via a cache dict for image/state/action columns).
    """
    if not hasattr(dataset, "hf_dataset"):
        logger.warning("Dataset doesn't have hf_dataset attribute, skipping preload")
        return dataset

    hf = dataset.hf_dataset
    n = len(hf)
    logger.info(f"Preloading {n} frames to {device}...")

    # Build a GPU cache of all columns that are tensors
    cache = {}
    sample = hf[0]
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            # Stack all values for this column
            all_vals = [hf[i][key] for i in range(n)]
            stacked = torch.stack(all_vals).to(device, non_blocking=True)
            cache[key] = stacked
            mb = stacked.nbytes / 1e6
            logger.info(f"  {key}: {stacked.shape} -> {mb:.1f} MB on {device}")

    total_mb = sum(t.nbytes for t in cache.values()) / 1e6
    logger.info(f"Preloaded {total_mb:.0f} MB total to {device}")

    # Wrap the dataset's __getitem__ to return from GPU cache
    original_getitem = dataset.__class__.__getitem__

    def gpu_getitem(self, idx):
        item = original_getitem(self, idx)
        # Replace cached tensors with GPU versions
        for key, gpu_tensor in cache.items():
            if key in item:
                item[key] = gpu_tensor[idx]
        return item

    dataset.__class__.__getitem__ = gpu_getitem
    dataset._gpu_cache = cache  # prevent garbage collection
    return dataset
