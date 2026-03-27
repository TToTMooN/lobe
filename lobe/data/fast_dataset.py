"""Fast tensor dataset — loads pre-prepared .pt cache for zero-overhead training.

Created by scripts/prepare_dataset.py. Stores all frames as pre-resized
tensors in a single file. Loads to GPU in seconds.

Usage:
    from lobe.data.fast_dataset import FastDataset
    dataset = FastDataset("datasets/aloha_insertion_224.pt", device="cuda")
    # Zero data loading overhead — all frames in VRAM
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import Dataset


class FastDataset(Dataset):
    """Dataset backed by a pre-prepared .pt tensor cache."""

    def __init__(self, path: str | Path, device: str = "cpu", delta_timestamps: dict | None = None):
        path = Path(path)
        logger.info(f"Loading fast dataset: {path}")
        self.cache = torch.load(path, weights_only=False)
        self.meta_info = self.cache.pop("__meta__", {})
        self.n_frames = self.meta_info.get("n_frames", 0)
        self.device = device

        # Move tensors to device
        for key, val in self.cache.items():
            if isinstance(val, torch.Tensor):
                self.cache[key] = val.to(device, non_blocking=True)

        total_mb = sum(v.nbytes for v in self.cache.values() if isinstance(v, torch.Tensor)) / 1e6
        logger.info(f"Loaded {self.n_frames} frames ({total_mb:.0f} MB) to {device}")

        # Build delta timestamp indices if provided
        self.delta_indices = {}
        if delta_timestamps:
            fps = self.meta_info.get("fps", 10.0)
            for key, timestamps in delta_timestamps.items():
                self.delta_indices[key] = [round(t * fps) for t in timestamps]

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx):
        item = {}
        for key, tensor in self.cache.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if key in self.delta_indices:
                # Stack temporal window
                indices = [max(0, min(idx + d, self.n_frames - 1)) for d in self.delta_indices[key]]
                item[key] = torch.stack([tensor[i] for i in indices])
            else:
                item[key] = tensor[idx]

        # Add required metadata fields
        item["index"] = torch.tensor(idx)
        if "action_is_pad" not in item and "action" in item:
            horizon = item["action"].shape[0] if item["action"].ndim > 1 else 1
            item["action_is_pad"] = torch.zeros(horizon, dtype=torch.bool)

        return item

    class _Meta:
        """Minimal shim so FastDataset.meta.stats works like LeRobotDataset.meta.stats."""

        def __init__(self, stats):
            self.stats = stats

    @property
    def meta(self):
        """Compatible with LeRobotDataset.meta interface."""
        return self._Meta(self.meta_info.get("stats", {}))

    @property
    def stats(self):
        """Return dataset statistics from metadata."""
        return self.meta_info.get("stats", {})
