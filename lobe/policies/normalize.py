"""Normalize / Unnormalize for policy inputs and outputs.

Replaces lerobot.policies.normalize (removed in lerobot v0.5+).
Implements MIN_MAX normalization to [-1, 1] and MEAN_STD normalization.
"""

from __future__ import annotations

import numpy as np
import torch
from lerobot.configs.types import NormalizationMode
from torch import Tensor


def _to_tensor(x):
    """Convert numpy array or tensor to torch tensor."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.float() if isinstance(x, Tensor) else torch.tensor(x, dtype=torch.float32)


class Normalize(torch.nn.Module):
    """Normalize features using dataset statistics."""

    def __init__(self, features, normalization_mapping, dataset_stats):
        super().__init__()
        self.features = features
        self.normalization_mapping = normalization_mapping
        if dataset_stats is not None:
            for key, ft in features.items():
                mode = normalization_mapping.get(ft.type.value, NormalizationMode.IDENTITY)
                if mode == NormalizationMode.MIN_MAX and key in dataset_stats:
                    self.register_buffer(f"buffer_{key.replace('.', '_')}_min", _to_tensor(dataset_stats[key]["min"]))
                    self.register_buffer(f"buffer_{key.replace('.', '_')}_max", _to_tensor(dataset_stats[key]["max"]))
                elif mode == NormalizationMode.MEAN_STD and key in dataset_stats:
                    val = _to_tensor(dataset_stats[key]["mean"])
                    self.register_buffer(f"buffer_{key.replace('.', '_')}_mean", val)
                    self.register_buffer(f"buffer_{key.replace('.', '_')}_std", _to_tensor(dataset_stats[key]["std"]))

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)
        for key, ft in self.features.items():
            if key not in batch:
                continue
            mode = self.normalization_mapping.get(ft.type.value, NormalizationMode.IDENTITY)
            buf_key = key.replace(".", "_")
            if mode == NormalizationMode.MIN_MAX:
                mn = getattr(self, f"buffer_{buf_key}_min")
                mx = getattr(self, f"buffer_{buf_key}_max")
                batch[key] = (batch[key] - mn) / (mx - mn + 1e-8) * 2 - 1
            elif mode == NormalizationMode.MEAN_STD:
                mean = getattr(self, f"buffer_{buf_key}_mean")
                std = getattr(self, f"buffer_{buf_key}_std")
                batch[key] = (batch[key] - mean) / (std + 1e-8)
        return batch


class Unnormalize(torch.nn.Module):
    """Unnormalize features back to original scale."""

    def __init__(self, features, normalization_mapping, dataset_stats):
        super().__init__()
        self.features = features
        self.normalization_mapping = normalization_mapping
        if dataset_stats is not None:
            for key, ft in features.items():
                mode = normalization_mapping.get(ft.type.value, NormalizationMode.IDENTITY)
                if mode == NormalizationMode.MIN_MAX and key in dataset_stats:
                    self.register_buffer(f"buffer_{key.replace('.', '_')}_min", _to_tensor(dataset_stats[key]["min"]))
                    self.register_buffer(f"buffer_{key.replace('.', '_')}_max", _to_tensor(dataset_stats[key]["max"]))
                elif mode == NormalizationMode.MEAN_STD and key in dataset_stats:
                    val = _to_tensor(dataset_stats[key]["mean"])
                    self.register_buffer(f"buffer_{key.replace('.', '_')}_mean", val)
                    self.register_buffer(f"buffer_{key.replace('.', '_')}_std", _to_tensor(dataset_stats[key]["std"]))

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)
        for key, ft in self.features.items():
            if key not in batch:
                continue
            mode = self.normalization_mapping.get(ft.type.value, NormalizationMode.IDENTITY)
            buf_key = key.replace(".", "_")
            if mode == NormalizationMode.MIN_MAX:
                mn = getattr(self, f"buffer_{buf_key}_min")
                mx = getattr(self, f"buffer_{buf_key}_max")
                batch[key] = (batch[key] + 1) / 2 * (mx - mn + 1e-8) + mn
            elif mode == NormalizationMode.MEAN_STD:
                mean = getattr(self, f"buffer_{buf_key}_mean")
                std = getattr(self, f"buffer_{buf_key}_std")
                batch[key] = batch[key] * (std + 1e-8) + mean
        return batch
