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


def _broadcast_stat(stat: Tensor, target: Tensor) -> Tensor:
    """Reshape stat to broadcast against target.

    Stats match the feature shape (e.g. (C,) for images, (D,) for state).
    Target has extra leading dims (batch, time). We reshape stat to align
    its dims with the trailing dims of target, adding leading 1s.

    Examples:
        stat (3,) + target (B, T, 3, H, W) -> (1, 1, 3, 1, 1)  [image: C aligns to dim 2]
        stat (14,) + target (B, T, 14) -> (1, 1, 14)            [state: D aligns to last dim]
        stat (14,) + target (B, 14) -> (1, 14)                  [action: D aligns to last dim]
    """
    # Find where stat's first dim matches in target (scan from the right)
    n_trailing = 0
    for i in range(1, target.ndim + 1):
        if stat.ndim > 0 and target.shape[-i] == stat.shape[0]:
            n_trailing = i - 1
            break
    # Add trailing 1s for spatial dims, then leading 1s for batch/time
    for _ in range(n_trailing):
        stat = stat.unsqueeze(-1)
    while stat.ndim < target.ndim:
        stat = stat.unsqueeze(0)
    return stat


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
                mn = _broadcast_stat(getattr(self, f"buffer_{buf_key}_min"), batch[key])
                mx = _broadcast_stat(getattr(self, f"buffer_{buf_key}_max"), batch[key])
                batch[key] = (batch[key] - mn) / (mx - mn + 1e-8) * 2 - 1
            elif mode == NormalizationMode.MEAN_STD:
                mean = _broadcast_stat(getattr(self, f"buffer_{buf_key}_mean"), batch[key])
                std = _broadcast_stat(getattr(self, f"buffer_{buf_key}_std"), batch[key])
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
                mn = _broadcast_stat(getattr(self, f"buffer_{buf_key}_min"), batch[key])
                mx = _broadcast_stat(getattr(self, f"buffer_{buf_key}_max"), batch[key])
                batch[key] = (batch[key] + 1) / 2 * (mx - mn + 1e-8) + mn
            elif mode == NormalizationMode.MEAN_STD:
                mean = _broadcast_stat(getattr(self, f"buffer_{buf_key}_mean"), batch[key])
                std = _broadcast_stat(getattr(self, f"buffer_{buf_key}_std"), batch[key])
                batch[key] = batch[key] * (std + 1e-8) + mean
        return batch
