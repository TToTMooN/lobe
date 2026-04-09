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
    """Reshape stat to broadcast against target — prepend leading 1s.

    Stat is assumed to already match the trailing dims of target
    (e.g. (3, 1, 1) for images, (14,) for state). We just prepend
    enough leading 1s so ndims match, then PyTorch broadcasting handles the rest.
    """
    while stat.ndim < target.ndim:
        stat = stat.unsqueeze(0)
    return stat


class Normalize(torch.nn.Module):
    """Normalize features using dataset statistics."""

    def __init__(self, features, normalization_mapping, dataset_stats):
        super().__init__()
        self.features = features
        self.normalization_mapping = normalization_mapping
        # Always register buffers (with zero defaults if no stats) so checkpoint loading works.
        for key, ft in features.items():
            mode = normalization_mapping.get(ft.type.value, NormalizationMode.IDENTITY)
            buf_key = key.replace(".", "_")
            shape = ft.shape if hasattr(ft, "shape") else (1,)
            # For images (CHW), use per-channel stats (C,1,1). Otherwise use full shape.
            if len(shape) == 3:
                stat_shape = (shape[0], 1, 1)
            else:
                stat_shape = shape
            if mode == NormalizationMode.MIN_MAX:
                if dataset_stats is not None and key in dataset_stats:
                    self.register_buffer(f"buffer_{buf_key}_min", _to_tensor(dataset_stats[key]["min"]))
                    self.register_buffer(f"buffer_{buf_key}_max", _to_tensor(dataset_stats[key]["max"]))
                else:
                    self.register_buffer(f"buffer_{buf_key}_min", torch.zeros(stat_shape))
                    self.register_buffer(f"buffer_{buf_key}_max", torch.ones(stat_shape))
            elif mode == NormalizationMode.MEAN_STD:
                if dataset_stats is not None and key in dataset_stats:
                    self.register_buffer(f"buffer_{buf_key}_mean", _to_tensor(dataset_stats[key]["mean"]))
                    self.register_buffer(f"buffer_{buf_key}_std", _to_tensor(dataset_stats[key]["std"]))
                else:
                    self.register_buffer(f"buffer_{buf_key}_mean", torch.zeros(stat_shape))
                    self.register_buffer(f"buffer_{buf_key}_std", torch.ones(stat_shape))

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
        # Always register buffers (with zero defaults if no stats) so checkpoint loading works.
        for key, ft in features.items():
            mode = normalization_mapping.get(ft.type.value, NormalizationMode.IDENTITY)
            buf_key = key.replace(".", "_")
            shape = ft.shape if hasattr(ft, "shape") else (1,)
            if len(shape) == 3:
                stat_shape = (shape[0], 1, 1)
            else:
                stat_shape = shape
            if mode == NormalizationMode.MIN_MAX:
                if dataset_stats is not None and key in dataset_stats:
                    self.register_buffer(f"buffer_{buf_key}_min", _to_tensor(dataset_stats[key]["min"]))
                    self.register_buffer(f"buffer_{buf_key}_max", _to_tensor(dataset_stats[key]["max"]))
                else:
                    self.register_buffer(f"buffer_{buf_key}_min", torch.zeros(stat_shape))
                    self.register_buffer(f"buffer_{buf_key}_max", torch.ones(stat_shape))
            elif mode == NormalizationMode.MEAN_STD:
                if dataset_stats is not None and key in dataset_stats:
                    self.register_buffer(f"buffer_{buf_key}_mean", _to_tensor(dataset_stats[key]["mean"]))
                    self.register_buffer(f"buffer_{buf_key}_std", _to_tensor(dataset_stats[key]["std"]))
                else:
                    self.register_buffer(f"buffer_{buf_key}_mean", torch.zeros(stat_shape))
                    self.register_buffer(f"buffer_{buf_key}_std", torch.ones(stat_shape))

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
