"""FastLeRobotDataset — subclass of LeRobotDataset with optimized data loading.

Key optimization: bypass HF datasets' `set_transform` for non-image columns when
querying action chunks. Without this, querying 50 future actions for a single
sample triggers decoding of 100 PNG images (the action chunk indices, all columns)
which are then thrown away — 12x slower than necessary.

This subclass replaces `_query_hf_dataset` with a fast path that accesses the
underlying Arrow table directly for non-image columns.
"""
from __future__ import annotations

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class FastLeRobotDataset(LeRobotDataset):
    """LeRobotDataset with optimized non-image column queries.

    Drop-in replacement for LeRobotDataset. Same constructor signature.
    """

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """Override: fast path for non-image columns.

        For action/state queries (called frequently for action chunks), access
        the Arrow table directly instead of going through `hf_dataset[key]`,
        which triggers `set_transform` and decodes all columns including PNGs.
        """
        result: dict = {}
        image_like_keys = set(self.meta.camera_keys) | set(self.meta.video_keys)
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            if key not in image_like_keys:
                # Fast path: bypass set_transform for non-image columns.
                try:
                    col = self.hf_dataset.data.column(key)
                    vals = [col[i].as_py() for i in relative_indices]
                    result[key] = torch.stack([torch.tensor(v) for v in vals])
                    continue
                except Exception:
                    pass  # Fall through to standard path on any failure
            try:
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result
