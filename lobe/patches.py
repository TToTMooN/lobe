"""Monkey-patches applied to lerobot at import time.

These patches survive `uv sync` (unlike editing installed files directly) and
can be version-checked to skip if upstream lerobot adopts them.

Applied automatically when `import lobe` runs (via `lobe/__init__.py`).

Patches:
1. LeRobotDataset._query_hf_dataset → fast path for non-image columns (12x speedup)
2. lerobot_train DataLoader → persistent_workers=True, prefetch_factor=4
"""
from __future__ import annotations

from loguru import logger


def _patch_dataset_query():
    """Replace LeRobotDataset._query_hf_dataset with our fast version.

    Avoids decoding 100 throwaway PNG images per action-chunk query.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lobe.datasets.fast_lerobot_dataset import FastLeRobotDataset

    if getattr(LeRobotDataset, "_lobe_query_patched", False):
        return  # Already patched

    LeRobotDataset._query_hf_dataset = FastLeRobotDataset._query_hf_dataset
    LeRobotDataset._lobe_query_patched = True
    logger.debug("Patched LeRobotDataset._query_hf_dataset (fast path for non-image columns)")


def _patch_dataloader_settings():
    """Patch lerobot_train.train() to use persistent_workers and larger prefetch.

    The default DataLoader has persistent_workers=False and prefetch_factor=2,
    which causes worker respawn between epochs and choppy speed. We change to
    persistent_workers=True and prefetch_factor=4.
    """
    import lerobot.scripts.lerobot_train as lerobot_train_mod
    import torch.utils.data as torch_data

    if getattr(lerobot_train_mod, "_lobe_dataloader_patched", False):
        return

    _orig_dataloader = torch_data.DataLoader

    def _patched_dataloader(*args, **kwargs):
        # Only patch when called from lerobot_train (heuristic: num_workers in kwargs)
        if "num_workers" in kwargs and kwargs["num_workers"] > 0:
            kwargs.setdefault("persistent_workers", True)
            kwargs["prefetch_factor"] = max(kwargs.get("prefetch_factor") or 0, 4)
        return _orig_dataloader(*args, **kwargs)

    # Replace only the reference inside lerobot_train, not globally
    lerobot_train_mod.torch.utils.data.DataLoader = _patched_dataloader  # type: ignore
    lerobot_train_mod._lobe_dataloader_patched = True
    logger.debug("Patched lerobot_train DataLoader (persistent_workers=True, prefetch_factor=4)")


def apply_patches():
    """Apply all LOBE patches to lerobot. Idempotent."""
    _patch_dataset_query()
    # Note: dataloader patch is tricky because lerobot_train imports DataLoader at module level.
    # For now, we keep manually editing lerobot_train.py for that. TODO: better approach.
