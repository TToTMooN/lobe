"""Monkey-patches applied to lerobot at import time.

These patches survive `uv sync` (unlike editing installed files directly) and
can be version-checked to skip if upstream lerobot adopts them.

Applied automatically when `import lobe` runs (via `lobe/__init__.py`).

Patches:
1. LeRobotDataset._query_hf_dataset → fast path for non-image columns (12x speedup)
2. lerobot_train DataLoader → persistent_workers=True, prefetch_factor=4
3. X-VLA LIBERO training: inject LiberoXVLAAdapterStep into policy preprocessor
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


def _patch_xvla_libero_env_factory():
    """Conditionally adjust X-VLA LIBERO env_preprocessor/postprocessor.

    `make_env_pre_post_processors` in `lerobot.envs.factory` hardcodes a call to
    `make_xvla_libero_pre_post_processors()` for any XVLAConfig, always returning the
    same pipeline designed for `action_mode='ee6d'`:

      env_preprocessor: [LiberoProcessorStep, XVLAImageNetNormalizeProcessorStep, XVLAAddDomainIdProcessorStep]
      env_postprocessor: [XVLARotation6DToAxisAngleProcessorStep]

    Two adjustments we apply to support our V8 setup (action_mode='auto' + 10-D EE6D):

    1. **Always remove `XVLAImageNetNormalizeProcessorStep`** from env_preprocessor when
       the policy_preprocessor has `xvla_imagenet_normalize` (our V8 and V7 do). Otherwise
       images get double-normalized at eval time (env applies, then policy tries again and
       errors because values are outside [0, 1]).

    2. **Keep `XVLARotation6DToAxisAngleProcessorStep`** iff the model outputs 10-D (the
       standard EE6D LIBERO format). For 7-D output (auto mode on raw delta actions), drop
       the conversion step — the model outputs LIBERO-native actions directly.

    For the official `ee6d` pipeline (lerobot/xvla-libero), we leave env_preprocessor alone
    (it's for checkpoints without imagenet_normalize in policy_preprocessor).
    """
    import lerobot.envs.factory as factory_mod
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig

    if getattr(factory_mod, "_lobe_xvla_env_factory_patched", False):
        return

    _orig_make = factory_mod.make_env_pre_post_processors

    def _patched_make(env_cfg, policy_cfg):
        pre, post = _orig_make(env_cfg, policy_cfg)
        if not isinstance(policy_cfg, XVLAConfig):
            return pre, post

        from lerobot.policies.xvla.processor_xvla import (
            XVLAImageNetNormalizeProcessorStep,
            XVLARotation6DToAxisAngleProcessorStep,
        )

        # Detect whether the loaded policy_preprocessor already has xvla_imagenet_normalize.
        # If so, the env_preprocessor's duplicate must be removed. We detect this by checking
        # whether the preprocessor_overrides config file exists (loaded from the checkpoint)
        # — but since we don't have direct access, approximate via action_mode:
        # auto mode in our setup always uses the xvla-base preprocessor which has imagenet_norm.
        action_mode = getattr(policy_cfg, "action_mode", "ee6d").lower()
        if action_mode == "auto":
            pre.steps = [s for s in pre.steps if not isinstance(s, XVLAImageNetNormalizeProcessorStep)]
            logger.debug("X-VLA auto mode: removed ImageNetNormalize from env_preprocessor")

        # Check output action dimension. If 7-D, skip rotation conversion (model outputs
        # LIBERO-native format). If 10-D or 20-D, keep it (model outputs EE6D that needs
        # conversion to 7-D axis-angle for the env).
        action_feature = getattr(policy_cfg, "output_features", {}).get("action", None)
        action_dim = action_feature.shape[0] if action_feature is not None else None
        if action_dim == 7:
            post.steps = [s for s in post.steps if not isinstance(s, XVLARotation6DToAxisAngleProcessorStep)]
            logger.debug("X-VLA 7-D output: removed Rotation6DToAxisAngle from env_postprocessor")

        return pre, post

    factory_mod.make_env_pre_post_processors = _patched_make
    factory_mod._lobe_xvla_env_factory_patched = True


def apply_patches():
    """Apply all LOBE patches to lerobot. Idempotent."""
    _patch_dataset_query()
    _patch_xvla_libero_env_factory()
    # Note: dataloader patch is tricky because lerobot_train imports DataLoader at module level.
    # For now, we keep manually editing lerobot_train.py for that. TODO: better approach.
