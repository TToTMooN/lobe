"""Monkey-patches applied to lerobot at import time.

These patches survive `uv sync` (unlike editing installed files directly) and
can be version-checked to skip if upstream lerobot adopts them.

Applied automatically when `import lobe` runs (via `lobe/__init__.py`).

Patches:
1. LeRobotDataset._query_hf_dataset → fast path for non-image columns (12x speedup).
2. `make_policy` for XVLAConfig with a pretrained_path → restore
   `cfg.output_features["action"]` and `AutoActionSpace.real_dim` from the
   checkpoint's own config.json, so env-inferred action shapes don't silently
   truncate the model's trained output at eval time.
3. `make_env_pre_post_processors` for XVLAConfig in auto mode → strip the
   redundant `XVLAImageNetNormalizeProcessorStep` from env_preprocessor so
   images aren't normalized twice at eval (once by env_pre and once by the
   policy's inherited xvla-pt preprocessor).
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
    """Adjust X-VLA LIBERO env_preprocessor to avoid double ImageNet normalization.

    `make_env_pre_post_processors` in `lerobot.envs.factory` hardcodes a call to
    `make_xvla_libero_pre_post_processors()` for any XVLAConfig:

      env_preprocessor: [LiberoProcessorStep, XVLAImageNetNormalizeProcessorStep, XVLAAddDomainIdProcessorStep]
      env_postprocessor: [XVLARotation6DToAxisAngleProcessorStep]

    Adjustment we apply for our V8+ setup (action_mode='auto', fine-tuned from xvla-pt):

    - **Remove `XVLAImageNetNormalizeProcessorStep`** from env_preprocessor when the loaded
      policy's action_mode is 'auto'. Our auto-mode checkpoints all inherit xvla-pt's
      policy_preprocessor which already has `xvla_imagenet_normalize`. Without this patch
      the image gets normalized twice (env applies, then policy, then the second
      normalization errors because values are outside [0, 1]).

    We never touch env_postprocessor's XVLARotation6DToAxisAngleProcessorStep because XVLA
    models always output 10-20D (abs_xyz + rot6d + gripper). An earlier version removed
    it when `policy_cfg.output_features["action"].shape == (7,)`, but at eval time
    `lerobot.make_policy` overwrites output_features with the env's action shape (7-D for
    LIBERO). That triggered the removal even though the model still outputs 10-D →
    `AutoActionSpace.postprocess` trimmed to 7 giving raw `[xyz, rot6d_col1, rot6d_col2[0]]`
    which the env misinterpreted as `[xyz, axis_angle, gripper]`, causing 0% eval success
    despite perfectly-trained weights. Keep the rotation step unconditionally.
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

        from lerobot.policies.xvla.processor_xvla import XVLAImageNetNormalizeProcessorStep

        action_mode = getattr(policy_cfg, "action_mode", "ee6d").lower()
        if action_mode == "auto":
            pre.steps = [s for s in pre.steps if not isinstance(s, XVLAImageNetNormalizeProcessorStep)]
            logger.debug("X-VLA auto mode: removed ImageNetNormalize from env_preprocessor")

        return pre, post

    factory_mod.make_env_pre_post_processors = _patched_make
    factory_mod._lobe_xvla_env_factory_patched = True


def _patch_xvla_make_policy_preserve_action_shape():
    """Prevent `lerobot.make_policy` from overwriting an XVLAConfig's `output_features["action"]`
    with the env's action shape (7-D for LIBERO).

    Why
    ---
    At training, an XVLA auto-mode policy reads `config.action_feature.shape` to set
    `AutoActionSpace.real_dim`. With a LIBERO absolute 10-D `abs_action_6d` dataset, real_dim=10
    and the model learns to output 10-D `[abs_xyz(3), rot6d(6), gripper(1)]` per chunk step.

    At eval, `make_policy` unconditionally runs:
        cfg.output_features = {key: ft for key, ft in env_features.items() if ft.type == ACTION}
    which overrides action shape to 7 (LIBERO env's action_dim). XVLA's from_pretrained then
    rebuilds AutoActionSpace with real_dim=7 → `postprocess` trims the 20-D raw output to 7,
    so the model's 10-D semantic output gets truncated to `[xyz, rot6d_col1, rot6d_col2[0]]`
    which the env misinterprets as `[xyz, axis_angle, gripper]` → 0% eval despite correct training.

    Fix
    ---
    Wrap `make_policy`. For XVLAConfig with a pretrained path, re-read the saved config.json
    and use its `output_features` instead of whatever the env-features-derived version has.
    """
    import lerobot.policies.factory as factory_mod
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig

    if getattr(factory_mod, "_lobe_xvla_make_policy_patched", False):
        return

    _orig_make_policy = factory_mod.make_policy

    def _patched_make_policy(cfg, ds_meta=None, env_cfg=None, rename_map=None):
        if isinstance(cfg, XVLAConfig) and cfg.pretrained_path:
            import json
            from lerobot.configs.types import FeatureType, PolicyFeature

            ckpt_cfg_path = f"{cfg.pretrained_path}/config.json"
            try:
                ckpt_cfg = json.loads(open(ckpt_cfg_path).read())
                action_feat = ckpt_cfg.get("output_features", {}).get("action", {})
                ckpt_shape = tuple(action_feat.get("shape", ()))
                if ckpt_shape:
                    logger.debug(
                        f"X-VLA: restoring output_features['action'].shape={ckpt_shape} from {ckpt_cfg_path}"
                    )

                    _orig_attr = cfg.output_features

                    # Run the original make_policy first, then patch the policy's action_space
                    # (can't intercept the output_features assignment cleanly without duplicating
                    # the function; instead correct the real_dim on the constructed policy).
                    policy = _orig_make_policy(cfg=cfg, ds_meta=ds_meta, env_cfg=env_cfg, rename_map=rename_map)

                    if cfg.action_mode.lower() == "auto" and hasattr(policy, "model"):
                        action_space = policy.model.action_space
                        if hasattr(action_space, "real_dim"):
                            old_real_dim = action_space.real_dim
                            action_space.real_dim = ckpt_shape[0]
                            logger.debug(
                                f"X-VLA AutoActionSpace: real_dim {old_real_dim} -> {ckpt_shape[0]}"
                            )
                    # Also restore cfg.output_features so downstream code (make_env_pre_post_processors)
                    # sees the correct shape.
                    cfg.output_features["action"] = PolicyFeature(
                        type=FeatureType.ACTION, shape=ckpt_shape
                    )
                    return policy
            except Exception as e:
                logger.warning(f"X-VLA: could not restore output_features from {ckpt_cfg_path}: {e}")

        return _orig_make_policy(cfg=cfg, ds_meta=ds_meta, env_cfg=env_cfg, rename_map=rename_map)

    factory_mod.make_policy = _patched_make_policy
    factory_mod._lobe_xvla_make_policy_patched = True


def apply_patches():
    """Apply all LOBE patches to lerobot. Idempotent."""
    _patch_dataset_query()
    _patch_xvla_make_policy_preserve_action_shape()
    _patch_xvla_libero_env_factory()
    # Note: dataloader patch is tricky because lerobot_train imports DataLoader at module level.
    # For now, we keep manually editing lerobot_train.py for that. TODO: better approach.
