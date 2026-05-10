# Patches

LOBE applies a small set of monkey-patches to lerobot at import time. They live in `lobe/patches.py` and are applied automatically when you `import lobe` — which the `lobe-train`, `lobe-eval`, and `lobe-serve` CLI wrappers do via `lobe/cli.py`, and which `scripts/_lobe_train_entry.py` does for accelerate-launched multi-GPU training.

## Why patches?

Some lerobot internals have known issues that we either can't wait for upstream fixes on, or that are actually LOBE-specific conventions (not bugs). Editing the installed `.venv/lib/python3.13/site-packages/lerobot/...` files works but is fragile — `uv sync` wipes them. Monkey-patches survive `uv sync` and can be version-checked to skip if upstream adopts the fix.

## Patches applied (v1.0)

### 1. `LeRobotDataset._query_hf_dataset` — fast path for non-image columns

**Bug**: When querying action chunks (e.g. 30 future actions for one sample), lerobot calls `self.hf_dataset[key][indices]`. This triggers HF datasets' `set_transform`, which **decodes all columns** for those rows — including the ~30 PNG/JPEG images at the action-chunk indices that get immediately discarded (only the action column is kept). This is **~12× slower** than necessary.

**Fix** (`_patch_dataset_query`): For non-image columns, access the underlying Arrow table directly with `self.hf_dataset.data.column(key)`, bypassing `set_transform`. Image columns still go through the slow path because they actually need decoding.

**Impact**: Data loading goes from ~250 ms/sample to ~5 ms/sample. Combined with bf16, training speed was no longer data-bound.

**Idempotent**: tracks `LeRobotDataset._lobe_query_patched`.

### 2. `make_policy` — preserve trained action shape for XVLA

**Bug**: `lerobot.policies.factory.make_policy` unconditionally runs:

```python
cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
```

where `features` comes from the ENV (at eval time) or the dataset (at training time). For LIBERO at eval, this sets `cfg.output_features["action"].shape = (7,)` because the env's action_dim is 7 (`[xyz, axis_angle, gripper]`). When the XVLA model's `AutoActionSpace` then reads `config.action_feature.shape[-1]` to set `real_dim`, it gets 7 instead of the 10 the model was trained on. `AutoActionSpace.postprocess` then trims the 20-D raw model output to just the first 7 dims, which after the env-postprocessor's rot6d→axis-angle step gets interpreted as `[xyz, rot6d_col0, rot6d_col1[0]]` — garbage that the env misreads as `[xyz, axis_angle, gripper]`. **Eval silently scores 0% despite train loss 0.001.**

**Fix** (`_patch_xvla_make_policy_preserve_action_shape`): Wrap `make_policy`. For `XVLAConfig` with a `pretrained_path`, read the checkpoint's own `config.json` after `make_policy` returns the policy, and restore:

1. `policy.model.action_space.real_dim` to the checkpoint's saved action shape
2. `cfg.output_features["action"]` to a `PolicyFeature` with the checkpoint's saved shape

Now when `make_env_pre_post_processors` runs it sees the real shape, and at model inference `AutoActionSpace.postprocess` trims to the correct `real_dim`.

**Idempotent**: tracks `lerobot.policies.factory._lobe_xvla_make_policy_patched`.

### 3. `make_env_pre_post_processors` — strip double ImageNet normalization

**Bug**: For `XVLAConfig`, `make_xvla_libero_pre_post_processors()` adds `[LiberoProcessorStep, XVLAImageNetNormalizeProcessorStep, XVLAAddDomainIdProcessorStep]` to the env_preprocessor. But the policy_preprocessor inherited from `xvla-pt` (our `xvla-pt-v8` starter and similar) already has `xvla_imagenet_normalize` as a step. Both fire at eval, meaning images get normalized twice: `(img - mean)/std` applied once → values in roughly `[-2.1, 2.6]`, then applied again → garbage. The second normalization step actually errors out with "values outside [0, 1] range" and gets swallowed silently, leaving uninitialized normalized images.

**Fix** (`_patch_xvla_libero_env_factory`): Wrap `make_env_pre_post_processors`. When the policy is XVLA in `auto` mode (our convention for checkpoints derived from `xvla-pt`), strip `XVLAImageNetNormalizeProcessorStep` from `env_pre.steps` so only the policy_preprocessor's normalization fires.

**Idempotent**: tracks `lerobot.envs.factory._lobe_xvla_env_factory_patched`.

## Verifying patches

```bash
uv run python -c "
import lobe
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import lerobot.policies.factory as f
import lerobot.envs.factory as ef
print('query_patched:      ', getattr(LeRobotDataset, '_lobe_query_patched', False))
print('make_policy_patched:', getattr(f, '_lobe_xvla_make_policy_patched', False))
print('env_factory_patched:', getattr(ef, '_lobe_xvla_env_factory_patched', False))
"
```

All three should print `True`.

## Patches NOT in `lobe/patches.py`

These are distinct pitfalls you may hit and silently lose hours to — they're not lerobot bugs so patches wouldn't help, but they deserve naming:

1. **`TrainPipelineConfig.validate()` overwrites `self.optimizer` / `self.scheduler`** with the policy's presets (`get_optimizer_preset()` / `get_scheduler_preset()`) when `use_policy_training_preset=True` (the default). The presets read from `XVLAConfig.optimizer_*` / `XVLAConfig.scheduler_*` fields, not from the top-level `--optimizer.*` / `--scheduler.*` CLI flags you pass. **Always use `--policy.optimizer_*` / `--policy.scheduler_*` for XVLA fine-tuning.** `scripts/_lobe_train_entry.py` doesn't protect you from this.

2. **draccus tuple parsing**: `--policy.optimizer_betas 0.9 0.95` fails at CLI parsing. You can't set a tuple field from the command line in this lerobot build. Edit the checkpoint `config.json` directly if you need a non-default value.

See `docs/workflows/xvla_finetune.md` for the full X-VLA fine-tune recipe that uses these patches correctly.

## How to add a new patch

Edit `lobe/patches.py` and add a function:

```python
def _patch_my_thing():
    from lerobot.somewhere import SomeClass
    if getattr(SomeClass, "_lobe_my_patched", False):
        return  # idempotent
    SomeClass.some_method = my_better_version
    SomeClass._lobe_my_patched = True
    logger.debug("Patched SomeClass.some_method")
```

Then call it from `apply_patches()` (also in `patches.py`).

## How to retire a patch

If lerobot upstream adopts your fix, version-check and skip:

```python
def _patch_dataset_query():
    import lerobot
    if lerobot.__version__ >= "0.6.0":  # version where upstream fixed it
        return
    ...
```
