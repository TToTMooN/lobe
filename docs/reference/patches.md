# Patches

LOBE applies a small set of monkey-patches to lerobot at import time. They live in `lobe/patches.py` and are applied automatically when you `import lobe` (which all CLI commands do via `lobe/cli.py`).

## Why patches?

Some lerobot internals have known issues that we cannot wait for upstream fixes. Editing the installed `.venv/lib/python3.13/site-packages/lerobot/...` files works but is fragile — `uv sync` wipes them. Monkey-patches survive `uv sync` and can be version-checked to skip if upstream adopts the fix.

## Patches applied

### 1. `LeRobotDataset._query_hf_dataset` — fast path for non-image columns

**File**: `lobe/datasets/fast_lerobot_dataset.py` (the new method) and `lobe/patches.py:_patch_dataset_query` (the monkey-patch)

**Bug**: When querying action chunks (e.g. 50 future actions for one sample), lerobot calls `self.hf_dataset[key][indices]`. This triggers HF datasets' `set_transform`, which **decodes all columns** for those rows — including the 100 PNG images for the action chunk indices. The images are immediately discarded (only the action column is kept). This is **12× slower** than necessary.

**Fix**: For non-image columns, access the underlying Arrow table directly with `self.hf_dataset.data.column(key)`, bypassing `set_transform`. Image columns still go through the slow path because they actually need decoding.

**Impact**: Data loading goes from ~250 ms/sample to ~5 ms/sample. Combined with bf16, training speed went from 2 steps/s to 25 steps/s on FM (12.5×).

**Idempotent**: Tracks `LeRobotDataset._lobe_query_patched` flag to avoid double-patching.

### 2. DataLoader `persistent_workers + prefetch_factor` (currently disabled)

**Reasoning**: lerobot's DataLoader defaults are `persistent_workers=False, prefetch_factor=2`, which causes worker respawn between epochs and choppy speed. We want `persistent_workers=True, prefetch_factor=4`.

**Status**: We tried to monkey-patch this but lerobot imports `DataLoader` at module level, making it tricky to intercept cleanly. Currently we manually edit the installed `lerobot_train.py` for this. **TODO**: find a clean monkey-patch approach (or upstream fix).

## How to add a patch

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

## How to remove a patch

If lerobot upstream adopts your fix, version-check and skip:

```python
def _patch_dataset_query():
    import lerobot
    if lerobot.__version__ >= "0.6.0":  # version where upstream fixed it
        return
    ...
```

## Verifying patches

```bash
uv run python -c "
import lobe
from lerobot.datasets.lerobot_dataset import LeRobotDataset
print('Query patched:', getattr(LeRobotDataset, '_lobe_query_patched', False))
"
```

Should print `Query patched: True`.
