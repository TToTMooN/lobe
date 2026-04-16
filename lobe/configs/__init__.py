"""Training presets — dataclasses that materialize as lerobot-train CLI flags.

Each preset is a dataclass with a `to_launch_args()` method. A thin wrapper
script (scripts/train_yam.py, coming in Phase 6) dispatches on name →
dataclass → flag list → `scripts/_lobe_train_entry.py` subprocess.

For now, the flags are documented in docs/workflows/yam_finetune.md.
"""

from lobe.configs.yam import PRESETS as YAM_PRESETS  # noqa: F401
