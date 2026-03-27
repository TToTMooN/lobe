"""Training configs — base dataclasses + per-env presets.

Adding a new environment:
    1. Create lobe/configs/myenv.py with a PRESETS dict
    2. Import it below — presets auto-merge into the global PRESETS

Adding a new policy type:
    1. Add a new dataclass in base.py (e.g. VLAPolicyConfig)
    2. Add it to the TrainPipelineConfig.policy union
    3. Use it in your env preset file
"""

# Import per-env presets and merge
from lobe.configs.aloha import PRESETS as ALOHA_PRESETS
from lobe.configs.base import (
    DiffusionPolicyConfig,
    EnvConfig,
    FMPolicyConfig,
    LoggingConfig,
    PerformanceConfig,
    TrainConfig,
    TrainPipelineConfig,
    WandbConfig,
)
from lobe.configs.libero import PRESETS as LIBERO_PRESETS
from lobe.configs.pusht import PRESETS as PUSHT_PRESETS

PRESETS: dict[str, tuple[str, TrainPipelineConfig]] = {**PUSHT_PRESETS, **ALOHA_PRESETS, **LIBERO_PRESETS}

__all__ = [
    "EnvConfig",
    "FMPolicyConfig",
    "DiffusionPolicyConfig",
    "TrainConfig",
    "PerformanceConfig",
    "LoggingConfig",
    "WandbConfig",
    "TrainPipelineConfig",
    "PRESETS",
]
