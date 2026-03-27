"""Environment registry — maps env names to modules with load_dataset() and evaluate()."""

from __future__ import annotations

import importlib
from types import ModuleType

ENV_REGISTRY: dict[str, str] = {
    "pusht": "lobe.envs.pusht",
    "libero": "lobe.envs.libero",
    "yam": "lobe.envs.yam_bimanual",
    "aloha": "lobe.envs.aloha",
}


def get_env(name: str) -> ModuleType:
    """Get an environment module by name."""
    if name not in ENV_REGISTRY:
        raise ValueError(f"Unknown env '{name}'. Available: {list(ENV_REGISTRY.keys())}")
    return importlib.import_module(ENV_REGISTRY[name])
