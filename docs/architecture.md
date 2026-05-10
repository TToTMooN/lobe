# Architecture

## Layout

```
lobe/
├── lobe/                                # Our package
│   ├── __init__.py                      # Auto-registers custom policies + applies patches
│   ├── cli.py                           # lobe-train, lobe-eval, lobe-serve entry points
│   ├── patches.py                       # Monkey-patches applied to lerobot on import
│   ├── video_compat.py                  # PyAV patch for torchvision VideoReader removal
│   ├── serve.py                         # WebSocket policy server (any lerobot policy)
│   ├── experiment_log.py                # experiments.tsv logger
│   ├── envs/                            # Per-env constants (FPS, dims) for serving
│   │   ├── pusht.py
│   │   ├── libero.py
│   │   └── yam_bimanual.py
│   ├── datasets/
│   │   └── fast_lerobot_dataset.py     # Optimized LeRobotDataset subclass
│   └── policies/
│       └── flow_matching/               # Custom Flow Matching policy
│           ├── configuration_*.py       # @register_subclass("flow_matching")
│           ├── modeling_*.py            # FlowMatchingPolicy(PreTrainedPolicy)
│           ├── processor_*.py           # Pre/post-processor pipeline
│           ├── flow_transformer.py
│           └── vision_encoder.py
├── lerobot_policy_flow_matching/        # Plugin discovery package
├── docs/                                # This documentation
├── BENCHMARKS.md                        # Benchmark protocol
├── CLAUDE.md                            # Project notes & first principles
├── experiments.tsv                      # Append-only experiment log
└── pyproject.toml                       # uv-managed dependencies
```

## Design

### Thin wrapper over lerobot

LOBE provides three CLI commands that all delegate to lerobot internals:

```python
# lobe/cli.py
def train():
    from lerobot.scripts.lerobot_train import main
    main()

def eval():
    from lerobot.scripts.lerobot_eval import main
    main()

def serve():
    from lobe.serve import main
    main()
```

The trick is what runs **before** these calls. `lobe/__init__.py` imports `lobe.patches` and the custom policy modules, registering everything with lerobot's class registry **before** lerobot's CLI code parses arguments. So when `lobe-train --policy.type=flow_matching ...` runs, lerobot already knows what `flow_matching` is.

### Patches via subclass + monkey-patch

We do not edit installed lerobot files (those are lost on `uv sync`). Instead:

1. **`lobe/datasets/fast_lerobot_dataset.py`** subclasses `LeRobotDataset` and overrides `_query_hf_dataset` with our 12× faster version (bypasses HF datasets `set_transform` for non-image columns).
2. **`lobe/patches.py`** monkey-patches `LeRobotDataset._query_hf_dataset` at import time, so any code that creates a `LeRobotDataset` (including lerobot-train and lerobot-eval) gets the optimization.

This is idempotent and can be version-checked to skip if upstream lerobot adopts the fix.

### Custom policy as first-class citizen

A custom policy (like our Flow Matching) becomes a first-class lerobot policy by:

1. Subclassing `PreTrainedConfig` with `@PreTrainedConfig.register_subclass("flow_matching")`
2. Subclassing `PreTrainedPolicy`
3. Providing a `make_*_pre_post_processors()` factory function

Once registered, it works with `lobe-train --policy.type=flow_matching ...`, `lobe-eval --policy.path=...`, and `lobe-serve --checkpoint=...` with no extra glue.

See [Adding new policies](policies/adding.md) for the full template.

### Serving

`lobe-serve` loads any lerobot-format checkpoint via `policy_cls.from_pretrained()`, runs it through the saved preprocessor pipeline, and exposes a WebSocket endpoint compatible with limb's `WebSocketPolicyClient` protocol.

Because the preprocessor and postprocessor are saved with the checkpoint, serving works for any policy type — the server doesn't need to know whether it's a Diffusion Policy, Flow Matching, SmolVLA, or pi0.

## What we deliberately did **not** build

| Anti-pattern | Why we avoided it |
|---|---|
| Custom training loop (`scripts/train.py`) | lerobot-train already does multi-GPU, eval integration, checkpointing |
| Custom dataset format (FastDataset .pt) | lerobot's dataset is fast enough after our patch |
| Custom config system (tyro dataclasses) | lerobot uses draccus, also fine, no reason to fork |
| Per-env evaluation harness | lerobot-eval covers PushT, LIBERO, MetaWorld, AlohaSim |
| Custom checkpoint format | use lerobot's `pretrained_model/` dir convention |

If you ever need a custom training loop (e.g. for RL fine-tuning, curriculum learning, or online learning), add a focused script for that specific use case rather than a general-purpose `train.py`.
