# Adding a new custom policy

This guide walks through adding a new policy that works with `lobe-train`, `lobe-eval`, and `lobe-serve`. Use the Flow Matching policy as the template.

## Step 1: Configuration

Create `lobe/policies/your_method/configuration_your_method.py`:

```python
from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig

@PreTrainedConfig.register_subclass("your_method")     # ← critical: registers as --policy.type
@dataclass
class YourMethodConfig(PreTrainedConfig):
    # Define hyperparameters as dataclass fields
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    optimizer_lr: float = 1e-4

    normalization_mapping: dict = field(default_factory=lambda: {
        "VISUAL": NormalizationMode.MEAN_STD,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    })

    def get_optimizer_preset(self):
        return AdamConfig(lr=self.optimizer_lr)

    def get_scheduler_preset(self):
        return DiffuserSchedulerConfig(name="cosine", num_warmup_steps=500)

    @property
    def observation_delta_indices(self):
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self):
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self):
        return None

    def validate_features(self):
        # Check input/output features match what the model expects
        pass
```

## Step 2: Model

Create `lobe/policies/your_method/modeling_your_method.py`:

```python
import torch
from torch import nn, Tensor
from lerobot.policies.pretrained import PreTrainedPolicy
from lobe.policies.your_method.configuration_your_method import YourMethodConfig

class YourMethodPolicy(PreTrainedPolicy):
    config_class = YourMethodConfig
    name = "your_method"

    def __init__(self, config: YourMethodConfig, dataset_stats=None, dataset_meta=None):
        super().__init__(config)
        config.validate_features()
        self.config = config
        # ... build your neural network here
        self.net = nn.Module()  # placeholder

    def get_optim_params(self):
        return self.net.parameters()

    def reset(self):
        pass  # reset internal state if any

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        # Training: compute loss
        # IMPORTANT: batch arrives pre-normalized by the processor pipeline
        loss = ...  # your loss
        return loss, None

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        # Inference: produce one action
        action = ...  # (batch, action_dim)
        return action

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        # Inference: produce a chunk of actions
        actions = ...  # (batch, horizon, action_dim)
        return actions
```

!!! warning "No internal normalization"
    Do **not** add `Normalize`/`Unnormalize` layers inside the model. The processor pipeline handles it. The batch arrives pre-normalized, and you return raw model outputs (the postprocessor unnormalizes).

## Step 3: Processor

Create `lobe/policies/your_method/processor_your_method.py`:

```python
from typing import Any
import torch
from lobe.policies.your_method.configuration_your_method import YourMethodConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

def make_your_method_pre_post_processors(
    config: YourMethodConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
):
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
```

## Step 4: Register on import

Edit `lobe/__init__.py`:

```python
from lobe.patches import apply_patches
apply_patches()

import lobe.policies.flow_matching.configuration_flow_matching  # noqa
import lobe.policies.flow_matching.modeling_flow_matching  # noqa
import lobe.policies.your_method.configuration_your_method  # noqa  ← add
import lobe.policies.your_method.modeling_your_method  # noqa  ← add
```

## Step 5: Test

```bash
# Should now work without any other changes
lobe-train --policy.type=your_method --dataset.repo_id=lerobot/pusht --steps=100
lobe-eval --policy.path=<checkpoint> --env.type=pusht
lobe-serve --checkpoint=<checkpoint>
```

That's it. The policy is now a first-class citizen of LOBE and lerobot.
