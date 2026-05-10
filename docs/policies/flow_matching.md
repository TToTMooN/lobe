# Flow Matching (LOBE Custom)

Flow Matching policy with the same UNet architecture as Diffusion Policy, but using conditional flow matching instead of DDPM/DDIM.

## Differences from Diffusion Policy

| | Diffusion (DDPM) | Flow Matching |
|---|---|---|
| Forward process | `noisy = scheduler.add_noise(x, eps, t)` | `x_t = (1-t)*x + t*eps` (linear interp) |
| Loss target | Predict noise ε | Predict velocity `x - eps` |
| Inference | Iterative DDPM/DDIM steps | Euler ODE integration |
| Loss scale | Small (0.01–0.05) | Larger (0.3–0.5), different magnitude |
| Architecture | UNet (down_dims) | **Same UNet** |

The two should achieve **similar success rates** when given the same architecture and hyperparameters. Our head-to-head experiments confirm this:

| Model | Same large UNet, batch=256 | LIBERO 4-suite |
|---|---|---|
| Diffusion | ✓ | 36.5% |
| Flow Matching | ✓ | 33.75% |

(Both fail at this config — the issue is hyperparameter, not policy.)

## Training

```bash
lobe-train --policy.type=flow_matching \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --batch_size=8 --steps=100000 \
  --optimizer.lr=1e-4 \
  --policy.repo_id=fm-libero \
  --output_dir=/path/to/output
```

Multi-GPU works the same way as for built-in policies.

## Configuration

`FlowMatchingConfig` extends `PreTrainedConfig`. Key fields:

| Field | Default | Description |
|---|---|---|
| `n_obs_steps` | 2 | Observation steps to condition on |
| `horizon` | 16 | Action prediction horizon |
| `n_action_steps` | 8 | Actions to execute per query |
| `down_dims` | (256, 512, 1024) | UNet channel dims (use (512,1024,2048) for parity with DP) |
| `vision_backbone` | resnet18 | Image encoder |
| `crop_shape` | None | Optional image crop |
| `sigma` | 0.0 | OT path noise (0 = deterministic) |
| `num_inference_steps` | 10 | Euler ODE steps at inference |
| `optimizer_lr` | 1e-4 | Default LR |
| `normalization_mapping` | MEAN_STD all | Per-feature norm mode |

## Implementation files

- `lobe/policies/flow_matching/configuration_flow_matching.py` — Config dataclass with `@PreTrainedConfig.register_subclass("flow_matching")`
- `lobe/policies/flow_matching/modeling_flow_matching.py` — `FlowMatchingPolicy(PreTrainedPolicy)` and `FlowMatchingModel`
- `lobe/policies/flow_matching/processor_flow_matching.py` — `make_flow_matching_pre_post_processors()`
- `lobe/policies/flow_matching/flow_transformer.py` — DiT-style transformer backbone (alternative to UNet)
- `lobe/policies/flow_matching/vision_encoder.py` — ResNet18 + global pool

## Status

| Phase | Description | Status |
|---|---|---|
| Working | Trains with `lobe-train`, uses lerobot pipeline | ✓ |
| Working | Saves/loads via `pretrained_model/` checkpoint format | ✓ |
| Tech debt | Has internal `Normalize`/`Unnormalize` layers | 🚧 v1.0 refactor pending |
| Tech debt | Tied to custom `lobe.policies.normalize` module | 🚧 v1.0 refactor pending |

The v1.0 refactor will move normalization into the processor pipeline (matching how Diffusion Policy is structured) and delete the internal `Normalize` layers entirely.
