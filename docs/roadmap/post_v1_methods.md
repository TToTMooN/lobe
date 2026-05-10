# Post-v1.0: Methods to Integrate

After v1.0 stabilizes (Phases 1-4 complete), the next milestone is broadening method coverage. Two papers are queued for integration:

1. **SimVLA** ([arXiv 2602.18224](https://arxiv.org/abs/2602.18224)) — minimal VLA baseline
2. **X-VLA** ([arXiv 2510.10274](https://arxiv.org/abs/2510.10274)) — soft-prompt cross-embodiment VLA

Detailed technical research reports (after reading the actual code):

- [`simvla.md`](simvla.md) — SimVLA full technical dossier (architecture, training, ablations, integration plan)
- [`xvla.md`](xvla.md) — X-VLA full technical dossier (same)

Both are flow-matching-based VLAs with public code. Integration follows the same pattern we used for our `flow_matching` policy: register a config + model + processor, then `lobe-train --policy.type=...` works automatically.

## Lerobot ecosystem fit (key finding)

| | SimVLA | X-VLA |
|---|---|---|
| **Already in lerobot?** | ❌ no | ✅ **YES — first-party port** |
| **Code license** | Apache-2.0 | Apache-2.0 (lerobot port) |
| **VLM backbone** | SmolVLM-500M (Idefics3) | Florence-2-large |
| **Uses lerobot dataset format** | ❌ raw HDF5 | ✅ via lerobot processor pipeline |
| **Uses lerobot processor pipeline** | ❌ custom action norm + augs in their dataloader | ✅ standard `make_xvla_pre_post_processors` |
| **Uses lerobot optimizer/scheduler** | ❌ custom AdamW + 3 param groups in their train script | ✅ standard `get_optimizer_preset` / `get_scheduler_preset` |
| **Uses lerobot accelerate launch** | ✅ yes (their `train_smolvlm.py` is accelerate-based) | ✅ yes |
| **HF Hub checkpoint matches lerobot `pretrained_model/`** | ❌ raw `.pth` state dict, not `from_pretrained` compatible | ✅ yes — `lerobot/xvla-pt-base` and others on HF |
| **Cross-embodiment data plumbing** | n/a (single dataset) | ⚠️ needs integer `domain_id` field on every dataset sample, which lerobot does not natively expose |
| **Integration effort** | ~1 week (vendor their dataloader, write a wrapper) | ~1 week (config layer + per-dataset domain_id metadata) |

### What this means

**X-VLA fits the lerobot ecosystem natively.** Lerobot 0.5.1 ships a complete first-party port at `lerobot/policies/xvla/` (co-authored by the original X-VLA author `2toINF`). It already:
- Registers as `--policy.type=xvla` (we saw this in `PreTrainedConfig.get_known_choices()`)
- Has `XVLAConfig`, `XVLAPolicy`, `make_xvla_pre_post_processors()` — the standard trio
- Loads HF Hub checkpoints via `from_pretrained`
- Hooks into `lobe-train` / `lobe-eval` with no extra glue

The remaining work for X-VLA in LOBE is **just**: (a) figure out how to attach `domain_id` to LIBERO and YAM samples so the soft-prompt lookup works, (b) decide whether to fine-tune from `lerobot/xvla-pt-base` or train from scratch, (c) add a row to `BENCHMARKS.md`. **Effort: ~1 week of config and data plumbing, not a reimplementation.**

**SimVLA does NOT fit the lerobot ecosystem out of the box.** They wrote their own everything: dataloader (raw LIBERO HDF5, not the `HuggingFaceVLA/libero` repacking), optimizer (3 param groups with VLM LR×0.1, freeze-then-unfreeze schedule), action normalization (per-dim min-max stats computed offline), training loop (custom Accelerate driver), and even the inference path (they use `forward_vlm_efficient` which bypasses chat templates). The model itself is simple (SmolVLM + plain ViT action head with FM loss), but the **load-bearing details** that get them to 98.6% are scattered across their training script and dataloader, NOT the model code.

The integration question for SimVLA is: do we want to reimplement carefully (risk: miss a load-bearing detail like the Beta(1.5,1) noise sampling or trajectory shuffling, end up at 9.9% instead of 98.6%), or vendor their entire training script under `lobe/policies/simvla/vendored/` and only wrap the inference path for `lobe-serve`? **Recommendation: vendor it.** The SimVLA paper's whole point is that "tiny details" dominate; the only way to honor that is to keep their exact training driver intact.

---

## 1. SimVLA: Minimal VLA Baseline

| Property | Value |
|---|---|
| **Paper** | [arXiv 2602.18224](https://arxiv.org/abs/2602.18224) |
| **Project page** | https://frontierrobo.github.io/SimVLA |
| **Code** | https://github.com/LUOyk1999/SimVLA |
| **Checkpoints** | https://huggingface.co/collections/YuankaiLuo/simvla |
| **License** | CC BY-NC-SA 4.0 |
| **Params** | 0.5B |
| **Authors** | Yuankai Luo et al. |

### Method
- **Architecture**: Standard VLM encoder produces fused vision-language tokens **once per control step** + **lightweight action transformer** running flow-matching denoising
- **Decoupled perception/control**: VLM runs once, action head iterates internally
- **Action representation**: Continuous, normalized per-dimension, action chunks executed receding-horizon
- **Training emphasis**: standardized hyperparameters (LR sweep, warmup, VLM LR multiplier preserved); careful trajectory shuffling to avoid brittle long-horizon optimization

### What's novel
The core thesis: **architectural simplicity matters less than careful tuning**. SimVLA argues that small details (LR schedule, action norm, trajectory shuffling) drive most of the gains attributed to complex architectures. They claim to outperform multi-billion-param models with 0.5B params under "strictly matched evaluation."

### Why integrate
- **Strong baseline at small size** (0.5B vs SmolVLA 0.45B, similar scale)
- **Reproduction-friendly**: minimal architecture means easy debugging
- **Hypothesis test for our work**: if SimVLA's "carefully tuned baseline" hits ~85% on LIBERO, it confirms our hypothesis that hyperparameters dominate over policy type (FM vs DP within same arch)
- **Decoupled perception/control** is a clean pattern we can study

### Integration plan

**Estimated effort**: 1-2 days. SimVLA is simple enough that we may be able to subclass `SmolVLAPolicy` and just swap the action head.

**Steps**:
1. **Read SimVLA code**: `git clone https://github.com/LUOyk1999/SimVLA && ls`
2. **Identify the action head module** (the lightweight transformer doing flow matching)
3. **Identify the VLM backbone they use** — likely SmolVLM2 or similar (for clean weight loading from HF)
4. **Create `lobe/policies/simvla/`**:
   - `configuration_simvla.py`: dataclass with `@PreTrainedConfig.register_subclass("simvla")`. Fields: `vlm_backbone_name`, `action_horizon`, `n_inference_steps`, `action_dim`, etc.
   - `modeling_simvla.py`: `SimVLAPolicy(PreTrainedPolicy)` with `forward()` and `predict_action_chunk()`. The model wraps a HF `AutoModel` for the VLM and a custom `ActionHead` module.
   - `processor_simvla.py`: copy from `processor_flow_matching.py` (same pattern)
5. **Add to `lobe/__init__.py`**: `import lobe.policies.simvla.configuration_simvla`
6. **Test**: `lobe-train --policy.type=simvla --dataset.repo_id=HuggingFaceVLA/libero --steps=200`
7. **Reproduce LIBERO**: train with their exact config and compare to their reported numbers
8. **Add to BENCHMARKS.md** as a new row

**Risks**:
- They use a custom action representation (chunked, normalized) that may not match lerobot's `observation.state` / `action` convention out of the box. Some adapter glue likely needed.
- Their checkpoint format may differ from `pretrained_model/`. May need a one-time conversion script.
- License is CC BY-NC-SA — fine for research but means we cannot use SimVLA-derived weights commercially.

**Open questions to answer first**:
- Exact LIBERO-4-suite numbers (paper has them in figures, need to read PDF)
- Training compute used (how many GPU-hours)
- Does their HF checkpoint reproduce when loaded into their code?

---

## 2. X-VLA: Cross-Embodiment Soft-Prompt VLA

| Property | Value |
|---|---|
| **Paper** | [arXiv 2510.10274](https://arxiv.org/abs/2510.10274) |
| **Project page** | https://thu-air-dream.github.io/X-VLA/ |
| **Code** | https://github.com/2toinf/X-VLA |
| **Checkpoints** | https://huggingface.co/collections/2toINF/x-vla |
| **License** | CC BY-NC-ND 4.0 |
| **Params** | 0.9B |
| **LIBERO** | **98.1%** (already in our published baseline table) |
| **Authors** | Jinliang Zheng et al. (Tsinghua, Shanghai AI Lab) |

### Method
- **Architecture**: Standard Transformer encoders + flow-matching action head
- **Soft prompts**: Per-data-source learnable embedding vectors. Each dataset gets its own prompt. At training time, the model conditions on the data source's prompt; at inference time you pick the prompt for the target embodiment.
- **Disentangled streams**: Separates high-dimensional inputs (images) from low-dimensional inputs (state, goals) for training stability
- **Cross-embodiment**: Trained on heterogeneous data from many robots (single embodiment doesn't scale data, so they pool across embodiments)

### What's novel
**Soft prompts as embodiment adapters** — instead of having separate models per robot or training one giant model that has to handle every embodiment in its weights, they use a small per-robot prompt vector that "tells the shared backbone which robot it's controlling". This is the same idea as prompt tuning in NLP, applied to robotics.

### Why integrate
- **#1 published result on LIBERO** (98.1%, beats pi0.5's 97.5%)
- **Cross-embodiment training** is genuinely useful for limb (we'll have multiple robots)
- **Soft prompt mechanism** is simple to implement and could transfer to our other policies
- **Concrete recipe for scaling**: their data mixture and training schedule are documented

### Integration plan

**Estimated effort**: 3-5 days. X-VLA is more complex than SimVLA — soft prompts add machinery and the cross-embodiment data loading is non-trivial.

**Steps**:
1. **Read X-VLA code**: `git clone https://github.com/2toinf/X-VLA`
2. **Understand the soft-prompt mechanism**: how prompts are added to the Transformer (prepended tokens? added to layer norms? FiLM-style?)
3. **Create `lobe/policies/xvla/`**:
   - `configuration_xvla.py`: needs `n_embodiments`, `prompt_dim`, `prompt_init`, `flow_inference_steps`, plus standard PreTrainedConfig fields. Register as `@PreTrainedConfig.register_subclass("xvla")`.
   - `modeling_xvla.py`: `XVLAPolicy(PreTrainedPolicy)`. Holds a learnable `nn.Embedding(n_embodiments, prompt_dim)` and a Transformer encoder. `forward()` looks up the prompt for the current batch's data source and passes it through.
   - `processor_xvla.py`: needs to track **which embodiment each sample came from**. This is the tricky part — lerobot datasets don't have an embodiment field by default. We'd need to add it via the dataset's `info.json` or via a metadata processor step.
4. **Cross-embodiment dataset loader**: optional for v1 — could just train on a single dataset first (skip soft prompts) and add cross-embodiment later.
5. **Add to `lobe/__init__.py`**
6. **Test on LIBERO** (single embodiment) first, then see if we can add YAM data alongside.

**Risks**:
- **Cross-embodiment data plumbing**: lerobot's dataset abstraction doesn't natively expose "which dataset/embodiment did this sample come from". We'd either patch the dataset to expose it, or include the embodiment id in the action chunk metadata.
- **License is CC BY-NC-ND**: most restrictive of the bunch — no derivatives, no commercial use. We can run it for research but cannot release modifications.
- **Training data is large** — full reproduction of the 98.1% LIBERO requires their pretraining data mix, which may be hundreds of GB. Fine-tuning their released checkpoint on LIBERO is more realistic.

**Open questions to answer first**:
- Does their HF checkpoint reproduce 98.1% on LIBERO when loaded fresh? (The known SmolVLA reproduction gap suggests we should not trust published numbers blindly.)
- How are soft prompts initialized? (Random? From a base prompt?)
- Can we use a pretrained X-VLA backbone and only train the soft prompts on a new dataset? (This would be the "lightweight adaptation" use case.)

---

## Comparison

| | SimVLA | X-VLA |
|---|---|---|
| Params | 0.5B | 0.9B |
| LIBERO published | TBD (need PDF) | **98.1%** |
| Architecture | VLM + action head | Transformer + soft prompts |
| Novelty | Simplicity / careful tuning | Soft prompts for cross-embodiment |
| License | CC BY-NC-SA | CC BY-NC-ND (no derivatives) |
| Code | github.com/LUOyk1999/SimVLA | github.com/2toinf/X-VLA |
| Integration effort | 1-2 days | 3-5 days |
| Why integrate first | Sanity check our hypothesis (tuning > arch) | Strongest published LIBERO |
| Risk | Medium (custom action format) | High (cross-embodiment plumbing, license) |

## Recommended order

1. **Finish v1.0 first** — Phases 1, 2, 3, 4 done; remaining: pin lerobot commit, write release notes
2. **Add SimVLA next** — easier integration, useful as a baseline calibration
3. **Add X-VLA second** — bigger payoff but more work; can defer the cross-embodiment soft prompts and just integrate the architecture first
4. **Iterate on hyperparameters** — once both are integrated, run them with our patched data loader and bf16 to see if we can match published numbers

## How adding a new policy fits in our v1.0 architecture

Both new policies will follow the **exact same template** as our `flow_matching` policy:

```
lobe/policies/<name>/
├── configuration_<name>.py    # @PreTrainedConfig.register_subclass("<name>")
├── modeling_<name>.py         # <Name>Policy(PreTrainedPolicy)
└── processor_<name>.py        # make_<name>_pre_post_processors()

# Add to lobe/__init__.py:
import lobe.policies.<name>.configuration_<name>  # noqa
import lobe.policies.<name>.modeling_<name>  # noqa
```

After registration, they work with:
- `lobe-train --policy.type=<name> --dataset.repo_id=...`
- `lobe-eval --policy.path=<checkpoint>`
- `lobe-serve --checkpoint=<checkpoint>` (with RTC if it's a flow-matching VLA)

This is the v1.0 promise: **adding a new method is a one-time integration cost, not a per-experiment cost**.
