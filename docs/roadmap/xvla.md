# X-VLA — Technical Research Report

**Paper:** Zheng et al., *X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model*, arXiv 2510.10274 (Oct 2025).

**Sources consulted:**
- arXiv HTML (`https://arxiv.org/html/2510.10274`)
- GitHub: `https://github.com/2toinf/X-VLA`
- HF collection: `https://huggingface.co/collections/2toINF/x-vla`
- Project page: `https://thu-air-dream.github.io/X-VLA/`
- **Lerobot ships a complete first-party implementation** at
  `/home/sunlingfeng/lobe/.venv/lib/python3.13/site-packages/lerobot/policies/xvla/`
  (co-authored with `2toINF`, same author; Apache-2.0). This is the SAME X-VLA.

Most "unknown" values in the paper's prose are nailed down by the lerobot config
defaults, which are taken from the upstream repo.

---

## TL;DR

- **0.9B** soft-prompted transformer on top of a **frozen Florence-2-large** VLM.
- **Flow-matching action head**, EE6D action layout (xyz + 6D rot + gripper, padded to 20-d).
- Soft prompts are **per-domain embedding tables** queried by an integer `domain_id`
  and **concatenated to the token sequence** (not FiLM, not cross-attn, not LN modulation).
- Claims **98.1% average on LIBERO** — highest reported as of the paper.
- `lerobot>=0.5.1` already registers `xvla` as a policy type, with preprocessor,
  optimizer, and scheduler wired in. Integration into LOBE is a thin
  configuration layer, not a rewrite.

---

## 1. Architecture

### 1.1 Top-level composition

`XVLAModel` (`modeling_xvla.py:43`) = `Florence2ForConditionalGeneration` (VLM) +
`SoftPromptedTransformer` (action head).

```
 image(s) ──► Florence2 vision tower ──► image features
                                          │
 text    ──► BART tokenizer ──► BART encoder (Florence2 language_model.encoder)
                                          │
                                          ├─ merged image+text tokens → vlm_features  (main view)
                                          └─ raw vision features → aux_visual_inputs (other views)

 proprio + action_noisy + t  ──► DomainAwareLinear  ──► action_tokens
                                                            │
 [action_tokens, vlm_proj(vlm_features), aux_visual_proj(aux)] + pos_emb
                                                            │
                                  + soft_prompts[domain_id]  (concat at tail)
                                                            │
                                  24 × TransformerBlock (pre-LN, MHSA + MLP)
                                                            │
                                  LayerNorm → DomainAwareLinear → pred velocity
```

The language decoder and `lm_head` of Florence-2 are **deleted** at construction
time (`modeling_xvla.py:80-84`) — only the BART encoder is kept.

### 1.2 Transformer encoder (soft-prompted head)

From `configuration_xvla.py:72-79` (defaults match upstream):

| field | value |
| --- | --- |
| `hidden_size` | **1024** |
| `depth` | **24** |
| `num_heads` | **16** (head_dim 64) |
| `mlp_ratio` | **4.0** → FFN hidden **4096** |
| `max_len_seq` | 512 |
| positional embedding | learned `nn.Parameter(1, 512, 1024)`, std=0.02 |
| attention | PyTorch fused SDPA, `qkv_bias=True`, attn_drop=0.1 |
| MLP | GELU(tanh), drop=0.1 |
| LN | pre-LN per block (`soft_transformer.py:258-286`) |

Pure vanilla ViT-style block. No AdaLN, no cross-attention blocks, no RoPE.
Time conditioning is **not** injected via AdaLN — it's concatenated into the
action token features before the first projection (see §1.5).

### 1.3 Vision / language encoder — Florence-2-large

- Model: `microsoft/Florence-2-large` (Florence2 = DaViT vision + BART language).
- DaViT vision tower: `dim_embed = [256, 512, 1024, 2048]`, `depths = [1,1,9,1]`,
  `num_heads = [8,16,32,64]`, `projection_dim = 1024`
  (`configuration_florence2.py` defaults). Output token dim **1024**.
- Language side: BART encoder only (decoder deleted). Tokenizer is
  `facebook/bart-large`, `max_length=64` (`configuration_xvla.py:66-69`).
- Default in lerobot: **NOT frozen** (`freeze_vision_encoder=False`,
  `freeze_language_encoder=False`). Upstream paper describes VLM as frozen, but the
  code actually trains it end-to-end at a **10× lower LR** via the
  `XVLAAdamWConfig` differential-LR optimizer (`configuration_xvla.py:168-183`).
  This is the "reduced LR for VLM modules" from the paper, not true freezing.
- **Multi-view routing** (`modeling_xvla.py:157-192`): only `image_features[:, 0]` —
  the first view — is merged with text tokens and passed through the BART encoder,
  producing `vlm_features` (the "high-level VL stream"). All other views skip the
  language model and are flattened into `aux_visual_inputs` (the "auxiliary visual
  stream"). **This is the "disentanglement"**: the main/scene view gets full
  vision-language fusion; wrist/auxiliary views only get vision-tower features.

### 1.4 State encoder

- Proprio is padded to `max_state_dim=32` and **broadcast** across all action
  tokens (`soft_transformer.py:379-381`).
- It's concatenated with the noisy action and a sinusoidal time embedding
  (`dim_time=32`, `max_period=100`, `soft_transformer.py:183-209`), then fed
  through `DomainAwareLinear(dim_action + dim_time + dim_proprio → hidden_size,
  num_domains=30)` (`soft_transformer.py:338-340`).
- `DomainAwareLinear` (`soft_transformer.py:215-255`) stores a full
  `[num_domains, out*in]` weight table in an `nn.Embedding` — i.e. **each domain
  has its own linear projection**. This is how heterogeneous state/action
  dimensions across embodiments are absorbed without touching the backbone.

### 1.5 Action head and flow matching

- **Flow matching with rectified OT path**, per `modeling_xvla.py:214-229`:
  ```python
  t = (rand + arange/B) % (1-eps)                # quasi-random t ∈ [0,1)
  x_t = noise * t + action_gt * (1-t)            # straight-line interpolation
  pred = transformer(x_t, proprio, t, ...)        # model predicts clean action
  ```
  Note the direction: model output is the **clean action**, not a velocity —
  loss is action-space MSE/BCE on the prediction itself (see §1.6).
  Inference is iterative refinement:
  `x_{i-1} = x1 * (i/N) + pred * (1 - i/N)`, `N = num_denoising_steps = 10`
  (`configuration_xvla.py:84`, `modeling_xvla.py:252-267`).
- **Chunk length**: `chunk_size = 32` anchor actions. Paper text says "30 anchor
  points over 4s"; the code defaults to 32.
- **Action dim (model-side)**: 20, dual-arm EE6D layout
  (`action_hub.py:113-172`):
  `[pos3, rot6d, gripper, pos3, rot6d, gripper]` (arm1 channels 0-9, arm2 10-19).
  Gripper indices `(9, 19)` are zeroed in the input (`preprocess`) and sigmoid-ed
  at output (`postprocess`).
- **Loss** (`EE6DActionSpace.compute_loss`, `action_hub.py:133-158`):
  - Position: MSE × **500**
  - Rotation (6D): MSE × **10**
  - Gripper: BCE × **1**
  (AGIBOT variant uses MSE for gripper × 10 instead of BCE.)

### 1.6 Soft prompts — the central novelty

This is the clearest information-bearing part of the implementation:

`soft_transformer.py:343-408`

```python
self.soft_prompt_hub = nn.Embedding(num_domains, len_soft_prompts * hidden_size)
nn.init.normal_(self.soft_prompt_hub.weight, std=0.02)
...
soft_prompts = self.soft_prompt_hub(domain_id).view(B, len_soft_prompts, hidden_size)
x = torch.cat([x, soft_prompts], dim=1)   # appended at end of sequence
```

| property | value |
| --- | --- |
| mechanism | **concatenated as extra tokens** at the **end** of the sequence |
| tokens per prompt | `len_soft_prompts = 32` |
| token dim | `hidden_size = 1024` |
| total params per domain | 32 × 1024 = 32,768 floats |
| storage | single `nn.Embedding(num_domains, 32*1024)` ("the hub") |
| `num_domains` | **30** default slots (paper uses 7 distinct domains in pretraining) |
| initialization | Gaussian `N(0, 0.02²)` — random |
| lookup | integer `domain_id ∈ [0, num_domains)` per batch element |
| trainable | yes by default (`train_soft_prompts=True`); can be frozen |
| LR | optional separate LR scale + warmup (`optimizer_soft_prompt_lr_scale`, `optimizer_soft_prompt_warmup_lr_scale`) |

Crucially, **three** other components are also domain-conditioned via
`DomainAwareLinear`:
1. action encoder (`action_tokens → hidden`)
2. action decoder (`hidden → action`)
3. optional VLM/aux projections when `use_hetero_proj=True`

So "per-domain parameters" are not only the soft prompts — they are the entire
action-space I/O plus optional visual projections. The paper's claim of ~0.04%
extra params is for the soft-prompt hub; the domain-aware linears are an
additional per-domain cost the paper glosses over. Defaults set
`use_hetero_proj=False` so only encoder/decoder/prompts are per-domain.

**Domain selection at inference**: `XVLAPolicy._get_domain_id` reads
`batch["domain_id"]` (or a configurable `domain_feature_key`) and falls back to
**0** if absent (`modeling_xvla.py:338-359`). For the HF LIBERO checkpoint,
domain id is hard-coded via `XVLAAddDomainIdProcessorStep(domain_id=0)` in the
processor pipeline (`processor_xvla.py:424-467`, `546`). So **at inference the
prompt is chosen by the processor config, not by anything in the data**.

**Adaptation to a new embodiment**: the paper describes a "two-step adaptation":
1. warm-up: freeze backbone, train only the new prompt slot;
2. joint: unfreeze and co-train.

The config supports this via `freeze_vision_encoder`, `freeze_language_encoder`,
`train_policy_transformer`, `train_soft_prompts`
(`configuration_xvla.py:96-100`, `modeling_xvla.py:124-155`).

### 1.7 Parameter budget

- Florence-2-large: ~0.77 B
- `SoftPromptedTransformer`: 24 blocks × (4 × 1024² attn + 2 × 1024 × 4096 mlp)
  ≈ 24 × (4.2 M + 8.4 M) ≈ 300 M
- Action-space and soft-prompt overhead: ~1 M (for 30 domains)

Total: **~0.9 B**, consistent with the paper.

---

## 2. Training procedure

### 2.1 What's in the config (lerobot defaults)

`configuration_xvla.py:103-115`:

| field | default |
| --- | --- |
| optimizer | `XVLAAdamWConfig` (custom AdamW with per-group LR) |
| LR (head) | **1e-4** |
| LR (VLM) | **1e-5** (head LR × 0.1, hard-coded in `XVLAAdamWConfig`) |
| soft-prompt LR scale | 1.0 (optional warmup) |
| betas | (0.9, 0.99) |
| eps | 1e-8 |
| weight_decay | 0.0 |
| grad_clip_norm | 10.0 |
| scheduler | `CosineDecayWithWarmupSchedulerConfig` |
| warmup_steps | **1,000** |
| decay_steps | **30,000** |
| decay_lr | 2.5e-6 |
| dtype | float32 default; bf16 supported via `dtype="bfloat16"` |

The upstream `train.py` command referenced on GitHub README is:
```
accelerate launch --mixed_precision bf16 train.py \
    --models 2toINF/X-VLA-Pt \
    --learning_rate 1e-4 --learning_coef 0.1 \
    --iters 50000 --freeze_steps 1000 --warmup_steps 2000
```
So distributed training is **HuggingFace Accelerate**, mixed-precision **bf16**,
**50 k fine-tune iters** is the paper's default post-pretraining recipe.
`learning_coef 0.1` = the 10× lower VLM LR. `freeze_steps 1000` = one-cycle
backbone freeze during soft-prompt warmup.

### 2.2 Compute figures

- **Pretraining GPU count / hours / batch size / total steps**: *not disclosed*
  in the paper HTML, not in the HF model cards, not in the GitHub README. This
  is a real gap. The HF model cards only state bf16 + base model.
- LIBERO fine-tune: the reported table entry "~30 k steps" in our CLAUDE.md was
  rounded; the upstream `--iters 50000` is the actual default.
- Inference latency: not reported. Expect ~10 flow-matching NFEs × one
  transformer-head pass + one VLM pass per chunk of 32 actions. Empirically,
  on a single A100 this class of 0.9 B VLAs sits around **100-200 ms / chunk**,
  i.e. **5-10 chunks/s ≈ 160-320 actions/s**. Not verified.

### 2.3 Action normalization

**None at the policy level.** `configuration_xvla.py:56-62`:
```python
"VISUAL": IDENTITY
"STATE":  IDENTITY
"ACTION": IDENTITY
```
Instead, the action-space classes apply **hard-coded scale factors** in the
loss (`XYZ_SCALE=500`, `ROT_SCALE=10`, `GRIPPER_SCALE=1`). This assumes
actions are already in meters / 6D rot units. **Any LOBE dataset using X-VLA
must either adhere to this convention or adjust the scales** — this is a
gotcha when porting YAM data.

---

## 3. Training data (pretraining)

Paper text (HTML extraction): "290 K episodes across 7 hardware platforms and
5 robotic arm types (single-arm to bi-manual)." Datasets mentioned by name:
**DROID, RoboMind, AgiBot**. BridgeData-v2 also appears in model cards
(`X-VLA-WidowX`, `X-VLA-Libero` both list Bridge Data V2 as training source).

**Exact per-dataset episode counts are not published** — only the aggregate
"290 K episodes" and "7 domains".

Observations on cross-embodiment handling visible in the code:
- **Action dimension**: all embodiments are forced into the **20-D EE6D layout**.
  Unused channels (e.g. single-arm → arm2 slots) are zero-padded. Alternative
  action spaces are registered for joint-space (`joint`, `franka_joint7`,
  `so101_bimanual`, `auto`) but the paper uses `ee6d` throughout.
- **Camera setup**: arbitrary number of views; the first is the "main" view
  that gets fused with language, the rest become `aux_visual_inputs`. Missing
  views are padded with zero images and masked (`modeling_xvla.py:306-336`).
  Config option `empty_cameras` adds placeholder views explicitly.
- **State dimension**: per-domain linear projection via `DomainAwareLinear`
  absorbs arbitrary proprio layouts up to `max_state_dim=32`.
- **Language**: everything goes through BART tokenizer, `max_length=64`.

---

## 4. Benchmark results

### 4.1 LIBERO (per-suite success rates)

From paper HTML extraction:

| Model | Params | Spatial | Object | Goal | Long | **Avg** |
| --- | --- | --- | --- | --- | --- | --- |
| X-VLA | 0.9 B | 75.7 | 98.6 | 97.8 | 97.6 | **98.1** * |
| OpenVLA-OFT | 7 B | — | — | — | — | 97.1 |
| π₀ (pi0) | 3 B | — | — | — | — | 94.1 |
| SmolVLA | 2 B | — | — | — | — | 88.8 |

*The Spatial 75.7% vs 98.1% average is suspicious — this is straight from the
HTML fetch and likely a transcription error in the automatic extractor (98.1
average with one 75.7 component is arithmetically inconsistent; the real
Spatial is probably ≈98%). **Verify against the PDF before quoting.** The
lerobot card (`xvla-libero`) would be the authoritative source.

SmolVLA number here (88.8%) conflicts with our CLAUDE.md reference of 87.3% —
consistent within rounding / different eval protocols.

Per-baseline LIBERO split numbers (Spatial / Object / Goal / Long) were not
extractable from the HTML; the paper likely has them in a table but the web
scrape only pulled average values for baselines. **This is a gap to fill by
reading the PDF directly**.

### 4.2 SimplerEnv

- **Simpler-WidowX** (BridgeV2 subset): **80.4%** (new SOTA vs prior 78.0%).
- Google-Robot results: a checkpoint exists (`X-VLA-Google-Robot`) but no
  number extracted from HTML.

### 4.3 Calvin ABC→D

Checkpoint `X-VLA-Calvin-ABC_D` exists. Numbers not extracted.

### 4.4 Real robot (Soft-Fold dataset)

- **1,200 episodes** cloth-folding, Franka-based platform.
- Reported **~100% success rate**, **33 folds/hour** (~2 min/fold).
- Claimed on par with closed-source π₀-folding.
- Dataset: `Facebear/XVLA-Soft-Fold` on HF.

### 4.5 PEFT (LoRA fine-tuning)

- ~**9 M tunable params ≈ 1% of model**.
- LIBERO: **93%** avg.
- Simpler-WidowX: **54%**.
- HF repos: `X-VLA-libero-{spatial,object,goal,long}-peft`,
  `X-VLA-simpler-widowx-peft`.

### 4.6 Inference latency

**Not reported**. Expect ~10 NFEs × transformer-head pass per 32-action chunk
plus one Florence-2 forward. No measurement available.

---

## 5. Code structure

### 5.1 Upstream repo (`github.com/2toinf/X-VLA`)

| path | role |
| --- | --- |
| `train.py` | main entry, launched via `accelerate launch` |
| `models/modeling_xvla.py` | XVLA model + soft-prompted transformer |
| `models/configuration_xvla.py` | config dataclass |
| `datasets/` | dataset loaders |
| `datasets/domain_handler/registry.py` | per-dataset adapters (key integration point for cross-embodiment data) |

Config system: Python dataclass, no YAML/hydra. Hyperparameters are CLI flags
parsed with `argparse`/`accelerate`.

### 5.2 Lerobot first-party port

`.venv/lib/python3.13/site-packages/lerobot/policies/xvla/`

| file | contents |
| --- | --- |
| `configuration_xvla.py` | `XVLAConfig` (dataclass, registered as `"xvla"`) |
| `modeling_xvla.py` | `XVLAModel`, `XVLAPolicy` (LeRobot `PreTrainedPolicy` wrapper) |
| `soft_transformer.py` | `SoftPromptedTransformer`, `DomainAwareLinear`, timestep embedding |
| `action_hub.py` | `EE6DActionSpace`, `JointActionSpace`, `AGIBOTEE6DActionSpace`, `FrankaJoint7ActionSpace`, `AutoActionSpace`, `BimanualSO101ActionSpace` |
| `modeling_florence2.py` | Florence-2 backbone (vendored from transformers) |
| `configuration_florence2.py` | Florence-2 config |
| `processor_xvla.py` | Pre/post processors (tokenizer, image normalize, domain-id, rotate6d→axis-angle); `make_xvla_libero_pre_post_processors` is LIBERO-specific |
| `utils.py` | Rotation conversions |

Policy factory already wires X-VLA in:
- `lerobot/policies/factory.py:45` — imports `XVLAConfig`
- `lerobot/policies/factory.py:126-128` — `name == "xvla"` → `XVLAPolicy`
- `lerobot/policies/factory.py:180-181` — config resolution
- `lerobot/policies/factory.py:376-383` — processor pipeline

### 5.3 HF checkpoints (2toINF org)

Full models (0.9 B each):
- `X-VLA-Pt` — pretraining foundation (cross-embodiment, ~290 k episodes)
- `X-VLA-Libero` — LIBERO-finetuned
- `X-VLA-WidowX` — BridgeV2 / Simpler-WidowX
- `X-VLA-Google-Robot` — SimplerEnv Google-Robot
- `X-VLA-Calvin-ABC_D` — Calvin ABC→D
- `X-VLA-RoboTwin2` — RoboTwin2 sim
- `X-VLA-VLABench` — VLABench sim
- `X-VLA-AgiWorld-Challenge` — AgiWorld challenge
- `X-VLA-SoftFold` — real Franka cloth-folding

LoRA adapters (Nov 17):
- `X-VLA-libero-{spatial,object,goal,long}-peft`
- `X-VLA-simpler-widowx-peft`

LeRobot-shipped variants:
- `lerobot/xvla-base` (Mar 2)
- `lerobot/xvla-libero`, `lerobot/xvla-widowx`, `lerobot/xvla-folding`

---

## 6. Integration into LOBE

### 6.1 How much is already done

**A lot.** Because `lerobot>=0.5.1` ships X-VLA natively:

- `PreTrainedConfig.register_subclass("xvla")` means `XVLAConfig` is addressable
  by the string `"xvla"`.
- `make_policy(cfg, ds_meta)` in lerobot's factory already returns `XVLAPolicy`
  when `cfg.type == "xvla"`.
- Processor pipeline is complete (tokenizer, image norm, domain-id injection,
  rotate6d → axis-angle postprocess).
- Optimizer with per-group LR (`XVLAAdamWConfig`) and cosine schedule are
  registered in `lerobot.optim`.
- HF checkpoint loading works via `XVLAPolicy.from_pretrained(...)` with a
  custom remapping hook (`modeling_xvla.py:418-497`).
- LIBERO-specific processor (`make_xvla_libero_pre_post_processors`) is
  already wired.

### 6.2 What LOBE actually needs to add

The work is **configuration + one preset**, not implementation:

1. **`lobe/configs/base.py`**: add `XVLAPolicyConfig` dataclass that wraps
   `lerobot.policies.xvla.XVLAConfig`. Pattern: same as the existing
   `DiffusionPolicyConfig` wrapper. ~30-50 LOC.

2. **`lobe/policies/factory.py`** (or equivalent `create_policy`): add an
   `elif cfg.type == "xvla"` branch that instantiates
   `XVLAPolicy(lerobot_cfg)`. ~10 LOC.

3. **`lobe/configs/libero.py`**: add `libero-xvla` preset with:
   ```python
   XVLAConfig(
       chunk_size=32, n_action_steps=32,
       dtype="bfloat16",
       florence_config={...Florence2 defaults...},
       action_mode="ee6d",
       num_image_views=3,   # image + image2 + empty
       empty_cameras=1,
       resize_imgs_with_padding=(224, 224),  # verify
   )
   ```
   Use `XVLAPolicy.from_pretrained("2toINF/X-VLA-Pt")` as init.

4. **Training entry point**: `scripts/train_vla.py` already uses
   `lerobot-train`; adding `--model xvla` should just work if the config is
   registered. Alternatively `scripts/train.py libero-xvla` through our own
   preset dispatcher.

5. **Evaluation**: `lerobot-eval --policy.type=xvla ...` works natively on
   LIBERO once trained.

### 6.3 Soft-prompt lookup with lerobot datasets

This is the one non-trivial wrinkle. Lerobot's `LeRobotDataset` **does not
natively emit a per-sample domain ID**. There are three options, in order of
increasing effort:

1. **Hard-coded domain per dataset** *(easy, matches current HF checkpoints)*:
   set `XVLAAddDomainIdProcessorStep(domain_id=K)` in the preprocessor. Every
   sample in that dataloader gets domain `K`. Works for single-embodiment
   fine-tuning (LIBERO, YAM alone). This is exactly what
   `make_xvla_libero_pre_post_processors` does.

2. **Mixture dataloader with per-sub-dataset domain** *(medium)*: when
   combining YAM + LIBERO in one run, build a `MixedDataset` that wraps
   multiple `LeRobotDataset` instances, each tagged with its own domain id.
   Inject the id as a field in `complementary_data`. The XVLAPolicy already
   looks up `batch["domain_id"]` via `_get_domain_id`
   (`modeling_xvla.py:338-359`), so this Just Works at the model side. Only
   the data pipeline needs a thin wrapper (~100 LOC + a collator).

3. **Per-episode metadata** *(medium-hard)*: if a single LeRobot dataset
   actually contains multi-embodiment data (e.g. DROID-style), store a
   `domain_id` column in the episode metadata and add a processor step that
   reads it per batch. Requires a small LeRobotDataset schema extension.

For the stated LOBE goal ("training on YAM data + LIBERO simultaneously"),
**option 2** is what we want.

### 6.4 Pretrained checkpoint as init

Yes — `XVLAPolicy.from_pretrained("2toINF/X-VLA-Pt")` works today. The
`load_state_dict(strict=True)` call in `modeling_xvla.py:490` means you must
match the config exactly (hidden_size 1024, depth 24, heads 16, num_domains 30,
ee6d action mode, bart tokenizer). This is the recommended path: init from
`X-VLA-Pt`, then fine-tune on LIBERO or YAM by writing to a fresh domain slot.

### 6.5 Cross-embodiment "YAM + LIBERO" pipeline sketch

```
YAM dataset (lerobot format)     ──► domain_id = 1, ee6d action (6-D YAM cmd → 20-D pad)
LIBERO dataset (HuggingFaceVLA)  ──► domain_id = 0, ee6d action (7-D → 20-D pad)
                                   │
                                   ├── MixedDataset (interleaved, balanced sampling)
                                   │
                                   └── XVLAPolicy (init from X-VLA-Pt)
                                           - domain 0 prompt: reused from LIBERO ckpt
                                           - domain 1 prompt: fresh random, warm-up then joint
```

Training recipe: 2-step adaptation as in the paper
(`freeze_vision_encoder=True`, `train_policy_transformer=False`,
`train_soft_prompts=True` for 1 k steps, then unfreeze). All supported by
existing config flags.

### 6.6 Engineering effort estimate

Assumes the lerobot installation stays at ≥0.5.1:

| task | effort |
| --- | --- |
| Wrap `XVLAConfig` in a LOBE dataclass + register in union | 0.5 day |
| `libero-xvla` preset with Florence-2-large config dict | 0.5 day |
| Smoke test: 100-step LIBERO fine-tune from `X-VLA-Pt` | 0.5 day |
| Full LIBERO reproduction (50 k steps, multi-GPU via accelerate) | ~1 day compute + 0.5 day babysit |
| `lerobot-eval` sanity run on `libero_10` | 0.5 day |
| MixedDataset for YAM + LIBERO (new code) | **1.5-2 days** |
| Domain-prompt warmup scheduling (cfg already exists, just wire it) | 0.5 day |
| Action-scale calibration for YAM EE commands (the 500× pos scale) | 0.5 day |
| Documentation + experiments.tsv rows | 0.5 day |

**Total: ~5-7 working days** for a full "LOBE can train X-VLA on LIBERO + YAM
from the public pretrained checkpoint" milestone. The lerobot-shipped
implementation removes 80% of the risk — we are not implementing Florence-2 or
flow matching, only writing data plumbing and presets.

### 6.7 Gotchas

1. **Action scales are hard-coded** in `EE6DActionSpace` (xyz×500, rot×10). YAM
   data in meters/rad is fine; if YAM commands are in millimetres or
   normalized units, subclass the action space with different scales.
2. **Gripper channels** at indices 9 and 19 are zeroed in the input and sigmoid-ed
   in the output. Single-arm data must still populate index 9 (primary gripper);
   index 19 is unused.
3. **Domain id 0 is the fallback**. If you forget to set `domain_id`, the
   policy silently uses domain 0, which may be LIBERO's prompt in a
   LIBERO-finetuned checkpoint, contaminating your YAM run.
4. **Tokenizer mismatch risk**: `tokenizer_max_length=64` is short; LIBERO
   instructions fit fine, but very long YAM instructions may truncate.
5. **`normalization_mapping = IDENTITY`** — X-VLA does *not* use lerobot's
   dataset stats. Its loss is scale-based instead. Don't try to add STATE/ACTION
   normalization without also removing the hard-coded scales.
6. **Strict checkpoint loading**: config must match the public checkpoint's
   hyperparameters bit-for-bit, or `load_state_dict(strict=True)` will fail.
7. **VLM is *not* frozen by default** in the lerobot implementation, despite
   paper text to the contrary. If you want a true frozen-VLM ablation, set
   `freeze_vision_encoder=True, freeze_language_encoder=True`.

---

## 7. Open questions / things I could not confirm

- **Per-dataset pretraining episode breakdown** — DROID vs Bridge vs RoboMind
  vs AgiBot splits. Paper quotes only the 290 k total.
- **Pretraining GPU count, GPU type, total wall-clock / GPU-hours**. Neither
  paper HTML, HF model card, nor GitHub README disclose these.
- **Exact LIBERO per-suite baseline table** (Diffusion Policy, Octo, OpenVLA,
  pi0-FAST, SmolVLA split by Spatial/Object/Goal/Long). Paper has a table but
  the HTML extraction only returned averages for baselines, and the
  `Spatial=75.7, Avg=98.1` for X-VLA is internally inconsistent and likely a
  bad extraction. Needs a direct PDF read.
- **Inference latency / throughput** — not measured in the paper.
- **Official pretraining step count** — 50 k is the fine-tune default in the
  GitHub README; pretraining step count is not stated.
- **Number of effective soft-prompt domains in the released `X-VLA-Pt`
  checkpoint** — code has 30 slots, paper says 7 domains were used; which
  slots are populated is undocumented. Inspecting the safetensors would tell us.

---

## 8. One-line summary for CLAUDE.md status section

> X-VLA (0.9 B, Florence-2 + 24-layer soft-prompted transformer, flow-matching
> ee6d head) is fully implemented in `lerobot>=0.5.1`
> (`lerobot.policies.xvla`). LOBE integration is a config wrapper + preset
> (~1-2 days) plus a MixedDataset for cross-embodiment training (~2 days).
> Init from `2toINF/X-VLA-Pt`. Reported LIBERO avg 98.1% (paper).
