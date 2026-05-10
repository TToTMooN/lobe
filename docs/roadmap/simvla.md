# SimVLA — Technical Summary and LOBE Integration Plan

> **Paper**: Luo et al., *SimVLA: A Simple VLA Baseline for Robotic Manipulation*,
> arXiv:2602.18224 (Feb 2026).
> **Code**: https://github.com/LUOyk1999/SimVLA (Apache-2.0)
> **Weights**: https://huggingface.co/YuankaiLuo/SimVLA-LIBERO (0.8B total, ~10.5k dl)
> **Project page**: https://frontierrobo.github.io/SimVLA

This doc is a ground-truth technical dossier — all architecture / training
details below are cited to specific source files in the authors' repo (cloned
at commit `main` on 2026-03-31), not paraphrased from the abstract. Where the
paper/code is silent we say "not specified".

---

## TL;DR

SimVLA is an intentionally minimal VLA: a frozen-ish `SmolVLM-500M-Instruct`
emits fused vision-language tokens **once per step**, and a plain **ViT-style
transformer action head** performs conditional flow-matching denoising.
**No cross-attention, no FiLM, no Q-Former, no action tokenizer.** The paper's
central claim is that with this simple recipe plus careful data shuffling,
action normalization, and a low VLM learning-rate multiplier, you get
**98.6 % average on LIBERO** at 0.5 B params and 9.3 GB training VRAM —
beating π0.5 (3 B, 96.9 %), OpenVLA-OFT (7 B, 97.1 %) and VLA-Adapter
(0.5 B, 97.3 %).

---

## 1. Architecture

### 1.1 VLM backbone
- Exact model: **`HuggingFaceTB/SmolVLM-500M-Instruct`** (Idefics3-family),
  loaded via `AutoModelForImageTextToText`.
  *Source:* `models/modeling_smolvlm_vla.py:70–78`.
- Vision tower = **SigLIP** (SmolVLM's built-in `vision_model`);
  connector = SmolVLM's `model.connector` (multi-modal projector).
- Text tower hidden size (queried at init) = **576**;
  `models/modeling_smolvlm_vla.py:82`.
- **Not** loaded with bfloat16 at construction — explicitly
  `torch_dtype=torch.float32` "for training stability"
  (`modeling_smolvlm_vla.py:72`). bf16 comes from `accelerate launch
  --mixed_precision bf16` wrapping it at runtime.
- Image resolution: **384 × 384** for published LIBERO runs
  (`train_smolvlm_{small,large}.sh`), though code supports 512.
- Language prompt: raw task string passed through SmolVLM tokenizer,
  padded to `max_length=50`. *Source:* `processing_smolvlm_vla.py:46,119–143`.
  No chat template is used during training; the efficient path
  `forward_vlm_efficient` concatenates raw vision-patch embeddings with raw
  text embeddings and runs them through `self.vlm.model.text_model` directly
  (`modeling_smolvlm_vla.py:202–321`). The chat-template path
  (`forward_vlm`) exists but is only used when language_instruction is
  passed positionally, and is effectively an inference-only fallback.

### 1.2 Action head
Implemented by class `SmolVLMActionTransformer` in
`models/transformer_smolvlm.py`.

Two modes are supported:
- **Concat mode** (default, used for all published results,
  `use_adaln=False`): plain pre-LN Transformer with LayerNorm, SDPA,
  `nn.GELU(approximate="tanh")` MLPs, `qkv_bias=True`,
  `mlp_ratio=4.0`, attn dropout **0.1**, MLP dropout **0.1**.
  (`transformer_smolvlm.py:162–179`.)
- **DiT / AdaLN mode** (`use_adaln=True`): DiT-style blocks with
  `adaLN_modulation` projecting the fused condition to 6×H for
  shift/scale/gate on MSA and MLP. *Not used* in the released recipe;
  both `train_smolvlm_*.sh` scripts ship `USE_ADALN=false`.

Dimensions (defaults from `configuration_smolvlm_vla.py` and the shell
scripts):

| Variant | hidden_size | depth | num_heads | mlp_ratio | Params |
|---------|-------------|-------|-----------|-----------|--------|
| Small   | 768         | 12    | 12        | 4.0       | ~80 M  |
| Large   | 1024        | 24    | 16        | 4.0       | ~302 M |

*Published LIBERO numbers use the Large variant* (comment
`# Estimated parameters: ~302M` in `train_smolvlm_large.sh`).

Other head hyper-params:
- `dim_time = 32` (sinusoidal time embedding dim for concat mode)
- `max_len_seq = 512` (learned 1-D positional embedding, init σ=0.02)
- `dim_action = 7` (LIBERO delta), `dim_proprio = 8`
- `num_actions` (action-chunk length) = **10** for LIBERO.

### 1.3 How perception and control are connected — **token concat +
self-attention only**

In the concat mode (the mode used for LIBERO results), the input to the
action transformer is built in `_forward_concat`
(`transformer_smolvlm.py:361–400`):

1. Build per-step token: `concat([noised_action_t, proprio_broadcast,
   time_emb_broadcast], dim=-1)` → Linear → `H`-dim tokens of shape
   `[B, T_action=10, H]`. Proprio and time are **broadcast along the
   action-chunk axis** so every denoising token sees them — no extra
   proprio token, no extra time token.
2. VLM tokens: `vlm_proj(vlm_features)` — a single `nn.Linear(576, H)` —
   projecting the fused last-hidden-state of the text model over the
   `[vision_patches … text_tokens]` sequence (variable length per batch,
   zero-padded).
3. **Concatenate along the sequence dim** into a single `[B, T_action +
   T_vlm, H]` sequence, add learned positional embeddings, run through
   depth×`TransformerBlock`, LayerNorm, then `action_decoder` (a single
   `nn.Linear(H, dim_action)`) applied **only to the first `T_action`
   tokens**.

Key properties:
- **Pure self-attention between action tokens and VLM tokens** — no
  cross-attention, no FiLM, no AdaLN in the published run.
- VLM features are not pooled; the full padded sequence is exposed as
  KV for every action token.
- **No causal mask** — action tokens attend freely to each other and to
  all VLM tokens.
- VLM tokens are discarded at the output linear.

Ablations (paper §Ablations, Table 6 row block): concat mode beats
cross-attention and AdaLN; AdaLN is included in the codebase but
disabled.

### 1.4 Flow matching
Implemented inline in `SmolVLMVLA.forward` (`modeling_smolvlm_vla.py:324–384`)
and `generate_actions` (`386–441`).

Training loss — conditional FM with **Beta time sampling** and
**velocity target `noise − action`**:

```python
t ~ Beta(1.5, 1.0) * 0.999 + 0.001        # line 350
x_t = t * noise + (1 - t) * action_norm   # line 370
u_t = noise - action_norm                 # line 371 (target velocity)
v_t = transformer(vlm, x_t, t, proprio)
loss = mean((v_t - u_t)**2)
```

Notes:
- This is the π0 / SmolVLA parametrization (`x_0 = action`, `x_1 =
  noise`, `v = x_1 − x_0`), **not** the Lipman OT-CFM `x_1 − x_t`
  closed form. Equivalent up to sign convention.
- Time sampling is **Beta(1.5, 1)** — skewed toward high noise
  (SmolVLA / π0 style), **not** uniform. This is a nontrivial detail
  that is not called out in the README but is load-bearing.
- Time is clipped to `[0.001, 0.9991]` — never exactly 0 or 1.
- **No loss weighting / SNR weighting** — raw `mean((v_t - u_t)**2)`.

Inference — **Euler integration**:

```python
x = randn(B, 10, 7)
dt = -1.0 / steps                 # default steps = 10
t = 1.0
while t > -dt/2:
    v = transformer(vlm, x, t, proprio)
    x = x + dt * v
    t += dt
return action_space.postprocess(x)  # unnormalize
```

Default **`steps = 10`** (Euler, constant step size). The VLM is
forwarded **once** and its features cached across all denoising steps —
`enc = self.forward_vlm_efficient(...)` is called before the while loop
(`modeling_smolvlm_vla.py:406`).

### 1.5 Action chunking
- Chunk length `num_actions = 10` for LIBERO
  (`train_smolvlm_large.sh:44`, `NUM_ACTIONS=10`).
- The action head predicts all 10 future actions **in parallel** (not
  autoregressively) — every chunk position has its own token in the
  action transformer.
- **No chunk stitching / temporal ensembling** is implemented in the
  released eval code. `evaluation/libero/serve_smolvlm_libero.py` simply
  returns the whole predicted chunk to the client per request
  (`serve_smolvlm_libero.py:176–186`), and `libero_client.py` presumably
  executes it open-loop (standard "receding horizon: execute k, replan"
  pattern). No exponential-moving-average a la ACT.

### 1.6 Language conditioning
- **No separate text encoder**, no CLIP. The raw language instruction
  is tokenized with `SmolVLMVLAProcessor.encode_language` — a thin
  wrapper over SmolVLM's tokenizer — to `[B, 50]` `input_ids`
  (`processing_smolvlm_vla.py:119–143`).
- In the efficient training forward (`forward_vlm_efficient`), the
  `input_ids` are embedded with SmolVLM's text-model input embeddings,
  concatenated **after** the vision patches for each sample, passed
  through SmolVLM's full text LM (`self.vlm.model.text_model`), and the
  resulting last hidden state is treated as the fused multimodal
  feature stream (`modeling_smolvlm_vla.py:265–321`).
- During training there is **no chat template, no system prompt, no
  `<image>` sentinels, no generation prompt**. SimVLA bypasses SmolVLM's
  normal prompting interface entirely and does manual embedding surgery
  on the LM input. This is a consequence of wanting a single fixed
  token layout per batch.

### 1.7 Proprio and normalization
- Proprio = **8-D** for LIBERO: `[ee_pos(3), ee_ori_axisangle(3),
  gripper_states(2)]`. Euler from the LIBERO HDF5 is converted to
  axis-angle at load time (`datasets/domain_handler/libero_hdf5.py:56–77,
  200–215`).
- Action = **7-D** delta: `[Δxyz(3), Δeuler(3), gripper_cmd(1)]`,
  already in `[-1, 1]` per LIBERO convention.
- Normalization scheme is **selected at action-space construction**
  (`models/action_hub.py:157–256`, `LiberoJointActionSpace`):
  - Default: **per-dim z-score** from `norm_stats/libero_norm.json`
    (shipped in the repo — means/stds precomputed across all LIBERO
    suites). `compute_libero_norm_stats.py` is the script that produces
    this file.
  - Optional: **quantile `[q01, q99] → [-1, 1]`** (set
    `use_quantile_norm=True`). Not enabled by default.
  - State and actions use **separate** stats dicts.
- Normalization is applied inside the model forward
  (`modeling_smolvlm_vla.py:353–365`), not in the dataloader, so the
  raw batch contains un-normalized actions.
- **Proprio normalization is on by default** and is one of the two
  things whose ablation causes near-collapse (paper Table 6).

### 1.8 Multi-view handling
- Up to `num_views = 3`; LIBERO uses **2 views**: `agentview_rgb`
  (third-person) and `eye_in_hand_rgb` (wrist). Both are rotated 180°
  before feeding to the VLM
  (`datasets/domain_handler/libero_hdf5.py:242,249` — `img[::-1, ::-1]`).
- The third view slot is zero-padded and masked out with `image_mask`.
- Views are processed by SigLIP **independently** (flattened to
  `[B·V, C, H, W]`, `transformer_smolvlm.py`-adjacent flow in
  `modeling_smolvlm_vla.py:240–277`), then their patch tokens are
  concatenated per-sample into a single vision-feature stream, which is
  then concatenated with text embeddings and fed through the LM.
  The VLM therefore sees something like
  `[patches_view0, patches_view1, text_tokens]` with no view-marker
  tokens.

---

## 2. Training procedure

All numbers below are from `train_smolvlm_large.sh` (the configuration
used for the 302 M / 98.6 % result) unless noted.

### 2.1 Optimizer
- **`torch.optim.AdamW`** (`train_smolvlm.py:29,188`).
- Betas: **`(0.9, 0.95)`** (GPT-style, not Adam default 0.999)
  (`train_smolvlm.py:104`).
- Eps: **not specified** — PyTorch default `1e-8`.
- Weight decay: **0.0** (`train_smolvlm.py:103`). Yes, zero.
- Grad clip: **`max_grad_norm=1.0`** (`train_smolvlm.py:105,409`).

### 2.2 Param groups and VLM LR multiplier
`build_optimizer` creates **three** param groups (`train_smolvlm.py:170–188`):

| Group            | Contents                                           | LR           |
|------------------|----------------------------------------------------|--------------|
| `action_heads`   | `transformer.action_encoder` + `transformer.action_decoder` (or `final_layer`) | `--learning_rate` |
| `transformer_core` | rest of `SmolVLMActionTransformer` (blocks, norms, vlm_proj, pos_emb) | `--learning_rate` |
| `vlm`            | **all** `model.vlm.parameters()` (SigLIP + LM + connector) | `--learning_rate * --learning_coef` |

- **VLM LR multiplier** (`--learning_coef`): **`0.1`** by default in the
  shell scripts. This is **the ablation that matters most** — Table 6
  shows `learning_coef=1.0` collapses to 44.2 % avg while `0.1` gets
  98.6 %. Setting it to 0.0 is equivalent to freezing.
- **Freeze-then-unfreeze schedule**: for the first `--freeze_steps=1000`
  **both** `vlm` and `transformer_core` groups have LR = 0 and **only
  the action head** (`action_encoder` + `action_decoder`) is trained.
  After step 1000 all three groups get their base LR simultaneously
  (`train_smolvlm.py:216–237`). So the VLM is not frozen throughout
  training — it is frozen for 1000 warmup-equivalent steps then
  joint-finetuned with a 10× lower LR.

### 2.3 LR schedule
- Peak LR (action_heads / transformer_core): **`2e-4`** for the Large
  variant, **`1e-4`** for the Small variant.
- Peak VLM LR: `peak * 0.1` = `2e-5` / `1e-5`.
- Warmup: **`--warmup_steps = 0`**. (The `linear_warmup_cosine` helper
  exists in the code but with `warmup_steps=0` is a no-op.)
- Decay: **`--use_cosine_decay`** flag is **NOT passed** in either shell
  script → schedule is effectively constant-after-freeze for
  `action_heads`/`transformer_core`/`vlm`.
  (`train_smolvlm.py:230–237`.)
- Min LR ratio: 0.1 if cosine were enabled.
- So the effective recipe is: 1000 steps with only the action head at
  `2e-4`, then 199 000 steps of constant joint training with
  `core=2e-4, vlm=2e-5`, no decay, no warmup.

This matches Table 6: "Learning rate is the dominant knob; warmup and
schedule are secondary."

### 2.4 Batch size, GPUs, steps
- `--batch_size = 64` **per process**.
- `accelerate launch --num_processes=4` → **4 GPUs, global batch = 256**
  (`train_smolvlm_large.sh:109–114`). The shell scripts hard-code
  `CUDA_VISIBLE_DEVICES=0,1,2,3` (small) or `4,5,6,7` (large), i.e.,
  **one 8-GPU node, two parallel runs**.
- Gradient accumulation: **none** (not configured in argparse, not used
  in training loop).
- Total iterations: `--iters = 200_000` (Large), same for Small.
- Wall-clock: **not reported** in paper/README. Given 200k steps ×
  batch-256 on 4 GPUs at 384×384 with SmolVLM-0.5B + 302 M head, a
  reasonable rough estimate is **~2–3 days on 4×A100 80 GB** (not a
  verified number; your mileage will vary).
- GPU-hours / $: **not reported**.

### 2.5 Mixed precision and memory
- **bf16 mixed precision** via `accelerate launch --mixed_precision bf16`
  (shell scripts). Model is constructed in fp32; gradients accumulated
  in bf16 under the Accelerate autocast wrapper.
- **Peak training VRAM = 9.3 GB** at `batch_size=8` (Table 1 — this is
  the "single-GPU memory" column, *not* the published 4×64 = 256
  batch). At the published batch=64 per GPU the actual footprint will
  be higher but still orders of magnitude less than OpenVLA-OFT
  (62.0 GB) or π0.5 (51.3 GB). The comparison in Table 1 is explicitly
  at fixed small batch for a fair memory reading.
- **No gradient checkpointing** (the flag `supports_gradient_checkpointing
  = True` exists on the class but isn't enabled in the training loop).
- **No DeepSpeed, no FSDP** — plain Accelerate DDP with
  `find_unused_parameters=True` (`train_smolvlm.py:257`).

### 2.6 Action / data augmentation and shuffling
- Image aug = `torchvision.transforms.ColorJitter(brightness=0.2,
  contrast=0.2, saturation=0.2, hue=0.0)` during training only, after
  resize to 384, before ImageNet mean/std normalization
  (`datasets/dataset_smolvlm.py:119–136`). No random crop, no rotation,
  no MixUp.
- **Trajectory shuffling** — the paper's other "load-bearing" detail —
  is implemented in two places:
  1. `SmolVLMDataReader._iter_one_dataset` shuffles trajectory indices
     on every epoch (`dataset_smolvlm.py:141–144`).
  2. `LiberoHDF5Handler._iter_demo` shuffles *intra-trajectory frame
     indices* (`libero_hdf5.py:221–223`), so within a trajectory, action
     chunks are emitted in random temporal order rather than left-to-right.
  Ablation (Table 6): disabling shuffling drops Avg from 98.6 → 9.9 %.
  Yes, really — it's catastrophic. The IterableDataset
  design means a non-shuffled run converges on one trajectory at a time,
  which the model simply cannot learn from.
- Language augmentation: supported via `lang_aug_map` but empty for
  LIBERO (stock task descriptions are used).
- Worker-side TF env is nuked to prevent GPU conflicts
  (`dataset_smolvlm.py:320–338`).

---

## 3. Training data

- **No pretraining on OXE / DROID / Bridge.** The VLM (`SmolVLM-500M-
  Instruct`) is loaded **directly from HuggingFace** and fine-tuned on
  LIBERO only. This is explicitly mentioned on the project page and is
  verified by the absence of any OXE/Bridge loading code in the repo.
- LIBERO training data used = **libero_10 + libero_goal + libero_object +
  libero_spatial + libero_90**
  (`train_smolvlm_large.sh:68`, `create_libero_meta.py` subset list).
  Note: the Large script includes `libero_90` (the 90-task pretraining
  suite) in the training mix; the README's "Prepare" step also lists
  only the four eval suites, so there is a slight inconsistency — the
  released scripts use `libero_90` but the README doesn't instruct you
  to. **If you reproduce, include libero_90.**
- One single generalist model is trained on the union. No per-task
  fine-tuning.
- Demos per task: **50** (LIBERO default). With libero_90 added, total
  episodes ≈ 90 × 50 + 4 × 10 × 50 = ~6 500 episodes, ~650k–1M env
  steps, ~2–4 GB on disk as 128×128 HDF5.

---

## 4. Benchmark results

### 4.1 LIBERO — main result (Table 2 in paper)

| Model         | Params | Spatial | Object | Goal | Long | **Avg** |
|---------------|-------:|--------:|-------:|-----:|-----:|--------:|
| Diffusion Policy (pub.) | — | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| OpenVLA       | 7 B    | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SmolVLA       | 0.45 B | — | — | — | — | **87.3** |
| π0-FAST       | 3 B    | — | — | — | — | 82.5 |
| π0.5          | 3 B    | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| OpenVLA-OFT   | 7 B    | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| VLA-Adapter   | 0.5 B  | 97.8 | 99.2 | 97.2 | 95.0 | 97.3 |
| **SimVLA (ours)** | **0.5 B** | **99.6** | **99.8** | **98.6** | **96.4** | **98.6** |

SimVLA wins every suite. The Long-horizon suite (libero_10) is the one
where larger models historically dominate — SimVLA edges VLA-Adapter by
1.4 points there.

### 4.2 Efficiency (Table 1)

| Model          | Params | Avg | Train VRAM (GB, bs=8) |
|----------------|-------:|----:|----------------------:|
| OpenVLA-OFT    | 7 B    | 97.1 | 62.0 |
| π0.5           | 3 B    | 96.9 | 51.3 |
| VLA-Adapter    | 0.5 B  | 97.3 | 24.7 |
| **SimVLA**     | **0.5 B** | **98.6** | **9.3** |

SimVLA is **2.7× lighter at peak memory than the next-best 0.5 B model**
and **6.7× lighter than π0.5**.

### 4.3 SimplerEnv

- **WidowX**: 95.8 % average; 100/100 on *Put Spoon on Towel* and *Put
  Eggplant in Basket*.
- **Google Robot**: 76.1 % average across Pick / Move / Open tasks.

### 4.4 Real robot — Galaxea R1 Lite
Zero-shot (no real-robot fine-tuning). The paper reports SimVLA
"broadly comparable to π0.5 under the same zero-shot protocol", with
typical success rates ~80 % on multi-stage manipulation. No numerical
tables released for this setup.

### 4.5 Ablations worth internalizing (Table 6)

| Knob turned off           | LIBERO Avg | Δ vs default |
|---------------------------|-----------:|-------------:|
| (default)                 | 98.6       | —            |
| Shuffling off             | 9.9        | **−88.7**    |
| Action normalization off  | 12.3       | −86.3        |
| VLM LR multiplier = 1.0   | 44.2       | −54.4        |
| LR = 5e-5                 | 90.6       | −8.0         |
| LR = 1e-4                 | 95.5       | −3.1         |
| LR = 5e-4                 | 72.7       | −25.9        |

Takeaway: **three cliff edges**: shuffling, normalization, and VLM LR
scaling. Architecture (concat vs cross-attn vs AdaLN, Small vs Large)
moves the needle by < 2 points by comparison.

### 4.6 Inference latency
Not reported in the paper or README. From the code path
(`generate_actions`, `modeling_smolvlm_vla.py:386–441`) we know:
- VLM forward: 1× per control step (no caching across control steps).
- Action-head forward: **10× per control step** (Euler, steps=10).
- Action chunk executed = 10 actions per VLM forward.

On a single A100 80 GB with bf16, a rough back-of-envelope for 302 M
head + 0.5 B VLM at 384×384 is ~50–100 ms per control step end-to-end,
or ~5–10 ms per executed action given the chunking. **Not verified.**

---

## 5. Code structure (author's repo)

Root = https://github.com/LUOyk1999/SimVLA

```
SimVLA/
├── train_smolvlm.py               # main training entry (argparse + accelerate)
├── train_smolvlm_large.sh         # recipe: 1024/24/16, LR 2e-4, 200k steps
├── train_smolvlm_small.sh         # recipe: 768/12/12, LR 1e-4, 200k steps
├── create_libero_meta.py          # builds datasets/metas/libero_train.json
├── compute_libero_norm_stats.py   # produces norm_stats/libero_norm.json
├── norm_stats/libero_norm.json    # shipped z-score stats
├── models/
│   ├── configuration_smolvlm_vla.py   # SmolVLMVLAConfig (PretrainedConfig)
│   ├── modeling_smolvlm_vla.py        # SmolVLMVLA (PreTrainedModel) — FM loss here
│   ├── transformer_smolvlm.py         # SmolVLMActionTransformer (concat + DiT)
│   ├── action_hub.py                  # LiberoJointActionSpace + norm stats
│   └── processing_smolvlm_vla.py      # SmolVLMVLAProcessor (tokenizer + img)
├── datasets/
│   ├── dataset_smolvlm.py             # IterableDataset + create_smolvlm_dataloader
│   ├── domain_config.py               # DATA_WEIGHTS (all 1.0)
│   └── domain_handler/
│       ├── libero_hdf5.py             # LiberoHDF5Handler (raw HDF5 reader)
│       └── registry.py                # get_handler_cls(dataset_name)
├── evaluation/
│   └── libero/
│       ├── serve_smolvlm_libero.py    # WebSocket policy server (msgpack_numpy)
│       ├── libero_client.py           # env stepper, runs on a separate GPU
│       └── run_eval_all.sh            # parallel 4-suite eval (4 GPUs)
└── requirements.txt                   # essentially empty (2 lines)
```

Config system: **plain argparse flags** driven by shell scripts. No yaml,
no hydra, no dataclass configs. 19 flags in `get_args_parser`
(`train_smolvlm.py:78–158`).

Dataset: **custom**. Does **not** use lerobot format. Reads raw LIBERO
HDF5 files directly via `h5py`. The format expected is the upstream
`Lifelong-Robot-Learning/LIBERO` release, not the `HuggingFaceVLA/libero`
repacking.

Inference server: **WebSocket + msgpack_numpy**, not OpenPI /
lerobot-eval. `libero_client.py` is a custom stepper that owns the
LIBERO env and talks to `serve_smolvlm_libero.py` over the wire.

Checkpoints on HF Hub (as of 2026-03-31):

| Repo                         | Size  | Trained on         |
|------------------------------|------:|--------------------|
| `YuankaiLuo/SimVLA-LIBERO`   | 0.8 B | LIBERO (10/90/goal/object/spatial) |

The 0.8 B = 0.5 B SmolVLM backbone + 0.3 B Large action transformer,
saved via `PreTrainedModel.save_pretrained(safe_serialization=True)` —
produces a standard `model.safetensors` + `config.json` + `state.json`
(custom, stores `global_step`). Reloadable with
`SmolVLMVLA.from_pretrained(path)`. **No SimVLA-Small released.** No
SimplerEnv or Galaxea checkpoints released.

---

## 6. Porting into LOBE — engineering plan

### 6.1 What "add SimVLA as a policy" means in LOBE's config system

LOBE policies currently live under `lobe/policies/` (e.g.
`flow_matching/modeling_flow_matching.py`), with a tyro dataclass config
in `lobe/configs/base.py` and a `create_policy()` factory in
`lobe/policies/factory.py`. To expose SimVLA as
`--policy.type=simvla` we need:

1. `lobe/configs/base.py`: add `SimVLAConfig` dataclass (vlm path,
   hidden_size, depth, num_heads, num_actions, learning_coef,
   freeze_steps) and add it to the `TrainPipelineConfig.policy` union.
2. `lobe/configs/libero.py`: add a `libero-simvla` preset mirroring
   `train_smolvlm_large.sh` (batch 64, LR 2e-4, learning_coef 0.1,
   freeze_steps 1000, num_actions 10, image_size 384, 200k steps).
3. `lobe/policies/simvla/`:
   - `modeling_simvla.py` — wraps the author's `SmolVLMVLA`.
   - `configuration_simvla.py` — thin mapping from our dataclass to
     `SmolVLMVLAConfig`.
4. `lobe/policies/factory.py`: wire the new `policy_type`.
5. `scripts/train.py`: the existing `train_vla.py` path already uses
   `accelerate`, but SimVLA is **not** a lerobot policy — it can't be
   trained via `lerobot-train`. The cleanest option is to reuse
   `scripts/train.py` (our own training loop) and add a SimVLA-specific
   collate/dataloader path.

### 6.2 Can we just vendor their code?

**Yes, mostly.** The repo is Apache-2.0. The clean path is:

- Copy `models/modeling_smolvlm_vla.py`, `models/transformer_smolvlm.py`,
  `models/configuration_smolvlm_vla.py`, `models/processing_smolvlm_vla.py`,
  `models/action_hub.py` into `lobe/policies/simvla/vendored/`.
- Change `from datasets import ...` to `from lobe.policies.simvla.vendored
  ...` (their code imports their own `datasets/` package, which shadows
  HuggingFace `datasets` — we will want to rename or move).
- Write a thin `LobeSimVLA` wrapper that exposes the lerobot `Policy`
  interface (`select_action`, `reset`, `forward`) around `SmolVLMVLA`.

### 6.3 Data pipeline — the nontrivial part

SimVLA's dataloader reads raw LIBERO HDF5 (`obs/agentview_rgb`,
`obs/eye_in_hand_rgb`, `actions`, `obs/ee_pos`, `obs/ee_ori`,
`obs/gripper_states`), converts Euler→axis-angle, normalizes at model
forward time, and emits samples as an **IterableDataset** that shuffles
both trajectories *and* intra-trajectory frame indices.

LOBE currently uses `lerobot.common.datasets` with the `HuggingFaceVLA/libero`
repacking (2 cameras, pre-normalized). Two options:

- **(A) Vendor their dataloader verbatim** and point it at raw LIBERO
  HDF5 (requires downloading the upstream LIBERO dataset ~2 GB). Least
  risky, 1 day of work.
- **(B) Rewrite the dataloader on top of LeRobot `LeRobotDataset`** to
  emit the same format. Must preserve the *trajectory + frame shuffling
  semantics* — the ablation shows this is not optional. 2–3 days of work
  and a real correctness risk.

Given the paper's catastrophic shuffling ablation (−88.7 %), **option A
is strongly recommended for first repro**.

### 6.4 Checkpoint format
- SimVLA ships a standard HF `PreTrainedModel` safetensors dir. We can
  `SmolVLMVLA.from_pretrained(...)` it directly. **No conversion to
  `lerobot_pretrained_model/` needed** unless we want `lerobot-eval` to
  load it, which it can't (the class isn't registered with lerobot).
- For our own serving path (`lobe/serve.py`) we can wrap the HF model
  as-is; it's already designed around a FastAPI `/act` endpoint on the
  model itself (`modeling_smolvlm_vla.py:443–508`) — we can ignore
  that and plug into LOBE's WebSocket server.

### 6.5 Does the action head depend on a specific tokenizer/processor?
- **Yes**, it depends on SmolVLM's tokenizer for language encoding and
  SmolVLM's image processor for patch sizes. The action head itself is
  tokenizer-agnostic (it takes `[B, T, 576]` features), but switching
  away from SmolVLM-500M means rebuilding the processor and changing
  `vlm_hidden_size=576`.
- The model internally uses `SmolVLM.model.connector` and
  `SmolVLM.model.text_model.get_input_embeddings()`, which assumes the
  `Idefics3`/SmolVLM module layout. Swapping to e.g. Qwen2-VL would
  require rewriting `forward_vlm_efficient`.

### 6.6 Can we eval with `lerobot-eval --env.type=libero`?
No, not without writing a lerobot-`Policy` adapter class. SimVLA's
native eval loop is a custom WebSocket server + client pair. The
cleanest fast path is to call `SmolVLMVLA.generate_actions(...)` from
inside a lerobot `Policy` subclass in our wrapper, then use
`lerobot-eval` as usual with our `--rename_map` for the two LIBERO
cameras. **~1 day of adapter work.**

### 6.7 Honest effort estimate

| Step | Effort | Risk |
|------|-------:|------|
| Vendor model code (`models/`, fix imports) | 2 h | low |
| Wrap as `lobe` policy (`LobeSimVLA`, config dataclass, factory entry) | 3 h | low |
| Raw LIBERO HDF5 dataloader path (option A above) | 6 h | low |
| LeRobot dataloader path preserving frame shuffling (option B) | 2 d | medium — shuffling ablation is load-bearing |
| Training-loop integration with `scripts/train.py` (AdamW groups, freeze_steps, learning_coef) | 4 h | low — exact pattern is in `train_smolvlm.py:170–237` |
| lerobot-`Policy` adapter for `lerobot-eval` | 1 d | low |
| First full LIBERO reproduction run (4×A100, ~2–3 days wall-clock) | 3 d | medium — expect to hit one of the three cliff edges |
| Debugging the first-run failure (likely shuffling or norm stats) | 1 d | — |
| Total engineering | **~1 week** | |
| Total wall-clock to first reproduced number | **~1.5 weeks** | |

The safest plan is: vendor their code with minimal changes, reproduce
the LIBERO result on their own dataloader to confirm the bits are
intact, **then** swap in the LeRobot dataloader. Doing both at once
will make any drop impossible to attribute.

### 6.8 Known unknowns / things to double-check before launching
- **Wall-clock training time is not reported.** Our 4×A100-80 GB budget
  may need `batch_size=32` per GPU instead of 64. If so, halve the LR
  to `1e-4` (Small recipe) to stay safe.
- **`libero_90` inclusion**: shell scripts include it, README doesn't.
  Reproduce with it.
- **Beta(1.5, 1) time sampling** is nontrivial and is not called out in
  the README — don't silently swap it for uniform during the port.
- **attn / MLP dropout = 0.1** in the action head — unusual for modern
  transformers; don't zero it.
- **weight_decay = 0.0** — also unusual, don't fix it.
- **torch_dtype=float32 at load, bf16 via Accelerate** — keep this exact
  order; loading in bf16 breaks gradient accumulation for some SmolVLM
  layers.
- **Eps for AdamW**: not specified → PyTorch default `1e-8`. Probably
  safe, but worth logging.
- **Inference latency** is also unreported. We should benchmark this
  ourselves since LOBE targets real robots.
