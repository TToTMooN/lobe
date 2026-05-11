# OpenPI pi0.5 — exhaustive pipeline reference, and what lobe FM gets wrong

Read the actual code (not the docs), traced every transform. This doc is the source of truth before any FM v2 change.

## The full pipeline, in exact order

### Configuration

`pi05_yam_place_vial` resolves to a `TrainConfig` whose `data` field is a `LeRobotAlohaDataConfig`. At creation time (`DataConfigFactory.create`), it builds a `DataConfig` with:

- `repo_id`: the local LeRobot dataset
- `norm_stats`: loaded from `assets_dir/asset_id/norm_stats.json` (or computed by `compute_norm_stats.py`)
- `use_quantile_norm`: **True for PI05** (set from `model_config.model_type != PI0` in `DataConfigFactory.create_base_config`)
- `repack_transforms` / `data_transforms` / `model_transforms`: three groups, each with `inputs[]` and `outputs[]`

The three groups compose like this in `transform_dataset()` (file: `src/openpi/training/data_loader.py:172`):

```
TransformedDataset(dataset, [
    *repack_transforms.inputs,
    *data_transforms.inputs,
    Normalize(norm_stats, use_quantiles=use_quantile_norm),   ←── here
    *model_transforms.inputs,
])
```

### Input pipeline (per sample) — TRAINING

Raw sample from LeRobotDataset:
```
{
  "observation.images.head_camera":         (3, H, W) uint8   # video-decoded
  "observation.images.left_wrist_camera":   (3, H, W) uint8
  "observation.images.right_wrist_camera":  (3, H, W) uint8
  "observation.state":                      (14,) float32     # joint pos + gripper
  "action":                                 (H, 14) float32   # chunk of H future actions
  "task":                                   "place the vial..." str
  ...indices, timestamps, etc
}
```

Pipeline:

**Step 1 — `repack_transforms.inputs`** (per `pi05_yam_place_vial` config):

```python
RepackTransform({
    "images": {
        "cam_high":        "observation.images.head_camera",
        "cam_left_wrist":  "observation.images.left_wrist_camera",
        "cam_right_wrist": "observation.images.right_wrist_camera",
    },
    "state":   "observation.state",
    "actions": "action",
})
```

Renames keys to match openpi's internal schema. After this step:
```
{"images": {cam_high: (3,H,W) uint8, ...}, "state": (14,), "actions": (H,14), "prompt": str}
```

**Step 2 — `data_transforms.inputs`** (built from `LeRobotAlohaDataConfig.create`):

```python
[
    AlohaInputs(adapt_to_pi=False),           # decode + restructure
    DeltaActions(mask=make_bool_mask(6,-1,6,-1)),  # subtract state from joint dims
]
```

**2a. `AlohaInputs(adapt_to_pi=False)`** does:
- For each image: `(3,H,W) uint8 → (H,W,3) uint8` (this is `convert_image()` with `einops.rearrange("c h w -> h w c")`)
- For state: keep as-is for adapt_to_pi=False (joint-flip and gripper-space-conversion skipped — that's Trossen-Aloha-specific)
- Adds `image_mask` field: each cam gets `True` if present, `False` if missing (auto-filled with zeros)
- Output schema: 
  ```
  {
      "image": {base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb},  # each (H,W,3) uint8
      "image_mask": {base_0_rgb: True, ...},
      "state": (14,),
      "actions": (H, 14),     # only during training
      "prompt": "...",
  }
  ```

**2b. `DeltaActions(mask=(6,-1,6,-1))`** does:
```python
state, actions = data["state"], data["actions"]   # state: (14,), actions: (H, 14)
# mask: True for joint dims (0-5, 7-12), False for gripper dims (6, 13)
actions[..., :14] -= np.expand_dims(np.where(mask, state[..., :14], 0), axis=-2)
# → for joint dims: actions[t, d] := actions[t, d] - state[d]   for all t in chunk
# → for gripper dims: actions[t, d] unchanged (still absolute)
```

**Critical**: this uses **state of the chunk's first frame** as the anchor for ALL H steps in the chunk. Same anchor for every step. Joints become delta-from-anchor; gripper stays absolute.

After this step, `actions` field holds a **mixed delta/absolute representation**:
- columns 0-5: delta from `state[d]`
- column 6: absolute gripper position
- columns 7-12: delta from `state[d]`
- column 13: absolute gripper position

**Step 3 — `Normalize(norm_stats, use_quantiles=True)`** does:

```python
for key in ["state", "actions"]:
    if use_quantiles:
        x_n = (x - q01) / (q99 - q01 + 1e-6) * 2 - 1   # [-1, 1] after q01-q99 clip-and-stretch
    else:
        x_n = (x - mean) / (std + 1e-6)
```

The `norm_stats` were **computed by `compute_norm_stats.py`** which ran the EXACT SAME PIPELINE through step 2b before measuring stats. So:

- `norm_stats["state"]` = stats of raw 14-D state distribution (q01/q99/mean/std)
- `norm_stats["actions"]` = stats of the **already-delta-transformed mixed-action distribution**:
  - dims 0-5, 7-12: delta-of-joint stats (small std, e.g. 0.02-0.06 rad)
  - dims 6, 13: absolute-gripper stats (bimodal 0/2.4)

**Step 4 — `model_transforms.inputs`** (for PI05):

```python
[
    InjectDefaultPrompt(default_prompt),       # if no prompt, use default
    ResizeImages(224, 224),                    # bilinear resize per image
    TokenizePrompt(PaligemmaTokenizer(max_token_len=200), discrete_state_input=True),
    PadStatesAndActions(model_action_dim=32),  # zero-pad 14→32
]
```

- **InjectDefaultPrompt**: `prompt = data.get("prompt") or default_prompt`
- **ResizeImages**: for each cam, `cv2.INTER_AREA` (downsize) or `INTER_LINEAR` (upsize) to 224×224
- **TokenizePrompt**: tokenizes the prompt for PaliGemma encoder, max 200 tokens
- **PadStatesAndActions**: `pad_to_dim` adds zeros to make state[14] → state[32] and actions[H,14] → actions[H,32]

After all four steps, model gets:
```
{
    "image": {3 cams of (224, 224, 3) uint8},
    "image_mask": ...,
    "state": (32,) float32   ← normalized state in 14 dims, zeros in remaining 18
    "actions": (H, 32) float32   ← normalized mixed-delta/absolute in 14 dims, zeros in remaining 18
    "tokenized_prompt": (200,) int32,
    "tokenized_prompt_mask": (200,) bool,
}
```

This is what `Pi0Model.compute_loss` consumes.

### Output pipeline — INFERENCE

`Policy.infer(obs)` does:
```
inputs = self._input_transform(obs)         # the 4 steps above (DeltaActions is a no-op at inference because no "actions" key in obs)
inputs = batch_dim_added
predicted_actions = self._sample_actions(rng, observation)   # model forward, returns (H, 32) normalized mixed-delta
outputs = {"state": inputs["state"], "actions": predicted_actions}
outputs = self._output_transform(outputs)
```

The `output_transforms` (built in `policy_config.create_trained_policy`):
```python
[
    *data_config.model_transforms.outputs,     # empty for pi05
    Unnormalize(norm_stats, use_quantiles=True),
    *data_config.data_transforms.outputs,      # AbsoluteActions + AlohaOutputs
    *repack_transforms.outputs,                # empty
]
```

**Output Step 1 — `Unnormalize`**:
```python
# For state and actions:
x = (x_n + 1) / 2 * (q99 - q01) + q01
# Now actions is (H, 32) raw mixed-delta-and-absolute
```

**Output Step 2 — `AbsoluteActions(mask=(6,-1,6,-1))`**:
```python
actions[..., :14] += np.expand_dims(np.where(mask, state[..., :14], 0), axis=-2)
# For joint dims: actions[t, d] = delta + state[d]   ← absolute!
# For gripper dims: actions[t, d] unchanged (was always absolute)
# Uses state from the current obs (passed through to outputs by Policy.infer above)
```

**Output Step 3 — `AlohaOutputs(adapt_to_pi=False)`**:
```python
actions = actions[:, :14]   # slice off the 32→14 padding
# adapt_to_pi=False → no joint flip / gripper space conversion
```

Client receives `(H, 14)` actions in raw joint+gripper space, ready for robot.

## What lobe FM v1' actually does (the diff)

| Step | OpenPI | Lobe FM v1' |
|---|---|---|
| Repack camera keys | yes (renames to AlohaInputs schema) | n/a (lerobot uses keys directly) |
| Image layout convert | (3,H,W) → (H,W,3) inside `AlohaInputs.convert_image` | lerobot keeps CHW; lobe.serve unpacks both at wire |
| **Delta computation** | `actions -= state` (anchor = current state, masked to joints) BEFORE normalize | `actions -= actions[:, 0:1, :]` (anchor = FIRST ACTION in chunk, all dims) AFTER normalize |
| **Stats compute target** | computed on POST-DeltaActions data → action stats reflect delta distribution | computed on raw action → action stats reflect absolute distribution |
| **Normalize quantile** | q01–q99 (use_quantile_norm=True for PI05) | MIN_MAX / MEAN_STD or QUANTILES, controlled separately |
| Resize images | 224x224 bilinear | 240x320 bilinear (matches our training-time resize_shape) |
| Pad state/action | 14→32 zero-pad (PI05 needs 32-D) | no pad (FM operates natively on 14-D) |
| Inference output add-back | `actions += state` for joint dims only (AbsoluteActions with mask) | `actions += state` for ALL dims (no mask) |
| Action slice | first 14 dims (AlohaOutputs) | already 14-D |

## Concrete bugs in current lobe FM with `delta_actions=True`

1. **Train-inference anchor mismatch**:
   - Train: subtract `action[0]` of chunk
   - Inference: add `state` (current obs)
   - These are different anchors. Approximately OK if `action[0] ≈ state` (true for teleop), but not exactly.

2. **Normalization is on the wrong distribution**:
   - Currently: norm_stats are over raw action distribution (range like [-1.5, 2.5])
   - Should be: stats over the (mixed-)delta distribution where joints range like [-0.1, 0.1] and gripper stays [0, 2.4]
   - Result: model sees inputs where joint deltas are ~5% of [-1, 1] range — wasted dynamic range

3. **Delta mask not applied — gripper is treated as delta too**:
   - OpenPI: `mask=make_bool_mask(6, -1, 6, -1)` → joints delta, gripper absolute
   - Lobe FM: `batch[ACTION] - batch[ACTION][:, 0:1]` → ALL 14 dims are delta'd, including gripper
   - For gripper (bimodal 0 ↔ 2.4), making this delta yields targets like ±2.4 (huge swings), and at inference adding back gripper-state turns "0 - 0 = 0" or "2.4 - 0 = 2.4" reasonably, but during training it learns target = ±2.4 which is hard with MSE loss

4. **Subtraction happens in normalized space, not raw**:
   - Current code: subtracts `action_n[0]` from `action_n[t]` — this is correct algebraically for affine norm (mean cancels) but the model still sees these tiny deltas relative to the action-norm σ. So the SCALE of the deltas the model learns is wrong relative to the [-1, 1] expected range.

## Implications for v2 design

To match OpenPI semantics in lobe FM, we need ALL of:

1. **Compute delta-aware action stats** for our dataset:
   - For each (frame_t, chunk_offset i) where i ∈ [0, H-1] and t+i is in-episode:
     - joint dim → `action[t+i] - state[t]`
     - gripper dim → `action[t+i]` (kept absolute)
   - Aggregate stats (mean, std, q01, q99) over all these (N_frames × H) points
   - **This is what `openpi/scripts/compute_norm_stats.py` does**:
     `RunningStats.update(batch).reshape(-1, batch.shape[-1])` flattens (B, H, A) →
     (B*H, A) so every (sample, timestep-in-chunk) counts as ONE data point.
   - Common pitfall: computing stats over single-step deltas (i=0 only) makes q01/q99
     ~1.5–2× too narrow. We hit this in the first FM v2 attempt — 6 % of normalized
     values landed outside [-1, 1]. Re-compute over **all i in [0, H-1]** to fix
     (`/tmp/compute_chunked_delta_stats.py`).

2. **Apply DeltaActions (with proper mask) BEFORE Normalize**, not after:
   - For lerobot's pipeline this means adding a custom preprocessor step that runs before `Normalize`
   - Simplest implementation: set lerobot's ACTION normalization to IDENTITY, do the delta subtraction inside `FlowMatchingPolicy.forward`, then manually normalize using delta stats

3. **Use `state` (not `action[0]`)** as the anchor in both training and inference — this is the part the current code mostly already does at inference, just inconsistent with training.

4. **Mask the delta**: joint dims (0-5, 7-12) get state subtracted; gripper dims (6, 13) stay absolute.

5. **At inference**:
   - Model output is normalized mixed-delta
   - Unnormalize using delta stats
   - For joint dims: add back state (which we stored on chunk request)
   - For gripper dims: already absolute, no add-back

6. **Inference latency**: this changes nothing at the wire (state → predict → action). Server still receives raw obs, server still sends raw absolute actions. The internal model just has different math.

## What `lobe.serve` needs to change

If the FM model handles delta internally (recommended), `lobe.serve` is unchanged — it still calls `policy.select_action(obs)` which returns absolute actions. Nothing wire-level changes. The only "leak" is that `select_action` needs to remember the chunk-start state for the duration of the chunk it served — already the case in the current implementation.

## Open questions before I code anything

Q1. Should we apply delta semantics ONLY to joints, gripper kept absolute (matching OpenPI exactly)?  
   **My recommendation: yes** — matches OpenPI, matches the bimodal nature of YAM gripper.

Q2. Compute delta-stats with what aggregation: mean/std/min/max (we have these for raw action) or q01/q99 (need extra compute)?  
   **My recommendation: q01/q99** — OpenPI uses use_quantile_norm=True for PI05 with good reason. Costs ~1 min to add via `augment_dataset_quantile_stats.py`-style script.

Q3. Where to store delta stats:  
  (a) Overwrite the `action` entry in `stats.json` with delta stats — but breaks if someone reloads the dataset without our preprocessing.  
  (b) Add a sibling `stats.json` with delta-specific entries — but needs custom loading code.  
   **My recommendation: (a)**, with a marker in the lobe FM config that says "expect delta-flavored action stats here". Single source of truth.

Q4. Action horizon for v2: keep 16 (matching prior v1') or bump to 32?  
   **My recommendation: keep 16 first**, isolate the delta+normalization fix as one experiment. Run a separate v3 with horizon=32 once we've validated the v2 fix is correct.

Q5. For the v2 retrain: use the same dataset (`8ml_vial_place_30fps`) or recompute on a different cut?  
   **My recommendation: same dataset**, since limb#11's resampling already fixed the rate issue.

## What I will NOT do until you confirm

- Touch any model code
- Modify stats.json
- Launch any retrain

When you reply with which of Q1–Q5 you agree/disagree with, I'll write the code, smoke-test it, and only then queue the training.
