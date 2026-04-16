# Milestone: multi-backbone policy training on real limb YAM data

**Status**: design — not yet implemented. Next session starts here.
**Preceded by**: v1.3 X-VLA LIBERO reproduction (91.25% avg, see [X-VLA fine-tune](../workflows/xvla_finetune.md)).

## Goal

LOBE should be able to take any limb-collected LeRobot v3.0 dataset (starting with [`ttotmoon/yam_pick_up_grey_cube`](https://huggingface.co/datasets/ttotmoon/yam_pick_up_grey_cube)) and fine-tune four policy backbones end-to-end with one flag each, producing a checkpoint that can be:

1. **Replayed offline** against held-out demonstrations for a quantitative MSE metric.
2. **Deployed on the real robot** via limb's policy-client interface for on-robot success-rate evaluation.

The four backbones, in order of decreasing implementation cost:

| Backbone | Rationale | Expected relative difficulty |
|---|---|---|
| Diffusion Policy | Simplest baseline, train-from-scratch, ImageNet-pretrained ResNet visual encoder. Proves the full pipeline. | easy |
| Flow Matching | Drop-in swap of the policy class above. Head-to-head fair comparison with DP. | easy |
| SmolVLA | Smaller VLA (~450M), pre-trained generalist. Tests pretrained-init path. | medium |
| X-VLA | Larger VLA (~0.9B), our v1.3 LIBERO recipe. Biggest expected absolute success rate on long-horizon bimanual. | medium |

## Dataset contract

LOBE's pipeline consumes a LeRobot v3.0 dataset with the following features. This is the contract that limb's data writer must satisfy ([TToTMooN/limb#6](https://github.com/TToTMooN/limb/issues/6), [TToTMooN/limb#7](https://github.com/TToTMooN/limb/pull/7)):

- **`observation.state`**: `list<float32>` length 14 (not 14 flat scalar columns), named `[left_joint_0..5, left_gripper, right_joint_0..5, right_gripper]`.
- **`action`**: `list<float32>` length 14, joint-space, named with the same convention as state (not generic `left_action_0..6`).
- **`observation.images.{head_camera, left_wrist_camera, right_wrist_camera}`**: `video` dtype, 480×640 RGB, H.264/HEVC, one mp4 per episode.
- **`meta/tasks.parquet`** with pandas index `"task"`.
- **`meta/episodes/chunk-000/file-000.parquet`** with columns `episode_index, data/chunk_index, data/file_index, tasks (list<str>), length, meta/episodes/chunk_index, meta/episodes/file_index`, plus per-episode `stats/<feature>/{min, max, mean, std, count}` columns for every non-video feature and per-channel for every video feature.
- **`meta/info.json`** with `codebase_version: "v3.0"`, per-feature `fps`, no `total_chunks` / `total_videos` / `chunks_size`.
- **File naming**: `file-{file_index:03d}.{parquet,mp4}` — not `episode_{episode_index:06d}.*`. Use `file_index == episode_index` until a task has enough episodes to concat multiple per file.

A future limb release may add: dataset `README.md` / card with teleoperator, camera extrinsics, hardware version, and date range. Optional but recommended.

### Data quality inspected on `yam_pick_up_grey_cube`

From `scripts/evaluate_yam_dataset.py` (TBD — the logic is already written as `/tmp/eval_yam.py` in this session, to be promoted):

- 10 episodes, 10,958 frames, ~36 s/ep at 30 fps (range 26–52 s).
- State/action are smooth, near-absolute-joint-position trajectories. Step-to-step diff norm < 0.006 rad; `corr(action[t], state[t+1])` is 0.97–0.99 per joint. Clean teleop recording.
- Gripper **action** range `[0, 2.4]` differs from gripper **state** range `[0.2, 1.0]`; correlation 0.98 so they track, but units differ. Not a blocker — both policies train fine on mismatched-unit targets as long as they are internally consistent. Flag in limb dataset README once it exists.
- All 3 video cameras × 10 episodes have video frame count exactly equal to parquet frame count. Perfect temporal alignment.
- Only 10 episodes / 1 task / 11k frames is **few-shot fine-tune territory**. Sufficient for a single-task bimanual pick-and-handover; nowhere near enough for from-scratch VLA training. DP/FM can train from scratch (classical DP paper shows 80%+ on LIBERO with 50 demos per task; 10 × 1000 frames ≈ 50 × 200, same order). SmolVLA and X-VLA must fine-tune from a pretrained checkpoint.

## Pipeline architecture

```
┌─────────────────────────────────────────────────────────────┐
│ limb data collection (external repo, PR #7 landed)         │
│   → HF dataset in LeRobot v3.0 format                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ LOBE                                                         │
│   scripts/validate_yam_dataset.py    — schema + sanity       │
│   lobe/configs/yam.py                — backbone presets      │
│   scripts/_lobe_train_entry.py       — existing entry point  │
│   scripts/eval_replay.py             — MSE-on-held-out eval  │
│   serve.py                           — policy server         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ limb serve                                                   │
│   limb receives policy actions via WebSocket and executes    │
│   them on the YAM hardware for on-robot success-rate eval    │
└─────────────────────────────────────────────────────────────┘
```

## Per-backbone feature mapping

### Diffusion Policy (`policy.type=diffusion`)

- **Input**: all 3 cameras + state
- **Camera inputs**: LeRobot diffusion policy expects `observation.image` (singular) by default; the three YAM cameras need to be renamed OR we use the multi-camera variant (`observation.images.image`, `.image2`, `.image3`). Prefer the latter with a `--rename_map` from launch config.
- **State**: `observation.state` with shape `[14]` — native match.
- **Action**: `[14]` — native match.
- **Visual encoder**: ImageNet-pretrained ResNet18 (lerobot default). Not fine-tuned from limb data alone (10 demos is too few to move ResNet18 weights without overfitting); the policy learns to use a frozen encoder.
- **Backbone**: UNet1d (lerobot default), `horizon=16`, `n_obs_steps=2`.
- **Training**: batch 64, 50k steps, LR 1e-4 cosine with 500-step warmup, bf16, 8×H100, ~2h.

### Flow Matching (`policy.type=flow_matching`)

- Identical to Diffusion Policy in feature mapping, visual encoder, and data pipeline.
- Only difference: flow-matching ODE head instead of denoising diffusion.
- Use the same config preset as DP with `policy.type=flow_matching`.
- **Why run both**: head-to-head fair comparison (same data + same training + same encoder). Published papers agree DP ≈ FM within 3% when controlled; expect similar success rate on YAM.

### SmolVLA (`policy.path=lerobot/smolvla_base`)

- **Input**: 2 cameras (front + wrist) by default; the third YAM camera can be passed via the `empty_cameras=0, num_image_views=3` route or dropped at eval.
- **State**: SmolVLA expects LIBERO-style 8-D state (eef_pos + axis_angle + gripper_qpos). YAM's 14-D joint-space state doesn't match. Options:
  1. Pad-and-trim: zero-pad 14 → SmolVLA's `max_state_dim` (probably 32 given its training data mix). Let `pad_vector` handle it. SmolVLA's state encoder processes whatever comes in, so this works mechanically but may underperform compared to EEF input.
  2. Add an FK layer to convert YAM joints → EEF 8-D at dataset-build time. More aligned with SmolVLA pretraining but requires YAM kinematics (limb presumably has this).
- **Action**: SmolVLA outputs 7-D (LIBERO single-arm). YAM needs 14-D. This is the harder mismatch. Options:
  1. Use SmolVLA's action head but rewrite it to produce 14-D (requires model surgery).
  2. Use SmolVLA as a visual-language encoder only; train a new action head on top. The "Foundation-Policy" pattern.
- **Decision for phase 4**: start with pad-and-trim state + (b) new action head. Ship this as the simplest working version.
- **Training**: fine-tune 20k steps, batch 32, LR 1e-5 AdamW with 500-step warmup, bf16. ~1h.

### X-VLA (`policy.path=xvla-pt-v8`, `action_mode=auto`)

- **Input**: 3 cameras natively supported (Florence-2 VLM handles up to 3 image views).
- **State**: AutoActionSpace auto-detects `real_dim` from dataset. YAM's 14-D state goes in, gets padded to `max_state_dim=20` with trailing zeros. Same mechanism as LIBERO's 10-D state (our V14 recipe).
- **Action**: 14-D joint-space. `AutoActionSpace(real_dim=14, max_dim=20)` handles it. No body→site grip-site rotation (that was a LIBERO-specific EEF concern; YAM is joint-space, no frame conversion needed).
- **Training**: match LIBERO v1.2 recipe — 60k steps, batch 128, constant LR 1e-4, `num_denoising_steps=20` at eval (our V17b finding). ~3h40m train + ~45min eval.
- **`lobe/patches.py`** machinery transfers without change. The two patches (`_patch_xvla_make_policy_preserve_action_shape`, `_patch_xvla_libero_env_factory`) are dataset-agnostic; the second is LIBERO-env-specific but only fires when the env is LIBERO, so it's a no-op for YAM replay-based eval.

## Eval protocol

**No sim for YAM.** The three options, ordered by practicality:

### Option A: Replay-based MSE on held-out episodes (primary)

- Split: 8 train episodes, 2 held-out.
- For each held-out episode, for each frame `t`:
  1. Feed `observation.images.*[t]` + `observation.state[t]` + task prompt to the policy.
  2. Get predicted action chunk of length `n_action_steps`.
  3. Compare the first `k` predicted actions to ground-truth `action[t:t+k]`. Report MSE per joint dim, summed across episodes.
- Also compute "action L∞ norm" — max per-step error across a chunk — to catch tails that MSE averages away.
- **Not a task-success metric.** But it's (1) automatable, (2) cheap, (3) correlated with on-robot success as long as action space is well-defined and the policy is near-deterministic. It tells you whether the policy is sane before you move to on-robot.
- Implementation: `scripts/eval_replay.py`. Reuses `LeRobotDataset` for held-out access and any policy's `predict_action_chunk` for forward passes.

### Option B: Visual rollout comparison

- For each held-out episode, render (1) the predicted action chunk from t=0 over the first camera frame, (2) the demonstrated action chunk. Side by side.
- Not quantitative. Useful for debugging and for human-in-the-loop check before spinning up the robot.
- Implementation: `scripts/eval_rollout_viz.py`. Stretch goal.

### Option C: On-robot success rate (ground truth)

- limb runs the policy server (`lobe-serve`) against a trained checkpoint; limb's own teleoperation UI runs N trials and records success/failure.
- The only evaluation that measures real task completion.
- Requires physical robot access and human time; should remain a separate manual step, not part of the automated training pipeline.
- Implementation: requires limb's policy-client interface (per [TToTMooN/limb#1](https://github.com/TToTMooN/limb/pull/1)) to be functional and a small `scripts/eval_onrobot.py` that wraps repeated rollout requests.

### Reporting

`experiments.tsv` row per `{dataset × backbone × config}` tuple with columns: replay MSE (option A), qualitative rollout note (option B), on-robot success rate when available (option C, dated separately).

## Phase breakdown and implementation order

Total **5–7 days of elapsed wall time**. Critical path is Phases 0–3; Phases 4–5 are mostly configuration after the pipeline works on one backbone.

### Phase 0 — Dataset validation + cleanup utility (half a day)

- `scripts/validate_yam_dataset.py` that:
  1. Loads a LeRobot v3.0 dataset path.
  2. Runs the same checks from `/tmp/eval_yam.py` (state/action ranges, temporal smoothness, video/parquet alignment, correlation).
  3. Emits a pass/fail + structured report to stdout + an optional `--output-json`.
- If limb#7 hasn't fully landed by the time we start, a thin `scripts/fix_limb_dataset.py` one-shot also rewrites flat parquet columns to list<float>, synthesizes missing meta files, and runs `lerobot.scripts.convert_dataset_v21_to_v30`. Once limb#7 is merged and the HF re-upload is clean, this utility degenerates to a no-op validator.

### Phase 1 — Diffusion Policy baseline (1 day)

- `lobe/configs/yam.py` with `YAMConfig` dataclass:
  ```python
  @dataclass
  class YAMBaseConfig:
      dataset_repo_id: str = "ttotmoon/yam_pick_up_grey_cube"
      dataset_root: str | None = None  # local cache path
      output_dir: str = "checkpoints/yam"
  ```
  plus `YAMDiffusionConfig(YAMBaseConfig)` with DP-specific fields (`n_obs_steps=2`, `horizon=16`, batch 64, steps 50k, LR 1e-4).
- `PRESETS` dict: `"yam_grey_cube_diffusion"` → `YAMDiffusionConfig(...)`.
- Launch via `scripts/_lobe_train_entry.py` (reused from LIBERO path). The lobe patches apply universally.
- Ship a `docs/workflows/yam_finetune.md` with the launch command and expected training time.
- Success criterion for phase 1: trained checkpoint loads without error, replay MSE on held-out episodes is finite and monotone-improving during training (logged to wandb).

### Phase 2 — Flow Matching baseline (0.5 day on top of Phase 1)

- Swap `policy.type=diffusion` → `policy.type=flow_matching` in the preset.
- Keep everything else (encoder, augmentation, training schedule).
- Run head-to-head and log both replay MSEs to `experiments.tsv`.

### Phase 3 — Replay-based eval protocol (1 day)

- `scripts/eval_replay.py`:
  1. Take `--policy.path=<checkpoint>` and `--dataset.repo_id=<dataset>`.
  2. Split the last 2 episodes as held-out (or a user-provided `--eval_episodes="8,9"` flag).
  3. For each held-out frame, run policy forward and compute MSE vs demonstrated action chunk.
  4. Report aggregated per-joint-dim MSE + "action L∞" + average over dataset.
  5. Log to wandb + print summary.
- Used by Phase 1/2 as the automated stopping criterion before on-robot eval.

### Phase 4 — SmolVLA fine-tune (1 day)

- Download `lerobot/smolvla_base`.
- Add `YAMSmolVLAConfig` preset with pad-and-trim state + new 14-D action head + fine-tune recipe (20k steps, LR 1e-5, warmup 500).
- The action head mismatch is the main code change — everything else is config.

### Phase 5 — X-VLA fine-tune (1 day, mostly config reuse)

- Start from `xvla-pt-v8` (already remapped for auto mode).
- `YAMXVLAConfig` preset with `action_mode=auto`, `real_dim=14`, `max_dim=20`, V14 LIBERO recipe scaled down to the smaller YAM dataset (e.g. 20k steps instead of 60k since data is ~6× smaller in episodes).
- Optionally run the two-stage pattern from V16 IF a YAM auxiliary dataset exists (e.g. other limb-collected tasks on the same robot). For the single-task `yam_pick_up_grey_cube` this reduces to single-stage.
- `num_denoising_steps=20` at eval time (V17b finding).

### Phase 6 — Standardization + docs (0.5 day)

- Single `docs/workflows/yam_finetune.md` showing the four launch commands side by side.
- `scripts/train_yam.py` wrapper that picks a preset and launches (so the user types `uv run python scripts/train_yam.py --preset=yam_grey_cube_diffusion` rather than assembling a 30-flag command).
- Update `CLAUDE.md` and `BENCHMARKS.md` with YAM results once Phase 3 produces numbers.
- Close the milestone.

## Non-goals (explicit)

- **Not** building a sim environment for YAM. Replay-based MSE + on-robot is the protocol.
- **Not** supporting from-scratch X-VLA training on 10 YAM episodes — that's undertrained territory; X-VLA needs its pretrained starter. Diffusion/FM from scratch is fine because they're smaller.
- **Not** building a mixture-of-configs inference layer (per the "best per-suite cherry-pick" observation from LIBERO V17). One config per deployment.
- **Not** expanding to multi-task YAM data until at least one single-task pipeline produces a working on-robot policy. Diversity before multi-task-ness is premature abstraction.

## Open questions for next session

1. **SmolVLA action head**: which of the two action-mismatch options (model surgery vs new-head-on-frozen-encoder) does the user prefer? Phase 4 is a fork in the road.
2. **Eval MSE threshold for pass/fail**: what MSE value corresponds to "ready for on-robot eval"? TBD from first DP run — we'll see the baseline MSE and can pick a threshold empirically.
3. **YAM FK for SmolVLA EEF state**: does limb already have a forward-kinematics function exposed? If yes, Phase 4 option (2) becomes viable. If no, we stick with pad-and-trim.
4. **Robot eval cadence**: how often do we go to the robot? Weekly? Only at milestone completion? This affects how tight Phase 3's MSE protocol needs to be.

## References

- Dataset: [`ttotmoon/yam_pick_up_grey_cube`](https://huggingface.co/datasets/ttotmoon/yam_pick_up_grey_cube)
- limb repo: [TToTMooN/limb](https://github.com/TToTMooN/limb)
- limb dataset fix: [TToTMooN/limb#7](https://github.com/TToTMooN/limb/pull/7)
- Existing LOBE X-VLA recipe: [`docs/workflows/xvla_finetune.md`](../workflows/xvla_finetune.md)
- V14 patches: [`lobe/patches.py`](../../lobe/patches.py)
- Experiments log: [`experiments.tsv`](../../experiments.tsv)
