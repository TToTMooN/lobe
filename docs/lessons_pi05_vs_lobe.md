# Lessons: why pi0.5 succeeds where FM/X-VLA fail on robot

## The observation that triggered this doc

From the YAM `place_the_vial` deployment audit (limb#9):

| Policy | wire-fidelity (chunk[0] vs GT, train frames) | inference latency | runs the task on hardware? |
|---|---|---|---|
| FM (`yam-place-vial-fm-v0`) | mean\|err\| 0.04 rad, MSE 0.0016 | 88 / 92 ms | wire OK, **task fails — drifts** |
| X-VLA (`yam-place-vial-xvla-v0`) | mean\|err\| 0.042 rad, MSE 0.017 | 88 / 92 ms | wire OK, **task fails — drifts** |
| **pi0.5 (`yam-place-vial-pi05-v0`)** | mean\|err\| **0.013** rad, MSE **0.00047** | 158 / 188 ms | **task works** |

pi0.5 has roughly **3× lower wire MSE** AND **completes the task in closed loop**. FM and X-VLA pass the per-frame test but their compounded errors over a multi-step trajectory drift the robot away from the demonstrated solution. The probes can't see this because they only check `chunk[0]` and `chunk[t]` against ground-truth at training-time states — they don't simulate the actual closed-loop the robot runs.

This doc lays out hypotheses for *why* pi0.5 wins and concrete experiments to close the gap on the LOBE side.

## What's structurally different (just the relevant axes)

| Axis | FM (`flow_matching`) | X-VLA | **pi0.5** |
|---|---|---|---|
| Vision encoder | ResNet18 + spatial-softmax (32 keypoints) | Florence-2 large (frozen) | PaliGemma frozen + flow expert |
| Backbone | UNet1D (3 stages, 512/1024/2048) | DiT-style transformer | Gemma transformer (flow head) |
| Pretraining | ImageNet (ResNet18 only) | ~5M-frame VLA mix (X-VLA-Pt) | **~10M-frame robot mix (pi05_base on hundreds of robots)** |
| Action repr | absolute joint positions | absolute joint positions (auto-pad to 20) | absolute joint positions (auto-pad to 32) |
| Action horizon | 16 trained / 8 served | 30 / 30 | 16 trained / **50 served** |
| State norm | MIN_MAX | IDENTITY (no norm) | **Q01-Q99 quantile** (asymmetric tail trim) |
| Action norm | MIN_MAX | IDENTITY | **Q01-Q99 quantile** |
| Conditioning | obs + ground-truth task | obs + prompt | obs + prompt + delta-t embed |
| Image preprocessing | bilinear interp 240×320 | per-camera mix (256 / 256 / 224) padded | per-camera padded to 224×224 |
| Training data this task | 5000 → 50000 steps, batch 56 | same | **5000 steps, batch 56** (much fewer steps because pretrained on 10M frames) |
| Optimizer | AdamW 1e-4 | AdamW 1e-4 const | AdamW with cosine decay 5e-5 |

## Hypotheses ranked by likelihood

### H1: Pretraining scale is the dominant factor (most likely)

pi05_base saw ~10M frames across hundreds of robots before our 5000-step fine-tune; X-VLA-Pt saw ~5M; FM/DP started from ImageNet. After fine-tuning on our 230k frames, pi0.5 only had to *adapt* a strong action prior. FM had to *learn it from scratch* on 5 epochs of the small dataset.

**Smoking-gun evidence to verify:** 
- FM's loss is still dropping at 50k (per wandb), and the 40k checkpoint is better than 50k on held-out — hints that FM is data-starved and dataset-size-limited, not optimizer-limited. More steps just overfit.
- pi0.5 reached 0.013 rad mean error in 5000 steps (no overfit yet observed).

**Concrete experiments:**
- Train FM with 5×–10× more YAM data (cross-task fine-tune mix) and see if wire-error catches pi0.5.
- Initialize FM's vision encoder from a stronger pre-trained source (DINOv2, SigLIP — the LOBE config already exposes these but we used ImageNet-ResNet18).

### H2: Quantile normalization vs MIN_MAX

pi0.5 normalizes by Q01–Q99 (asymmetric: clips the bottom and top 1% of training values, maps the rest to roughly [-1, 1]). FM uses MIN_MAX (linear stretch over the absolute extrema).

Why this matters for closed-loop drift: with MIN_MAX, a training-time outlier (rare extreme joint pose) compresses the dynamic range of "normal" joint positions. The model then represents *typical* states with low precision, so per-step prediction errors are larger in absolute terms. Q01–Q99 dedicates the model's representation budget to the bulk of the distribution.

**Concrete experiments:**
- Re-train FM with `--policy.normalization_mapping='{"VISUAL":"MEAN_STD","STATE":"Q01_Q99","ACTION":"Q01_Q99"}'` (lerobot supports this).
- Compare wire-fidelity AND closed-loop probe.

### H3: Action horizon mismatch

pi0.5 deploys with `horizon=50` (1.7 s of lookahead at 30 fps). FM ships only 8 (0.27 s). With short horizons, the closed-loop *immediately* depends on the model continuing to track well frame-to-frame. With long horizons, the policy commits to a stable plan and re-plans every 1.7 s — drift between plans is amortized.

**This is partially testable on the existing FM checkpoint**: serve with `--policy.n_action_steps=16` (use the full trained horizon, not just half). No retraining needed.

### H4: Vision encoder capacity

ResNet18 (12M params, ImageNet) vs PaliGemma (~2B params, robot-pretrained). For a fine-grained manipulation task like vial-into-stand, the difference in *what* the model can perceive (vial position relative to stand opening) is plausibly large. X-VLA uses Florence-2 which is bigger than ResNet18 but still smaller than PaliGemma — and X-VLA also fails on hardware.

**Concrete experiments:**
- Swap FM's `vision_encoder` from `spatial_softmax` to `dinov2` or `siglip` (both wired in the FM config). Train and compare.

### H5: Sampling/inference resolution differences

- FM uses 5-step Euler ODE solving by default at deploy. pi0.5 uses 10 internal steps (`num_denoising_steps=10` baked into the openpi config).
- More denoising steps = better action generation but slower.

**Concrete experiments:**
- Serve FM with `--num-inference-steps=10` and re-probe. Cheap, no retrain.

### H6: Training dataset rate mismatch (the user's resampling work directly addresses this)

The original `place_the_vial_into_the_stand_1to4` dataset was recorded at 50–58 Hz but labeled as 30 fps in `info.json` (limb's old converter hardcoded fps=30). So FM/X-VLA learned 50-Hz transitions but think they're 30 Hz.

The new `ttotmoon/8ml_vial_place_30fps` dataset is **honestly resampled** to 30 fps (limb#11). Re-training on this should immediately help FM and X-VLA — they'll see transitions that match what they'll see at deploy.

**This is the most actionable lesson** — it's already done by limb's resampler, and we just need to re-train.

## Action items for v1 retraining

In rough order of bang-for-buck:

1. **(✓ first thing to do)** Retrain FM and X-VLA on `ttotmoon/8ml_vial_place_30fps`. This alone may close most of the gap because the 50-Hz-labeled-as-30-Hz mismatch was a real data corruption — the model was learning the wrong dynamics.
2. Switch FM's normalization to Q01–Q99 (single config-line change in `lobe/configs/yam.py`).
3. Bump FM's deployment `n_action_steps` to 16 (use the full trained horizon, no retrain).
4. Probe FM with `--num-inference-steps=10` (no retrain).
5. Try DINOv2 vision encoder for FM (already configurable, ~2× train time).
6. (Long-term) Pre-train FM/DP on a multi-task YAM mix before fine-tuning on a single task — match the structural advantage pi0.5 enjoys.

## What this doc does NOT claim

- That FM "can't" beat pi0.5 — FM has 1/8 the parameters and competes with random init, that's a fair handicap.
- That the wire-fidelity probe is wrong — it's correct as far as it goes (per-frame chunk[0]). It just can't see compounded drift in closed-loop.
- That gripper-binarize / hz-mismatch / `--compile`-broken were red herrings — they were real bugs we fixed, but the underlying gap remains.

## Update (2026-05-11): FM v2 result

`yam-vial-place-fm-v2-h32` implements openpi-style mixed-delta (joints subtract chunk-start
state, gripper kept absolute) + Q01-Q99 over chunked-delta stats (every `(t, i)` pair for
`i ∈ [0, H-1]`). Pretraining and architecture left unchanged.

Wire-probe (chunk[0] vs GT on 30 training frames):

| metric | FM v2 | FM v0 | pi0.5 (target) |
|---|---|---|---|
| chunk[0] joint mean abs_max_err | **0.0286** | 0.053 | 0.013 |
| chunk[0] gripper mean abs_max_err | **0.0220** | 0.042 | — |

**~2× over v0 on both axes. About half the remaining gap to pi0.5 is closed.**
The rest is structural (pretraining, vision encoder, parameter count) — not a
recipe-level miss. Validates H2 (normalization) as the largest training-recipe lever.

See `docs/openpi_pipeline_full.md` for the exact implementation and a description of the
chunked-vs-single-step delta-stats pitfall that broke the first v2 attempt.

## Verification protocol for any v1 candidate

For each new checkpoint:

1. **Replay-eval** on held-out episodes (`scripts/eval_replay.py`). Lower MSE than v0 = good.
2. **Wire-fidelity probe** (`limb/scripts/diagnostics/probe_policy_server.py`). chunk[0] mean\|err\| should be < 0.02 rad to be in pi0.5's range.
3. **Closed-loop dry-run** (`limb/scripts/diagnostics/dry_run_policy.py`). max per-step Δ should stay within 0.03 rad/step at deploy hz.
4. **Hardware run** (only if 1–3 pass). The only test that catches drift.
