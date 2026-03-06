# Diffusion Policy vs Flow Matching: What Actually Changes

## Side-by-side: the only code that differs

Both use the **exact same architecture** — 1D conditional U-Net with FiLM modulation, ResNet/ViT vision encoder, spatial softmax, EMA.

The difference is **only in the noise process + loss + inference loop**.

### Training (loss computation)

**Diffusion (DDPM) — LeRobot `modeling_diffusion.py:336-374`:**
```python
# Sample noise
eps = torch.randn(trajectory.shape)
# Sample random INTEGER timestep from [0, num_train_timesteps)
timesteps = torch.randint(0, num_train_timesteps, (B,))
# Add noise via DDPM schedule (nonlinear beta schedule)
noisy_trajectory = noise_scheduler.add_noise(trajectory, eps, timesteps)
# Predict noise (or clean sample)
pred = unet(noisy_trajectory, timesteps, global_cond=obs_cond)
# Loss: predict eps or x0
loss = MSE(pred, eps)  # if prediction_type="epsilon"
```

**Flow matching — HRI-EU `flow_pusht.py:130-141`:**
```python
# Sample noise
x0 = torch.randn(trajectory.shape)
# Sample random CONTINUOUS timestep, get interpolated point + target velocity
t, xt, ut = ConditionalFlowMatcher.sample_location_and_conditional_flow(x0, trajectory)
# xt = (1 - t) * x0 + t * trajectory         (linear interpolation)
# ut = trajectory - x0                         (constant velocity field)
# Predict velocity field
vt = unet(xt, t, global_cond=obs_cond)
# Loss: predict velocity vector
loss = MSE(vt, ut)
```

### Inference (action generation)

**Diffusion (DDPM/DDIM) — LeRobot `modeling_diffusion.py:210-243`:**
```python
sample = torch.randn(B, horizon, action_dim)           # start from pure noise
noise_scheduler.set_timesteps(num_inference_steps)       # e.g. 100 or 10 (DDIM)
for t in noise_scheduler.timesteps:                      # iterate backwards
    pred = unet(sample, t, global_cond=obs_cond)
    sample = noise_scheduler.step(pred, t, sample)       # complex DDPM/DDIM step
```

**Flow matching — HRI-EU `flow_pusht.py:211-223`:**
```python
sample = torch.randn(B, horizon, action_dim)            # start from noise
num_steps = 1                                            # can be as few as 1!
for i in range(num_steps):
    t = torch.tensor([i / num_steps])
    vt = unet(sample, t, global_cond=obs_cond)          # predict velocity
    sample = sample + vt * (1 / num_steps)               # Euler step forward
```

## What's different, precisely

| Aspect | Diffusion (DDPM) | Flow Matching |
|--------|------------------|---------------|
| **Timestep** | Integer ∈ [0, T), discrete | Continuous ∈ [0, 1] |
| **Interpolation** | Nonlinear (beta schedule) | Linear: `xt = (1-t)*x0 + t*x1` |
| **Target** | Noise `eps` (or clean sample `x0`) | Velocity `ut = x1 - x0` |
| **Loss** | `MSE(pred, eps)` | `MSE(vt, ut)` |
| **Inference** | Reverse diffusion: ~10-100 DDPM/DDIM steps | Forward ODE: as few as **1 Euler step** |
| **Inference scheduler** | DDPM/DDIM scheduler (complex) | Simple Euler integration (trivial) |
| **Dependencies** | `diffusers` (DDPMScheduler, DDIMScheduler) | `torchcfm` (or ~10 lines of code) |

## Why flow matching is better (for our use case)

1. **Faster inference**: 1-step inference works (HRI-EU uses 1 step for PushT U-Net). Diffusion needs 10+ steps minimum. At 100 Hz control rate, this matters.

2. **Simpler code**: No noise schedule, no beta parameters, no DDPM/DDIM scheduler. Inference is literally `x = x + v * dt`. Training is literally `loss = MSE(net(xt, t, cond), x1 - x0)`.

3. **Straighter trajectories**: Linear interpolation = straighter paths in latent space = easier for the network to learn. DDPM's curved trajectories are harder to fit.

4. **More stable training**: HRI-EU reports "more stable training and evaluation" across all benchmarks.

5. **Same or better performance**: HRI-EU reports "comparable generalization performance, where flow matching performs marginally better in most cases."

## What you'd lose choosing pure Option A (patch diffusion → FM)

**Nothing functional.** The architecture stays identical. You just swap ~30 lines in the loss + inference. The U-Net, vision encoder, FiLM conditioning, EMA — all unchanged.

## What HRI-EU adds beyond basic flow matching (Option B)

1. **Affordance conditioning**: They add a prompt-tuned frozen vision model that predicts manipulation affordances (keypoints/regions of interest), then condition the flow matching policy on those affordances. This is their main research contribution — it's NOT just "diffusion with flow matching loss."

2. **Transformer backbone option**: They have both U-Net (`flow_pusht.py`) and Transformer (`flow_pusht_transformer.py`) variants. LeRobot's diffusion policy only has U-Net.

3. **TorchCFM integration**: They use `torchcfm.ConditionalFlowMatcher` which handles optimal transport coupling (not just independent coupling). This can improve sample quality.

## Recommendation: Option A+B hybrid

**Do Option A first** (takes ~1 hour of code):
- Fork LeRobot's `DiffusionPolicy` → `FlowMatchingPolicy`
- Replace `compute_loss()`: swap DDPM noising for linear interpolation + velocity target
- Replace `conditional_sample()`: swap DDPM reverse process for Euler forward integration
- Remove `diffusers` scheduler dependency (replace with ~10 lines)
- Add config option `num_inference_steps` (default 1, can go higher for quality)
- Everything else (U-Net, vision encoder, obs processing, action chunking) stays identical

This gives you a **clean A/B comparison**: same architecture, same data, diffusion loss vs flow matching loss.

**Then cherry-pick from Option B**:
- Add `torchcfm` optimal transport coupling (improves sample diversity) — ~5 lines
- Optionally add Transformer backbone as alternative to U-Net
- Skip the affordance stuff (it's their research contribution, not relevant to our direct control setup)

## Code changes needed (Option A)

In LeRobot's `modeling_diffusion.py`, only these methods change:

1. `DiffusionModel.__init__()` — remove `noise_scheduler`, add `sigma` param
2. `DiffusionModel.compute_loss()` — ~15 lines (replace DDPM forward with FM interpolation)
3. `DiffusionModel.conditional_sample()` — ~10 lines (replace DDPM reverse with Euler forward)
4. `DiffusionConfig` — remove DDPM scheduler params, add `num_inference_steps` (int, default 1)

Total: ~50 lines changed, ~30 lines removed. Same file structure, same policy interface.
