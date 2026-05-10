# Serving

`lobe-serve` exposes any trained policy over a WebSocket compatible with [limb's `WebSocketPolicyClient` protocol](https://github.com/TToTMooN/limb).

## Start the server

```bash
lobe-serve \
  --checkpoint=/path/to/checkpoints/100000/pretrained_model \
  --port=8000
```

The server prints metadata on startup:

```
Loaded policy: flow_matching | 274,844,078 params
Starting policy server on ws://0.0.0.0:8000
Metadata: {'model_name': 'lobe-flow_matching', 'policy_type': 'flow_matching',
           'action_horizon': 8, 'action_dim': 14,
           'image_keys': ['observation.images.head_camera', ...]}
```

## Protocol

1. Client connects to `ws://host:port`
2. Server sends metadata (msgpack):
   ```json
   {
     "model_name": "lobe-smolvla",
     "policy_type": "smolvla",
     "action_horizon": 50,
     "action_dim": 7,
     "image_keys": ["observation.images.camera1", ...]
   }
   ```
3. Client sends observation per timestep (msgpack with msgpack-numpy encoding):
   ```python
   {
     "state": np.float32(state_dim,),
     "images": {"camera1": np.uint8(H, W, 3), "camera2": ...},
     "prompt": "pick up the bowl"   # optional, for VLAs
   }
   ```
4. Server responds with action chunk:
   ```python
   {
     "actions": np.float32(action_horizon, action_dim),
     "timing": {"infer_ms": 13.4, "total_ms": 14.1}
   }
   ```
5. Repeat 3-4 until disconnect.

## Test client (Python)

```python
import asyncio
import msgpack
import msgpack_numpy
import numpy as np
import websockets

async def main():
    async with websockets.connect("ws://localhost:8000") as ws:
        meta = msgpack.unpackb(await ws.recv(), raw=False)
        print(f"Connected: {meta['model_name']}, action_dim={meta['action_dim']}")

        # YAM bimanual: 14-D state, 3 cameras at 240x320
        obs = {
            "state": np.zeros(14, dtype=np.float32),
            "images": {
                "head_camera": np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
                "left_wrist_camera": np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
                "right_wrist_camera": np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
            },
            "prompt": "pick up the grey cube and hand it to another hand",
        }
        await ws.send(msgpack.packb(obs, use_bin_type=True, default=msgpack_numpy.encode))

        resp = msgpack.unpackb(await ws.recv(), raw=False, object_hook=msgpack_numpy.decode)
        actions = resp["actions"]  # shape: (action_horizon, 14)
        print(f"Action: {actions.shape}, infer={resp['timing']['infer_ms']:.0f}ms")

asyncio.run(main())
```

## Modes

`lobe-serve` supports three modes:

### Chunk mode (default)

Server returns the full action chunk (e.g. 50 actions for SmolVLA) per observation. The robot client decides how many actions to execute before re-querying. This is the right mode for real robots because the policy forward pass (~1.3s for SmolVLA) is too slow to query at every 30 FPS timestep.

```bash
lobe-serve --checkpoint=<path>     # chunk mode is default
```

### Single-action mode

Server returns one action per observation. Useful for testing and for very lightweight policies. Run the policy on every robot timestep — only practical if inference time < 1/fps.

```bash
lobe-serve --checkpoint=<path> --no-chunk-mode
```

### Chunk mode + RTC (Real-Time Chunking)

Enables [Real-Time Chunking](https://www.physicalintelligence.company/research/real_time_chunking), which uses leftover actions from the previous chunk to guide generation of the next chunk via prefix attention. This produces smoother streaming inference at the cost of compute.

**When to use RTC**: only with VLAs (SmolVLA, pi0, etc.) where inference takes hundreds of milliseconds to seconds. The robot keeps executing the previous chunk while the server inference runs, then RTC stitches the new chunk seamlessly onto the in-flight one.

**When not to use RTC**: with fast policies like Diffusion Policy and our custom Flow Matching, inference is ~50ms — fast enough to query synchronously. Each new query produces a fresh chunk that the robot starts executing immediately. No stitching needed, no compute overhead.

**Supported policies**: SmolVLA, pi0, pi0_fast, pi05. These have `init_rtc_processor()` and accept RTC kwargs in `predict_action_chunk()`.

**Not supported (and not needed)**: Diffusion Policy and our custom Flow Matching policy — fast enough to run synchronously. Lerobot has not integrated RTC into Diffusion Policy, and we will not integrate it into our FM either unless someone wants it.

```bash
lobe-serve --checkpoint=<path> --chunk-mode --rtc
```

Additional RTC options:

| Flag | Default | Description |
|---|---|---|
| `--rtc-max-guidance-weight` | 10.0 | Max weight applied to prefix correction |
| `--rtc-execution-horizon` | 10 | Number of timesteps from prefix to use as guidance |
| `--rtc-inference-latency` | 0 (auto) | Estimated server inference time in seconds |

When `rtc-inference-latency=0`, the server tracks recent inference times in a rolling window and uses the average to compute `real_delay = latency * fps`. This determines how many actions from each new chunk are skipped (because they correspond to actions the robot already executed during inference time).

RTC requires the policy to support `predict_action_chunk()` with the `prev_chunk_left_over`, `inference_delay`, and `execution_horizon` kwargs.

## Inference speed flags

These reduce raw model forward-pass time. Use them when latency matters.

```bash
# DP: switch from DDPM-100 (450ms) to DDIM-10 (47ms) or DDIM-5 (35ms)
lobe-serve --checkpoint=<dp_ckpt> --noise-scheduler-type=DDIM --num-inference-steps=10

# FM: reduce ODE steps (default 10 → 5 or 3)
lobe-serve --checkpoint=<fm_ckpt> --num-inference-steps=5

# Any backbone: enable torch.compile (~1.3-10× speedup)
lobe-serve --checkpoint=<ckpt> --compile
```

| Flag | Effect |
|---|---|
| `--num-inference-steps=N` | Override denoising/ODE steps |
| `--noise-scheduler-type=DDIM` | Switch DP from DDPM to DDIM (fewer steps work) |
| `--compile` | Enable torch.compile (adds ~60-240s warmup on first call) |

## Verified end-to-end

### YAM (14-D bimanual joint-space, 3 cameras)

Tested with `scripts/test_serve_all.py` — starts server, sends synthetic
obs (14-D state + 3×240×320 images + task prompt), verifies action shape.

```bash
# Run all 4 backbones:
uv run python scripts/test_serve_all.py

# Run specific checkpoint:
uv run python scripts/test_serve_all.py --checkpoints dp:checkpoints/yam-grey-cube-dp-v0/checkpoints/050000/pretrained_model
```

| Backbone | Action shape | Infer (no compile) | Infer (compiled) |
|---|---|---|---|
| FM (5-step Euler) | (8, 14) | 23 ms | **18 ms** |
| SmolVLA (10 flow) | (50, 14) | 242 ms | **24 ms** |
| DP (DDIM-10) | (8, 14) | 47 ms | **35 ms** |
| X-VLA (10 flow) | (30, 14) | 81 ms | 78 ms |

### Inference benchmark

`scripts/bench_inference.py` measures raw forward-pass time (no deployment
patterns — just model inference with CUDA sync and proper warmup).

```bash
# All backbones, compiled vs uncompiled:
uv run python scripts/bench_inference.py --both

# Single backbone:
uv run python scripts/bench_inference.py --checkpoints fm:path/to/model --compile
```

### LIBERO (7-D single-arm, 2 cameras — original test)

| Test | Mode | Result |
|---|---|---|
| Load 450M SmolVLA checkpoint | — | ✓ ~7s |
| Single-action mode | default | ✓ shape `(1, 7)`, infer 1366 ms cold |
| Chunk mode | `--chunk-mode` | ✓ shape `(50, 7)`, infer 1403 ms cold |
| Chunk + RTC | `--chunk-mode --rtc` | ✓ infer 1284 → 270 → 254 ms across 3 obs |

## Production notes

- **Pin the Python**. The server holds the model in GPU memory; restart only when needed.
- **Use `--device=cuda:N`** to pin to a specific GPU if multiple are available.
- **Action horizon ≠ executed steps**. The robot decides how many actions to execute from each chunk before re-querying the server. For VLAs, `n_action_steps=10` is a good default.
