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
Loaded policy: smolvla | 450,046,176 params
Starting policy server on ws://0.0.0.0:8000
Metadata: {'model_name': 'lobe-smolvla', 'policy_type': 'smolvla',
           'action_horizon': 50, 'action_dim': 7,
           'image_keys': ['observation.images.camera1', ...]}
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
        print(f"Connected: {meta['model_name']}")

        obs = {
            "state": np.zeros(8, dtype=np.float32),
            "images": {
                "camera1": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                "camera2": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            },
            "prompt": "pick up the bowl",
        }
        await ws.send(msgpack.packb(obs, use_bin_type=True, default=msgpack_numpy.encode))

        resp = msgpack.unpackb(await ws.recv(), raw=False, object_hook=msgpack_numpy.decode)
        print(f"Action: {resp['actions'].shape}, infer={resp['timing']['infer_ms']}ms")

asyncio.run(main())
```

## Verified end-to-end

We tested with the SmolVLA 450M checkpoint:

| Test | Result |
|---|---|
| Load checkpoint | ✓ 450M params loaded in ~7s |
| Connect via WebSocket | ✓ |
| Send fake obs (state + 2 cameras + prompt) | ✓ |
| Receive action `(1, 7)` | ✓ |
| Inference time (first call) | 1357 ms |
| Inference time (warm) | ~50 ms expected |

## Production notes

- **Pin the Python**. The server holds the model in GPU memory; restart only when needed.
- **Use `--device=cuda:N`** to pin to a specific GPU if multiple are available.
- **Action horizon ≠ executed steps**. The robot decides how many actions to execute from each chunk before re-querying the server. For VLAs, `n_action_steps=10` is a good default.
