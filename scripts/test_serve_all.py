"""End-to-end serving test for all YAM checkpoints.

For each trained backbone, starts lobe-serve, connects a test client,
sends a synthetic observation, and verifies the returned action shape
matches (action_horizon, 14).

Usage:
    uv run python scripts/test_serve_all.py
    uv run python scripts/test_serve_all.py --checkpoints dp:path/to/pretrained_model
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time

import msgpack
import msgpack_numpy  # noqa: F401
import numpy as np
from loguru import logger

EXPECTED_ACTION_DIM = 14
IMAGE_SHAPE = (240, 320, 3)  # HWC uint8, matching the 240x320 training resize

DEFAULT_CHECKPOINTS = {
    "dp": "checkpoints/yam-grey-cube-dp-v0/checkpoints/050000/pretrained_model",
    "fm": "checkpoints/yam-grey-cube-fm-v0/checkpoints/030000/pretrained_model",
    "xvla": "checkpoints/yam-grey-cube-xvla-v0/checkpoints/020000/pretrained_model",
    "smolvla": "checkpoints/yam-grey-cube-smolvla-v0/checkpoints/020000/pretrained_model",
}

# Camera name mapping per backbone (must match what the policy was trained with)
CAMERA_NAMES = {
    "dp": ["head_camera", "left_wrist_camera", "right_wrist_camera"],
    "fm": ["head_camera", "left_wrist_camera", "right_wrist_camera"],
    "xvla": ["image", "image2", "image3"],
    "smolvla": ["camera1", "camera2", "camera3"],
}


def make_synthetic_obs(backbone: str) -> dict:
    """Build a synthetic observation matching the expected format for lobe-serve."""
    cams = CAMERA_NAMES.get(backbone, CAMERA_NAMES["dp"])
    return {
        "state": np.zeros(14, dtype=np.float32).tolist(),
        "images": {cam: np.random.randint(0, 255, IMAGE_SHAPE, dtype=np.uint8) for cam in cams},
        "prompt": "pick up the grey cube and hand it to another hand",
    }


async def test_client(port: int, backbone: str, timeout_s: float = 120.0) -> dict:
    """Connect to server, send one obs, verify response shape."""
    import websockets

    uri = f"ws://localhost:{port}"
    t0 = time.time()

    while time.time() - t0 < timeout_s:
        try:
            async with websockets.connect(uri, max_size=100 * 1024 * 1024) as ws:
                # Receive metadata
                meta_raw = await asyncio.wait_for(ws.recv(), timeout=10)
                meta = msgpack.unpackb(meta_raw, raw=False)
                logger.info(f"  [{backbone}] metadata: action_dim={meta.get('action_dim')}, "
                            f"action_horizon={meta.get('action_horizon')}, "
                            f"policy_type={meta.get('policy_type')}")

                # Send synthetic observation
                obs = make_synthetic_obs(backbone)
                obs_packed = msgpack.packb(obs, use_bin_type=True, default=msgpack_numpy.encode)
                await ws.send(obs_packed)

                # Receive action
                resp_raw = await asyncio.wait_for(ws.recv(), timeout=60)
                resp = msgpack.unpackb(resp_raw, raw=False, object_hook=msgpack_numpy.decode)

                actions = np.asarray(resp["actions"])
                timing = resp.get("timing", {})
                logger.info(f"  [{backbone}] actions shape: {actions.shape}, "
                            f"infer_ms={timing.get('infer_ms', '?')}")

                return {
                    "backbone": backbone,
                    "pass": actions.shape[-1] == EXPECTED_ACTION_DIM,
                    "action_shape": list(actions.shape),
                    "action_dim": int(actions.shape[-1]),
                    "action_horizon": int(actions.shape[0]) if actions.ndim == 2 else 1,
                    "infer_ms": timing.get("infer_ms"),
                    "metadata": meta,
                }
        except (ConnectionRefusedError, OSError):
            await asyncio.sleep(2)

    return {"backbone": backbone, "pass": False, "error": "timeout waiting for server"}


def test_one_checkpoint(backbone: str, checkpoint: str, port: int, gpu: int) -> dict:
    """Start server, run client test, kill server."""
    if not os.path.exists(checkpoint):
        logger.warning(f"[{backbone}] checkpoint not found: {checkpoint}")
        return {"backbone": backbone, "pass": False, "error": "checkpoint not found"}

    logger.info(f"[{backbone}] Starting server on port {port} (GPU {gpu})...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    server_proc = subprocess.Popen(
        [
            sys.executable, "-m", "lobe.serve",
            f"--checkpoint={checkpoint}",
            f"--port={port}",
            "--chunk-mode",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        result = asyncio.run(test_client(port, backbone))
    except Exception as e:
        result = {"backbone": backbone, "pass": False, "error": str(e)}
    finally:
        server_proc.send_signal(signal.SIGTERM)
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        # Dump server output on failure
        if not result.get("pass"):
            stdout = server_proc.stdout.read().decode() if server_proc.stdout else ""
            logger.error(f"  [{backbone}] server output (last 1000 chars):\n{stdout[-1000:]}")

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints", nargs="*",
        help="name:path pairs (e.g. dp:path/to/model). Defaults to all YAM checkpoints.",
    )
    parser.add_argument("--gpu", type=int, default=7, help="GPU to use for serving")
    parser.add_argument("--base-port", type=int, default=9100)
    args = parser.parse_args()

    checkpoints = dict(DEFAULT_CHECKPOINTS)
    if args.checkpoints:
        checkpoints = {}
        for item in args.checkpoints:
            name, path = item.split(":", 1)
            checkpoints[name] = path

    results = []
    for i, (backbone, ckpt) in enumerate(checkpoints.items()):
        port = args.base_port + i
        r = test_one_checkpoint(backbone, ckpt, port, args.gpu)
        results.append(r)
        status = "PASS" if r.get("pass") else "FAIL"
        logger.info(f"[{backbone}] {status} — shape={r.get('action_shape')} infer={r.get('infer_ms')}ms")

    logger.info("=" * 60)
    passed = sum(1 for r in results if r.get("pass"))
    logger.info(f"Results: {passed}/{len(results)} passed")
    for r in results:
        status = "PASS" if r.get("pass") else "FAIL"
        logger.info(f"  {r['backbone']}: {status} {r.get('action_shape', r.get('error', ''))}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
