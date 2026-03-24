"""WebSocket policy client — for testing the server and as reference for limb.

Usage:
    # Test with PushT gym
    uv run python -m lobe.client --host localhost --port 8000

    # Benchmark latency
    uv run python -m lobe.client --benchmark --n-steps 100
"""

from __future__ import annotations

from dataclasses import dataclass

import msgpack
import msgpack_numpy
import numpy as np
from loguru import logger
from websockets.sync.client import connect

msgpack_numpy.patch()


class PolicyClient:
    """Sync WebSocket client matching limb's WebSocketPolicyClient protocol."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.url = f"ws://{host}:{port}"
        self.ws = None
        self.metadata = None

    def connect(self) -> dict:
        """Connect and receive server metadata."""
        self.ws = connect(self.url)
        meta_bytes = self.ws.recv()
        self.metadata = msgpack.unpackb(meta_bytes, raw=False)
        logger.info(f"Connected to {self.url}: {self.metadata}")
        return self.metadata

    def infer(self, obs: dict) -> dict:
        """Send observation, receive actions."""
        if self.ws is None:
            self.connect()
        obs_bytes = msgpack.packb(obs, use_bin_type=True)
        self.ws.send(obs_bytes)
        resp_bytes = self.ws.recv()
        return msgpack.unpackb(resp_bytes, raw=False)

    def close(self):
        if self.ws:
            self.ws.close()
            self.ws = None


@dataclass
class ClientArgs:
    host: str = "localhost"
    port: int = 8000
    benchmark: bool = False
    n_steps: int = 100


def main():
    import time

    import tyro

    args = tyro.cli(ClientArgs)
    client = PolicyClient(args.host, args.port)
    client.connect()

    if args.benchmark:
        logger.info(f"Benchmarking {args.n_steps} inference calls...")
        latencies = []
        for i in range(args.n_steps):
            obs = {
                "state": np.random.randn(2).astype(np.float32),
                "images": {"cam": np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)},
            }
            t0 = time.perf_counter()
            resp = client.infer(obs)
            latencies.append((time.perf_counter() - t0) * 1000)
            if "error" in resp:
                logger.error(f"Server error: {resp['error']}")
                break
            if i == 0:
                logger.info(f"First response: actions shape={resp['actions'].shape}, timing={resp.get('timing')}")
        latencies = np.array(latencies)
        logger.info(
            f"Latency: mean={latencies.mean():.1f}ms, p50={np.median(latencies):.1f}ms, "
            f"p95={np.percentile(latencies, 95):.1f}ms, p99={np.percentile(latencies, 99):.1f}ms"
        )
    else:
        # Interactive test with PushT gym
        import gym_pusht  # noqa: F401
        import gymnasium
        from lerobot.envs.utils import preprocess_observation

        env = gymnasium.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels_agent_pos")
        obs, _ = env.reset(seed=42)

        for step in range(300):
            processed = preprocess_observation(obs)
            client_obs = {
                "state": processed["observation.state"].numpy().squeeze(),
                "images": {
                    "cam": (processed["observation.image"].numpy().squeeze().transpose(1, 2, 0) * 255).astype(np.uint8)
                },
            }
            resp = client.infer(client_obs)
            actions = resp["actions"]
            action = actions[0] if actions.ndim > 1 else actions
            obs, reward, term, trunc, info = env.step(action.clip(0, 512))
            if step % 50 == 0:
                timing = resp.get("timing", {})
                logger.info(f"Step {step}: reward={reward:.3f}, infer={timing.get('infer_ms', 0):.1f}ms")
            if term or trunc:
                logger.info(f"Episode done: success={info.get('is_success')}, steps={step + 1}")
                break

        env.close()

    client.close()


if __name__ == "__main__":
    main()
