"""WebSocket policy server — serves trained policies to limb robots.

Implements limb's WebSocketPolicyClient protocol:
1. Client connects via ws://host:port
2. Server sends metadata (msgpack): {model_name, action_horizon, action_dim, ...}
3. Client sends obs (msgpack): {state: float32, images: {cam: uint8}, prompt: str}
4. Server responds with actions (msgpack): {actions: float32(H, D), timing: {infer_ms}}
5. Loop 3-4 until disconnect

Compatible with limb's WebSocketPolicyClient on port 8000.

Usage:
    uv run python -m lobe.serve --checkpoint checkpoints/pusht_final/flow_matching_50000
    uv run python -m lobe.serve --checkpoint path/to/xvla --policy-type flow_matching
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import msgpack
import msgpack_numpy
import numpy as np
import torch
import tyro
from loguru import logger

from lobe.policies.factory import create_policy, load_checkpoint

# Register msgpack-numpy hooks
msgpack_numpy.patch()


@dataclass
class ServeConfig:
    checkpoint: str = ""
    policy_type: str = "flow_matching"  # flow_matching | diffusion
    # Policy config (must match checkpoint architecture)
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    num_inference_steps: int = 10
    # Dataset for normalization stats
    dataset_repo_id: str = "lerobot/pusht_image"
    env_name: str = "pusht"
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cuda"
    compile: bool = False


class PolicyServer:
    """Async WebSocket server wrapping any LeRobot policy."""

    def __init__(self, policy, config: ServeConfig):
        self.policy = policy
        self.config = config
        self.device = config.device

        # Metadata sent to client on connect
        self.metadata = {
            "model_name": f"lobe-{config.policy_type}",
            "policy_type": config.policy_type,
            "action_horizon": config.n_action_steps,
            "action_dim": self._get_action_dim(),
            "num_inference_steps": config.num_inference_steps,
            "env_name": config.env_name,
        }

    def _get_action_dim(self) -> int:
        """Infer action dim from policy config."""
        try:
            return self.policy.config.action_feature.shape[0]
        except Exception:
            return 0

    def _obs_to_batch(self, obs: dict) -> dict[str, torch.Tensor]:
        """Convert raw observation dict from client to policy input batch.

        Follows lerobot's preprocess_observation convention:
        - state -> observation.state (1, D) float32
        - images.{cam} -> observation.image (1, C, H, W) float32 [0, 1]
        """
        batch = {}

        if "state" in obs:
            state = np.asarray(obs["state"], dtype=np.float32)
            batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(self.device)

        if "images" in obs:
            images = []
            for cam_name, img in obs["images"].items():
                img = np.asarray(img, dtype=np.uint8)
                if img.ndim == 3 and img.shape[-1] == 3:
                    # HWC -> CHW
                    img = img.transpose(2, 0, 1)
                img_tensor = torch.from_numpy(img.copy()).float() / 255.0
                images.append(img_tensor)
            if images:
                # Stack as (1, C, H, W) for single image or handle multiple
                batch["observation.image"] = images[0].unsqueeze(0).to(self.device)

        return batch

    async def handle_client(self, websocket):
        """Handle a single client connection."""
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")

        # Send metadata on connect
        meta_bytes = msgpack.packb(self.metadata, use_bin_type=True)
        await websocket.send(meta_bytes)
        logger.info(f"Sent metadata: {self.metadata}")

        self.policy.reset()

        try:
            async for message in websocket:
                t0 = time.perf_counter()

                # Decode observation
                obs = msgpack.unpackb(message, raw=False)

                # Convert to policy batch
                batch = self._obs_to_batch(obs)

                # Inference
                t_infer = time.perf_counter()
                with torch.no_grad():
                    action = self.policy.select_action(batch)
                infer_ms = (time.perf_counter() - t_infer) * 1000

                # Convert to numpy
                action_np = action.cpu().numpy()
                if action_np.ndim == 1:
                    # Single action -> expand to (1, D) for consistency
                    action_np = action_np.reshape(1, -1)

                # Build response
                response = {
                    "actions": action_np.astype(np.float32),
                    "timing": {
                        "infer_ms": round(infer_ms, 2),
                        "total_ms": round((time.perf_counter() - t0) * 1000, 2),
                    },
                }

                # Send response
                resp_bytes = msgpack.packb(response, use_bin_type=True)
                await websocket.send(resp_bytes)

        except Exception as e:
            logger.error(f"Client {client_addr} error: {e}")
            try:
                error_resp = msgpack.packb({"error": str(e)}, use_bin_type=True)
                await websocket.send(error_resp)
            except Exception:
                pass

        logger.info(f"Client disconnected: {client_addr}")

    async def run(self):
        """Start the WebSocket server."""
        import websockets

        logger.info(f"Starting policy server on ws://{self.config.host}:{self.config.port}")
        logger.info(f"Policy: {self.config.policy_type} | Device: {self.config.device}")
        logger.info(f"Metadata: {self.metadata}")

        async with websockets.serve(self.handle_client, self.config.host, self.config.port):
            logger.info("Server ready. Waiting for connections...")
            await asyncio.Future()  # run forever


def main():
    config = tyro.cli(ServeConfig)

    # Load env module for dataset stats
    from lobe.envs import get_env

    env_module = get_env(config.env_name)
    dataset, features = env_module.load_dataset(config.dataset_repo_id)

    # Create and load policy
    # Create uncompiled first, load weights, then compile (avoids _orig_mod key mismatch)
    policy = create_policy(
        config.policy_type,
        features,
        dataset.meta.stats,
        n_obs_steps=config.n_obs_steps,
        horizon=config.horizon,
        n_action_steps=config.n_action_steps,
        num_inference_steps=config.num_inference_steps,
        compile_model=False,
    )

    if config.checkpoint:
        load_checkpoint(policy, config.checkpoint, config.device)
    else:
        logger.warning("No checkpoint specified — serving random policy")

    policy.to(config.device)
    policy.eval()

    if config.compile:
        logger.info("Compiling model with torch.compile (first inference will be slow)...")
        policy.flow_matching.unet = torch.compile(policy.flow_matching.unet, mode="reduce-overhead")

    n_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"Loaded policy: {n_params:,} params")

    # Start server
    server = PolicyServer(policy, config)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
