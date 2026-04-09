"""WebSocket policy server — serves any lerobot-format policy to limb robots.

Loads policies via lerobot's `make_policy` from a checkpoint directory (the
`pretrained_model/` subdirectory produced by lobe-train / lerobot-train).

Implements limb's WebSocketPolicyClient protocol:
1. Client connects via ws://host:port
2. Server sends metadata (msgpack): {model_name, action_horizon, action_dim, ...}
3. Client sends obs (msgpack): {state: float32, images: {cam: uint8}, prompt: str}
4. Server responds with actions (msgpack): {actions: float32(H, D), timing: {infer_ms}}
5. Loop 3-4 until disconnect

Usage:
    lobe-serve --checkpoint /path/to/checkpoints/050000/pretrained_model
    lobe-serve --checkpoint HuggingFaceVLA/smolvla_libero  # from HF Hub
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import lobe  # noqa: F401 — registers custom policies and applies patches

import msgpack
import msgpack_numpy  # noqa: F401 — registers numpy hooks
import numpy as np
import torch
import tyro
from loguru import logger


@dataclass
class ServeConfig:
    """lobe-serve configuration."""

    checkpoint: str = ""  # Path to checkpoint dir (or HF Hub repo_id)
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cuda"


def _build_obs_batch(obs: dict, device: str, image_keys: list[str]) -> dict[str, torch.Tensor]:
    """Convert client observation dict to policy input batch.

    Expected client format:
        {"state": [...], "images": {"image": uint8 HWC, "image2": ...}, "prompt": "..."}
    """
    batch: dict[str, torch.Tensor] = {}

    if "state" in obs:
        state = np.asarray(obs["state"], dtype=np.float32)
        batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)

    if "images" in obs:
        for cam_name, img in obs["images"].items():
            img = np.asarray(img, dtype=np.uint8)
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.transpose(2, 0, 1)  # HWC -> CHW
            tensor = torch.from_numpy(img.copy()).float().div(255.0).unsqueeze(0).to(device)
            key = f"observation.images.{cam_name}"
            batch[key] = tensor

    if "prompt" in obs:
        batch["task"] = obs["prompt"]

    return batch


class PolicyServer:
    """Async WebSocket server wrapping any lerobot policy."""

    def __init__(self, policy, preprocessor, postprocessor, config: ServeConfig):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.config = config
        self.device = config.device

        # Discover policy details from config
        cfg = policy.config
        self.image_keys = list(cfg.image_features.keys()) if cfg.image_features else []
        action_dim = cfg.output_features["action"].shape[0] if "action" in cfg.output_features else 0

        self.metadata = {
            "model_name": f"lobe-{cfg.type}",
            "policy_type": cfg.type,
            "action_horizon": getattr(cfg, "n_action_steps", 1),
            "action_dim": action_dim,
            "image_keys": self.image_keys,
        }

    async def handle_client(self, websocket):
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")

        # Send metadata on connect
        await websocket.send(msgpack.packb(self.metadata, use_bin_type=True))
        logger.info(f"Sent metadata: {self.metadata}")

        self.policy.reset()

        try:
            async for message in websocket:
                t0 = time.perf_counter()
                obs = msgpack.unpackb(message, raw=False, object_hook=msgpack_numpy.decode)
                batch = _build_obs_batch(obs, self.device, self.image_keys)

                # Apply preprocessor (normalization, batching) then policy then postprocessor
                t_infer = time.perf_counter()
                with torch.no_grad():
                    batch = self.preprocessor(batch)
                    action = self.policy.select_action(batch)
                    action = self.postprocessor(action)
                infer_ms = (time.perf_counter() - t_infer) * 1000

                action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
                if action_np.ndim == 1:
                    action_np = action_np.reshape(1, -1)

                response = {
                    "actions": action_np.astype(np.float32),
                    "timing": {
                        "infer_ms": round(infer_ms, 2),
                        "total_ms": round((time.perf_counter() - t0) * 1000, 2),
                    },
                }
                await websocket.send(msgpack.packb(response, use_bin_type=True, default=msgpack_numpy.encode))

        except Exception as e:
            logger.exception(f"Client {client_addr} error: {e}")
            try:
                await websocket.send(msgpack.packb({"error": str(e)}, use_bin_type=True))
            except Exception:
                pass

        logger.info(f"Client disconnected: {client_addr}")

    async def run(self):
        import websockets

        logger.info(f"Starting policy server on ws://{self.config.host}:{self.config.port}")
        logger.info(f"Metadata: {self.metadata}")

        async with websockets.serve(self.handle_client, self.config.host, self.config.port):
            logger.info("Server ready. Waiting for connections...")
            await asyncio.Future()  # run forever


def main():
    config = tyro.cli(ServeConfig)

    if not config.checkpoint:
        raise ValueError("--checkpoint is required (path to pretrained_model dir or HF repo_id)")

    # Load policy + preprocessor + postprocessor from checkpoint
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    logger.info(f"Loading policy from {config.checkpoint}")
    policy_cfg = PreTrainedConfig.from_pretrained(config.checkpoint)
    policy_cfg.pretrained_path = config.checkpoint
    policy_cfg.device = config.device

    # Use policy_cls.from_pretrained directly (skips dataset_meta requirement)
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(config.checkpoint)
    policy.to(config.device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=config.checkpoint,
    )

    n_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"Loaded policy: {policy_cfg.type} | {n_params:,} params")

    server = PolicyServer(policy, preprocessor, postprocessor, config)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
