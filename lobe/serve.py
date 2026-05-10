"""WebSocket policy server — serves any lerobot-format policy to limb robots.

Loads policies via lerobot's `from_pretrained` from a checkpoint directory (the
`pretrained_model/` subdirectory produced by lobe-train / lerobot-train).

Supports two modes:

1. **Single-action mode** (default): client sends one observation, server returns one action.
   Works with all policies.

2. **Chunk mode** (`--chunk-mode=true`): server returns the full action chunk per inference.
   Optionally enables Real-Time Chunking (RTC) when `--rtc=true` for smoother streaming
   inference with flow-matching policies (SmolVLA, pi0, FM). RTC uses leftover actions
   from the previous chunk to guide generation of the next chunk via prefix attention.

Implements limb's WebSocketPolicyClient protocol:
1. Client connects via ws://host:port
2. Server sends metadata (msgpack): {model_name, action_horizon, action_dim, ...}
3. Client sends obs (msgpack): {state: float32, images: {cam: uint8}, prompt: str}
4. Server responds with action(s) (msgpack): {actions: float32(H, D), timing: {infer_ms}}
5. Loop 3-4 until disconnect

Usage:
    lobe-serve --checkpoint=/path/to/pretrained_model
    lobe-serve --checkpoint=HuggingFaceVLA/smolvla_libero --chunk-mode --rtc
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import msgpack
import msgpack_numpy  # noqa: F401 — registers numpy hooks
import numpy as np
import torch
import tyro
from loguru import logger

import lobe  # noqa: F401 — registers custom policies and applies patches


@dataclass
class ServeConfig:
    """lobe-serve configuration."""

    checkpoint: str = ""  # Path to checkpoint dir (or HF Hub repo_id)
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cuda"

    # Inference speed — override denoising/ODE steps for faster serving
    num_inference_steps: int | None = None  # e.g. 10 for DDIM-10 (DP) or 3-5 for FM. None=use checkpoint default.
    noise_scheduler_type: str | None = None  # e.g. "DDIM" for DP (default DDPM is 100 steps = 450ms)
    compile: bool = False  # Enable torch.compile for ~2× inference speedup (adds warmup latency on first call)

    # Gripper binarization — threshold continuous gripper predictions to {0, max}
    gripper_binarize: bool = False  # Enable for YAM bimanual (gripper actions are bimodal 0/2.4)
    gripper_dims: tuple[int, ...] = ()  # Action dims to binarize, e.g. (6, 13) for YAM left/right grippers
    gripper_threshold: float = 0.5  # Threshold in normalized [0, 1] space (after min-max to [0, max])

    # Inference mode
    chunk_mode: bool = True  # Return full action chunk per inference (recommended for real robots)

    # RTC (only valid with chunk_mode=True and policies that support it: SmolVLA, pi0, pi0_fast, pi05)
    rtc: bool = False  # Enable Real-Time Chunking for smoother streaming inference
    rtc_max_guidance_weight: float = 10.0
    rtc_execution_horizon: int = 10
    rtc_inference_latency: float = 0.0  # Estimated server inference time in seconds (auto-tracked if 0)


def _build_obs_batch(obs: dict, device: str) -> dict[str, torch.Tensor]:
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
    """Async WebSocket server wrapping any lerobot policy.

    Three modes:
    - single-action: server returns one action per obs (default)
    - chunk: server returns the full action horizon per obs
    - chunk + RTC: server uses Real-Time Chunking for smoother streaming
    """

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
        action_horizon = getattr(cfg, "n_action_steps", 1)

        # RTC setup
        self.rtc_enabled = config.rtc and config.chunk_mode
        self.rtc_supported = hasattr(policy, "predict_action_chunk") and "rtc" in cfg.__class__.__name__.lower()

        if self.rtc_enabled and not hasattr(policy, "predict_action_chunk"):
            logger.warning(
                f"RTC requested but policy {cfg.type} does not support predict_action_chunk. "
                "Falling back to chunk mode without RTC."
            )
            self.rtc_enabled = False

        # Action queue for RTC (and inflight tracking)
        self._action_queue = None
        self._prev_chunk: torch.Tensor | None = None
        self._fps = getattr(cfg, "fps", 30)
        self._inference_latency = config.rtc_inference_latency
        self._latency_samples: list[float] = []  # rolling window of recent inference times

        self.metadata = {
            "model_name": f"lobe-{cfg.type}",
            "policy_type": cfg.type,
            "action_horizon": action_horizon,
            "action_dim": action_dim,
            "image_keys": self.image_keys,
            "chunk_mode": config.chunk_mode,
            "rtc_enabled": self.rtc_enabled,
        }

    def _record_latency(self, latency_s: float):
        """Track recent inference latencies for RTC delay computation."""
        self._latency_samples.append(latency_s)
        if len(self._latency_samples) > 20:
            self._latency_samples.pop(0)
        if self.config.rtc_inference_latency == 0:
            self._inference_latency = sum(self._latency_samples) / len(self._latency_samples)

    def _compute_rtc_delay(self) -> int:
        """Compute how many actions were 'consumed' during the last inference.

        With FPS=30 and latency=33ms, this is ~1 action.
        """
        return max(1, int(self._inference_latency * self._fps))

    @torch.no_grad()
    def _infer_actions(self, batch: dict) -> torch.Tensor:
        """Run inference and return action(s).

        Returns shape (action_horizon, action_dim) for chunk mode, or (action_dim,) for single mode.
        """
        batch = self.preprocessor(batch)

        if self.config.chunk_mode:
            # Policies with internal observation queues (DP, FM) need them populated
            # before predict_action_chunk works. select_action handles this via
            # populate_queues; predict_action_chunk reads from them directly.
            from lerobot.policies.utils import populate_queues

            if hasattr(self.policy, "_queues"):
                if "action" in batch:
                    batch.pop("action")
                if self.policy.config.image_features:
                    batch = dict(batch)
                    obs_images_key = "observation.images"
                    img_keys = list(self.policy.config.image_features.keys())
                    if img_keys and img_keys[0] in batch:
                        batch[obs_images_key] = torch.stack([batch[k] for k in img_keys], dim=-4)
                self.policy._queues = populate_queues(self.policy._queues, batch)

            kwargs = {}
            if self.rtc_enabled:
                kwargs["prev_chunk_left_over"] = self._prev_chunk
                kwargs["inference_delay"] = self._compute_rtc_delay()
                kwargs["execution_horizon"] = self.config.rtc_execution_horizon

            try:
                actions = self.policy.predict_action_chunk(batch, **kwargs)
            except TypeError:
                actions = self.policy.predict_action_chunk(batch)

            # actions shape: (1, horizon, action_dim) → strip batch dim
            actions = actions.squeeze(0)
        else:
            actions = self.policy.select_action(batch)

        actions = self.postprocessor(actions)

        # Update prev_chunk for next RTC iteration
        if self.rtc_enabled and self.config.chunk_mode:
            delay = self._compute_rtc_delay()
            self._prev_chunk = actions[delay:].clone() if actions.dim() == 2 else None

        return actions

    async def handle_client(self, websocket):
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")

        # Reset state for new client
        self.policy.reset()
        self._prev_chunk = None
        self._latency_samples.clear()

        # Send metadata on connect
        await websocket.send(msgpack.packb(self.metadata, use_bin_type=True))
        logger.info(f"Sent metadata: {self.metadata}")

        try:
            async for message in websocket:
                t0 = time.perf_counter()
                obs = msgpack.unpackb(message, raw=False, object_hook=msgpack_numpy.decode)
                batch = _build_obs_batch(obs, self.device)

                t_infer = time.perf_counter()
                actions = self._infer_actions(batch)
                infer_s = time.perf_counter() - t_infer
                self._record_latency(infer_s)

                if isinstance(actions, torch.Tensor):
                    actions_np = actions.cpu().float().numpy()
                else:
                    actions_np = np.asarray(actions)
                if actions_np.ndim == 1:
                    actions_np = actions_np.reshape(1, -1)

                if self.config.gripper_binarize and self.config.gripper_dims:
                    for d in self.config.gripper_dims:
                        if d < actions_np.shape[-1]:
                            col = actions_np[..., d]
                            col_max = col.max() if col.max() > 0.1 else 2.4
                            actions_np[..., d] = np.where(
                                col > col_max * self.config.gripper_threshold, col_max, 0.0
                            )

                response = {
                    "actions": actions_np.astype(np.float32),
                    "timing": {
                        "infer_ms": round(infer_s * 1000, 2),
                        "total_ms": round((time.perf_counter() - t0) * 1000, 2),
                        "rtc_delay": self._compute_rtc_delay() if self.rtc_enabled else 0,
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
        if self.rtc_enabled:
            logger.info(
                f"RTC enabled: max_guidance={self.config.rtc_max_guidance_weight}, "
                f"execution_horizon={self.config.rtc_execution_horizon}, "
                f"latency={self._inference_latency * 1000:.0f}ms"
            )

        async with websockets.serve(self.handle_client, self.config.host, self.config.port):
            logger.info("Server ready. Waiting for connections...")
            await asyncio.Future()  # run forever


def main():
    config = tyro.cli(ServeConfig)

    if not config.checkpoint:
        raise ValueError("--checkpoint is required (path to pretrained_model dir or HF repo_id)")

    if config.rtc and not config.chunk_mode:
        logger.warning("RTC requires --chunk-mode; auto-enabling chunk mode.")
        config.chunk_mode = True

    # Load policy + preprocessor + postprocessor from checkpoint
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    logger.info(f"Loading policy from {config.checkpoint}")
    policy_cfg = PreTrainedConfig.from_pretrained(config.checkpoint)
    policy_cfg.pretrained_path = config.checkpoint
    policy_cfg.device = config.device

    # Override RTC config on the policy if requested
    if config.rtc and hasattr(policy_cfg, "rtc_config"):
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        policy_cfg.rtc_config = RTCConfig(
            enabled=True,
            max_guidance_weight=config.rtc_max_guidance_weight,
            execution_horizon=config.rtc_execution_horizon,
        )
        logger.info(f"Set policy.rtc_config to {policy_cfg.rtc_config}")

    # Override inference speed settings
    if config.num_inference_steps is not None:
        policy_cfg.num_inference_steps = config.num_inference_steps
        logger.info(f"Override num_inference_steps={config.num_inference_steps}")
    if config.noise_scheduler_type is not None and hasattr(policy_cfg, "noise_scheduler_type"):
        policy_cfg.noise_scheduler_type = config.noise_scheduler_type
        logger.info(f"Override noise_scheduler_type={config.noise_scheduler_type}")
    if config.compile and hasattr(policy_cfg, "compile_model"):
        policy_cfg.compile_model = True
        logger.info("Enabled torch.compile for inference")

    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(config.checkpoint, config=policy_cfg)
    policy.to(config.device)
    policy.eval()

    # Initialize RTC processor if the policy supports it
    if config.rtc and hasattr(policy, "init_rtc_processor"):
        policy.init_rtc_processor()
        logger.info(f"Initialized RTC processor for {policy_cfg.type}")

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
