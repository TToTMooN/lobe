"""Benchmark raw model inference latency for all YAM policy backbones.

Measures pure forward-pass time (model inference only — no deployment
patterns like action chunking amortization or RTC). Reports median
latency across N runs after warmup.

Usage:
    uv run python scripts/bench_inference.py
    uv run python scripts/bench_inference.py --checkpoints dp:path/to/model
    uv run python scripts/bench_inference.py --compile  # test torch.compile speedup
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
from loguru import logger

import lobe  # noqa: F401
import lobe.video_compat  # noqa: F401

DEFAULT_CHECKPOINTS = {
    "dp": "checkpoints/yam-grey-cube-dp-v0/checkpoints/050000/pretrained_model",
    "fm": "checkpoints/yam-grey-cube-fm-v1/checkpoints/050000/pretrained_model",
    "xvla": "checkpoints/yam-grey-cube-xvla-v0/checkpoints/020000/pretrained_model",
    "smolvla": "checkpoints/yam-grey-cube-smolvla-v0/checkpoints/020000/pretrained_model",
}

FAST_CONFIGS = {
    "dp": {"noise_scheduler_type": "DDIM", "num_inference_steps": 10},
    "fm": {"num_inference_steps": 5},
    "xvla": {},
    "smolvla": {},
}

IMAGE_SHAPE = (3, 240, 320)

CAMERA_KEYS = {
    "dp": ["head_camera", "left_wrist_camera", "right_wrist_camera"],
    "fm": ["head_camera", "left_wrist_camera", "right_wrist_camera"],
    "xvla": ["image", "image2", "image3"],
    "smolvla": ["camera1", "camera2", "camera3"],
}


def bench_one(
    backbone: str,
    checkpoint: str,
    device: str,
    compile_model: bool,
    n_warmup: int,
    n_measure: int,
    extra_configs: dict | None = None,
) -> dict:
    import os
    if not os.path.exists(checkpoint):
        return {"backbone": backbone, "error": "checkpoint not found"}

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = checkpoint
    cfg.device = device

    if extra_configs:
        for k, v in extra_configs.items():
            setattr(cfg, k, v)
    if compile_model and hasattr(cfg, "compile_model"):
        cfg.compile_model = True
        cfg.compile_mode = "reduce-overhead"

    cls = get_policy_class(cfg.type)
    policy = cls.from_pretrained(checkpoint, config=cfg)
    policy.to(device)
    policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    cams = CAMERA_KEYS.get(backbone, CAMERA_KEYS["dp"])
    obs = {
        "observation.state": torch.zeros(1, 14),
        "task": "pick up the grey cube",
    }
    for cam in cams:
        obs[f"observation.images.{cam}"] = torch.rand(1, *IMAGE_SHAPE)

    # Warmup (includes torch.compile JIT)
    logger.info(f"  [{backbone}] warming up ({n_warmup} iters, compile={compile_model})...")
    for _ in range(n_warmup):
        policy.reset()
        o = preprocessor(dict(obs))
        policy.select_action(o)

    # Measure
    times = []
    for _ in range(n_measure):
        policy.reset()
        o = preprocessor(dict(obs))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        action = policy.select_action(o)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    action_shape = list(action.shape)
    del policy
    torch.cuda.empty_cache()

    return {
        "backbone": backbone,
        "compile": compile_model,
        "steps": getattr(cfg, "num_inference_steps", None) or getattr(cfg, "num_train_timesteps", "?"),
        "scheduler": getattr(cfg, "noise_scheduler_type", getattr(cfg, "ode_solver", "?")),
        "median_ms": float(np.median(times)),
        "min_ms": float(np.min(times)),
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "action_shape": action_shape,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="*", help="name:path pairs")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--both", action="store_true", help="Benchmark both compiled and uncompiled")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=20)
    args = parser.parse_args()

    checkpoints = dict(DEFAULT_CHECKPOINTS)
    if args.checkpoints:
        checkpoints = {}
        for item in args.checkpoints:
            name, path = item.split(":", 1)
            checkpoints[name] = path

    compile_modes = [False, True] if args.both else [args.compile]

    results = []
    for backbone, ckpt in checkpoints.items():
        fast = FAST_CONFIGS.get(backbone, {})
        for do_compile in compile_modes:
            logger.info(f"[{backbone}] compile={do_compile} configs={fast}")
            r = bench_one(backbone, ckpt, args.device, do_compile, args.warmup, args.measure, fast)
            results.append(r)
            if "error" in r:
                logger.warning(f"  [{backbone}] {r['error']}")
            else:
                logger.info(
                    f"  [{backbone}] compile={do_compile}: "
                    f"median={r['median_ms']:.1f}ms min={r['min_ms']:.1f}ms "
                    f"action={r['action_shape']}"
                )

    logger.info("=" * 70)
    logger.info(f"{'Backbone':<12} {'Compile':<9} {'Steps':<7} {'Median':>8} {'Min':>8} {'Action'}")
    logger.info("-" * 70)
    for r in results:
        if "error" in r:
            logger.info(f"{r['backbone']:<12} {'—':<9} {'—':<7} {'SKIP':>8} {'':>8} {r['error']}")
        else:
            compile_str = "yes" if r["compile"] else "no"
            logger.info(
                f"{r['backbone']:<12} {compile_str:<9} {str(r['steps']):<7} "
                f"{r['median_ms']:>7.1f}ms {r['min_ms']:>7.1f}ms {r['action_shape']}"
            )


if __name__ == "__main__":
    sys.exit(main() or 0)
