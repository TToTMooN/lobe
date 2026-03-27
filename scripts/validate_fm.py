"""Validate FM policy across configurations — run after code changes to verify nothing is broken.

Tests multiple (backbone, normalization, env) combinations and reports success rates.
Each run: 10k steps (fast), 10 eval rollouts. Not meant for SOTA — just sanity checking.

Usage:
    uv run python scripts/validate_fm.py                    # all tests
    uv run python scripts/validate_fm.py --tests 1,2        # specific tests
    uv run python scripts/validate_fm.py --steps 25000      # longer training
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from lerobot.configs.types import NormalizationMode
from loguru import logger

import lobe.video_compat  # noqa: F401
from lobe.envs import get_env
from lobe.policies.factory import create_policy
from lobe.policies.normalize import Normalize, Unnormalize  # noqa: F401

# ── Test configurations ──────────────────────────────────────────────────────

TESTS = {
    1: {"backbone": "transformer", "norm": "MEAN_STD", "env": "pusht", "desc": "transformer+MEAN_STD on PushT"},
    2: {"backbone": "unet", "norm": "MEAN_STD", "env": "pusht", "desc": "unet+MEAN_STD on PushT"},
    3: {"backbone": "unet", "norm": "MIN_MAX", "env": "pusht", "desc": "unet+MIN_MAX on PushT (regression)"},
    4: {
        "backbone": "transformer",
        "norm": "MEAN_STD",
        "env": "libero",
        "desc": "transformer+MEAN_STD on LIBERO-10",
    },
    5: {
        "backbone": "unet",
        "norm": "MEAN_STD",
        "env": "libero",
        "desc": "unet+MEAN_STD on LIBERO-10",
    },
}

ENV_CONFIGS = {
    "pusht": {
        "name": "pusht",
        "dataset": "lerobot/pusht_image",
        "horizon": 16,
        "n_action_steps": 8,
        "n_obs_steps": 2,
        "resize_shape": None,
        "eval_task": "",
    },
    "libero": {
        "name": "libero",
        "dataset": "HuggingFaceVLA/libero",
        "horizon": 16,
        "n_action_steps": 8,
        "n_obs_steps": 1,
        "resize_shape": (224, 224),
        "eval_task": "",
    },
}


@dataclass
class Args:
    tests: str = ""  # comma-separated test IDs, empty = all
    steps: int = 10000
    batch_size: int = 256
    eval_rollouts: int = 10
    device: str = "cuda"


def run_test(test_id: int, test: dict, args: Args) -> dict:
    env_cfg = ENV_CONFIGS[test["env"]]
    env_module = get_env(env_cfg["name"])

    logger.info(f"Loading dataset: {env_cfg['dataset']}")
    # Build delta_timestamps matching our horizon/n_obs_steps (not env defaults)
    fps = env_module.FPS if hasattr(env_module, "FPS") else 10.0
    n_obs = env_cfg["n_obs_steps"]
    horizon = env_cfg["horizon"]
    dt = env_module.delta_timestamps()
    # Override action timestamps with our horizon
    dt["action"] = [i / fps for i in range(1 - n_obs, 1 - n_obs + horizon)]
    # Override obs timestamps with our n_obs_steps
    obs_ts = [i / fps for i in range(1 - n_obs, 1)]
    for k in list(dt.keys()):
        if k != "action":
            dt[k] = obs_ts
    from lobe.data.loading import load_lerobot_dataset

    dataset, features = load_lerobot_dataset(env_cfg["dataset"], dt)

    # Override normalization if needed
    norm_mode = NormalizationMode.MEAN_STD if test["norm"] == "MEAN_STD" else NormalizationMode.MIN_MAX

    policy = create_policy(
        "flow_matching",
        features,
        dataset.meta.stats,
        n_obs_steps=env_cfg["n_obs_steps"],
        horizon=env_cfg["horizon"],
        n_action_steps=env_cfg["n_action_steps"],
        num_inference_steps=10,
        compile_model=False,
        resize_shape=env_cfg["resize_shape"],
        fm_backbone=test["backbone"],
    )

    # Override normalization mapping after creation
    norm_map = {"VISUAL": NormalizationMode.MEAN_STD, "STATE": norm_mode, "ACTION": norm_mode}
    policy.config.normalization_mapping = norm_map
    # Rebuild normalize/unnormalize with new mapping
    policy.normalize_inputs = type(policy.normalize_inputs)(policy.config.input_features, norm_map, dataset.meta.stats)
    policy.normalize_targets = type(policy.normalize_targets)(
        policy.config.output_features, norm_map, dataset.meta.stats
    )
    policy.unnormalize_outputs = type(policy.unnormalize_outputs)(
        policy.config.output_features, norm_map, dataset.meta.stats
    )

    policy.to(args.device)
    n_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # Train
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4, weight_decay=1e-6)

    policy.train()
    data_iter = iter(dataloader)
    t0 = time.perf_counter()
    losses = []

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
            loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())

        if step % 1000 == 0:
            avg = sum(losses[-1000:]) / min(len(losses), 1000)
            sps = step / (time.perf_counter() - t0)
            logger.info(f"  Step {step}/{args.steps} | loss={avg:.6f} | {sps:.1f} steps/s")

    elapsed = time.perf_counter() - t0
    final_loss = sum(losses[-100:]) / min(len(losses), 100)

    # Eval
    has_eval = hasattr(env_module, "evaluate")
    success_rate, avg_reward = 0.0, 0.0
    if has_eval:
        policy.eval()
        kwargs = {"task": env_cfg["eval_task"]} if env_cfg["eval_task"] else {}
        success_rate, avg_reward = env_module.evaluate(policy, args.device, n_rollouts=args.eval_rollouts, **kwargs)
        logger.info(f"  Eval: success={success_rate * 100:.0f}%, reward={avg_reward:.3f}")

    # Log to experiments.tsv
    from lobe.experiment_log import log_experiment

    gpu_name = torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU"
    log_experiment(
        env=env_cfg["name"],
        policy="flow_matching",
        backbone=test["backbone"],
        norm=test["norm"],
        steps=args.steps,
        batch_size=args.batch_size,
        n_params=n_params,
        final_loss=final_loss,
        success_rate=success_rate,
        avg_reward=avg_reward,
        train_s=elapsed,
        gpu=gpu_name,
        notes=f"validate_fm test {test_id}",
    )

    return {
        "test_id": test_id,
        "desc": test["desc"],
        "backbone": test["backbone"],
        "norm": test["norm"],
        "env": test["env"],
        "n_params": n_params,
        "final_loss": final_loss,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "steps": args.steps,
        "elapsed_s": elapsed,
        "steps_per_s": args.steps / elapsed if elapsed > 0 else 0,
    }


def main():
    args = tyro.cli(Args)

    if args.tests:
        test_ids = [int(x) for x in args.tests.split(",")]
    else:
        test_ids = sorted(TESTS.keys())

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        logger.warning("CUDA not available, using CPU")

    results = []
    for tid in test_ids:
        test = TESTS[tid]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Test {tid}: {test['desc']}")
        logger.info(f"{'=' * 60}")
        try:
            result = run_test(tid, test, args)
            results.append(result)
        except Exception as e:
            import traceback

            logger.error(f"Test {tid} FAILED: {e}\n{traceback.format_exc()}")
            results.append({"test_id": tid, "desc": test["desc"], "error": str(e)})

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'=' * 60}")
    for r in results:
        if "error" in r:
            logger.info(f"  [{r['test_id']}] {r['desc']}: FAILED — {r['error']}")
        else:
            sps = r.get("steps_per_s", r["steps"] / r["elapsed_s"] if r["elapsed_s"] > 0 else 0)
            logger.info(
                f"  [{r['test_id']}] {r['desc']}: "
                f"success={r['success_rate'] * 100:.0f}% loss={r['final_loss']:.6f} "
                f"| {sps:.1f} steps/s | {r['elapsed_s']:.0f}s | {r['n_params']:,} params"
            )

    # Save results
    out = Path("checkpoints/validation_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    logger.info(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
