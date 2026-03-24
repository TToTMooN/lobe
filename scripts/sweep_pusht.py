"""PushT Sweep — batch eval across inference steps, horizons, action chunks.

Usage:
    uv run python scripts/sweep_pusht.py --checkpoint checkpoints/pusht/flow_matching_50000
    uv run python scripts/sweep_pusht.py --inference-steps 1,2,4,8,16 --n-rollouts 10
    uv run python scripts/sweep_pusht.py --sweep-type policy_compare --checkpoint checkpoints/pusht/flow_matching_50000
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from loguru import logger

import lobe.video_compat  # noqa: F401
from lobe import pusht


@dataclass
class Args:
    policy_type: str = "flow_matching"
    checkpoint: str = ""
    dataset_repo_id: str = pusht.DEFAULT_DATASET
    sweep_type: str = "inference_steps"  # inference_steps | action_steps | horizon | policy_compare
    inference_steps: str = "1,2,4,8,16"
    action_steps: str = "1,2,4,8"
    horizons: str = "8,16,32"
    n_rollouts: int = 3
    max_steps: int = pusht.MAX_STEPS
    seed: int = 42
    device: str = "cuda"
    save_dir: str = "results/"
    wandb: bool = False
    wandb_project: str = "lobe-sweep"


def run_sweep_config(
    policy_type,
    stats,
    features,
    checkpoint,
    device,
    horizon,
    n_action_steps,
    num_inference_steps,
    n_rollouts,
    max_steps,
    seed,
):
    """Run n_rollouts for a single config and return aggregated metrics."""
    results = []
    for i in range(n_rollouts):
        policy = pusht.create_policy(
            policy_type,
            features,
            stats,
            horizon=horizon,
            n_action_steps=n_action_steps,
            num_inference_steps=num_inference_steps,
        )
        pusht.load_checkpoint(policy, checkpoint, device)
        policy.to(device)
        policy.eval()
        r = pusht.run_rollout(policy, device, seed + i, max_steps)
        results.append(r)

    return {
        "avg_reward": float(np.mean([r["avg_reward"] for r in results])),
        "std_reward": float(np.std([r["avg_reward"] for r in results])),
        "avg_max_reward": float(np.mean([r["max_reward"] for r in results])),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "avg_steps": float(np.mean([r["steps"] for r in results])),
        "avg_latency_ms": float(np.mean([r["avg_latency_ms"] for r in results])),
    }


def print_table(title, rows):
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Sweep: {title}")
    logger.info(f"{'=' * 80}")
    logger.info(f"|{'Config':>17} |{'Avg Reward':>14} |{'Success Rate':>14} |{'Latency(ms)':>14} |{'Avg Steps':>14} |")
    logger.info(f"|{'-' * 17}|{'-' * 14}|{'-' * 14}|{'-' * 14}|{'-' * 14}|")
    for r in rows:
        label = r.get("label", "?")
        reward_str = f"{r['avg_reward']:.4f}+/-{r['std_reward']:.4f}"
        success_str = f"{r['success_rate'] * 100:.0f}%"
        logger.info(
            f"|{label:>17} |{reward_str:>14}|{success_str:>14} |{r['avg_latency_ms']:12.2f} |{r['avg_steps']:12.1f} |"
        )
    logger.info(f"{'=' * 80}")


def main():
    args = tyro.cli(Args)
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    logger.info(f"Device: {device}")

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, config=vars(args))

    logger.info("Loading dataset...")
    dataset, features = pusht.load_dataset(args.dataset_repo_id)
    stats = dataset.meta.stats

    rows = []

    if args.sweep_type == "inference_steps":
        values = [int(x.strip()) for x in args.inference_steps.split(",")]
        for v in values:
            logger.info(f"Running: inference_steps={v}")
            r = run_sweep_config(
                args.policy_type,
                stats,
                features,
                args.checkpoint,
                device,
                pusht.HORIZON,
                pusht.N_ACTION_STEPS,
                v,
                args.n_rollouts,
                args.max_steps,
                args.seed,
            )
            r["label"] = f"steps={v}"
            rows.append(r)
            if run:
                metrics = {f"sweep/{k}": val for k, val in r.items() if isinstance(val, (int, float))}
                run.log({"inference_steps": v, **metrics})

    elif args.sweep_type == "action_steps":
        values = [int(x.strip()) for x in args.action_steps.split(",")]
        for v in values:
            logger.info(f"Running: n_action_steps={v}")
            r = run_sweep_config(
                args.policy_type,
                stats,
                features,
                args.checkpoint,
                device,
                pusht.HORIZON,
                v,
                10,
                args.n_rollouts,
                args.max_steps,
                args.seed,
            )
            r["label"] = f"chunk={v}"
            rows.append(r)

    elif args.sweep_type == "horizon":
        values = [int(x.strip()) for x in args.horizons.split(",")]
        for v in values:
            logger.info(f"Running: horizon={v}")
            r = run_sweep_config(
                args.policy_type,
                stats,
                features,
                args.checkpoint,
                device,
                v,
                min(8, v),
                10,
                args.n_rollouts,
                args.max_steps,
                args.seed,
            )
            r["label"] = f"H={v}"
            rows.append(r)

    elif args.sweep_type == "policy_compare":
        for ptype in ["flow_matching", "diffusion"]:
            logger.info(f"Running: policy={ptype}")
            n_steps = 10
            # Derive checkpoint for the other policy type
            ckpt = args.checkpoint
            if ckpt:
                ckpt_path = Path(ckpt)
                this_type = "flow_matching" if "flow_matching" in ckpt_path.name else "diffusion"
                if ptype != this_type:
                    derived = ckpt_path.parent / ckpt_path.name.replace(this_type, ptype)
                    if derived.exists():
                        ckpt = str(derived)
                        logger.info(f"Using derived checkpoint: {ckpt}")
                    else:
                        logger.warning(f"No checkpoint found for {ptype} at {derived}, using random init")
                        ckpt = ""
            r = run_sweep_config(
                ptype,
                stats,
                features,
                ckpt,
                device,
                pusht.HORIZON,
                pusht.N_ACTION_STEPS,
                n_steps,
                args.n_rollouts,
                args.max_steps,
                args.seed,
            )
            r["label"] = f"{ptype}({n_steps})"
            rows.append(r)

    print_table(args.sweep_type, rows)

    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_path = save_dir / f"sweep_{args.sweep_type}.json"
    results_path.write_text(json.dumps(rows, indent=2))
    logger.info(f"Results saved to {results_path}")

    if run:
        run.finish()


if __name__ == "__main__":
    main()
