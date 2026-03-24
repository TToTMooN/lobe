"""PushT Sweep — batch eval across inference steps, horizons, action chunks.

Runs headless rollouts, saves JSON results, prints comparison table.

Usage:
    uv run python scripts/sweep_pusht.py
    uv run python scripts/sweep_pusht.py --inference_steps 1,2,4,8,16 --n_rollouts 5
    uv run python scripts/sweep_pusht.py --sweep_type action_steps --action_steps 1,2,4,8
    uv run python scripts/sweep_pusht.py --wandb
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium
import numpy as np
import torch
import tyro
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from loguru import logger

import lobe.video_compat  # noqa: F401 — patch video decoding for torch nightly
from lobe.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lobe.policies.flow_matching.modeling_flow_matching import FlowMatchingPolicy


@dataclass
class Args:
    policy_type: str = "flow_matching"
    checkpoint: str = ""
    dataset_repo_id: str = "lerobot/pusht_image"
    sweep_type: str = "inference_steps"  # inference_steps | action_steps | horizon | policy_compare
    inference_steps: str = "1,2,4,8,16"
    action_steps: str = "1,2,4,8"
    horizons: str = "8,16,32"
    n_rollouts: int = 3
    max_steps: int = 300
    seed: int = 42
    device: str = "cuda"
    save_dir: str = "results/"
    wandb: bool = False
    wandb_project: str = "lobe-sweep"


def load_dataset_info(repo_id: str):
    fps = 10.0
    n_obs_steps = 2
    horizon = 16
    obs_timestamps = [i / fps for i in range(1 - n_obs_steps, 1)]
    action_timestamps = [i / fps for i in range(1 - n_obs_steps, 1 - n_obs_steps + horizon)]
    delta_timestamps = {
        "observation.image": obs_timestamps,
        "observation.state": obs_timestamps,
        "action": action_timestamps,
    }
    is_video = "image" not in repo_id
    kwargs = {"video_backend": "torchcodec"} if is_video else {}
    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, **kwargs)
    features = dataset_to_policy_features(dataset.meta.features)
    return dataset.meta.stats, features


def make_policy(policy_type, stats, features, horizon, n_action_steps, num_inference_steps, checkpoint, device):
    input_features = {k: v for k, v in features.items() if v.type != FeatureType.ACTION}
    output_features = {k: v for k, v in features.items() if v.type == FeatureType.ACTION}

    if policy_type == "flow_matching":
        config = FlowMatchingConfig(
            n_obs_steps=2,
            horizon=horizon,
            n_action_steps=n_action_steps,
            num_inference_steps=num_inference_steps,
        )
        config.input_features = input_features
        config.output_features = output_features
        policy = FlowMatchingPolicy(config, dataset_stats=stats)
    elif policy_type == "diffusion":
        config = DiffusionConfig(n_obs_steps=2, horizon=horizon, n_action_steps=n_action_steps)
        config.input_features = input_features
        config.output_features = output_features
        policy = DiffusionPolicy(config, dataset_stats=stats)
    else:
        raise ValueError(f"Unknown: {policy_type}")

    if checkpoint and Path(checkpoint).exists():
        ckpt = Path(checkpoint)
        if ckpt.is_dir():
            for name in ["model.pt", "model.safetensors", "pytorch_model.bin"]:
                if (ckpt / name).exists():
                    ckpt = ckpt / name
                    break
        if ckpt.is_file():
            state_dict = torch.load(ckpt, map_location=device, weights_only=True)
            # Strip _orig_mod. prefix from torch.compile if present
            state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
            policy.load_state_dict(state_dict)
            logger.info(f"Loaded checkpoint: {ckpt}")

    policy.to(device)
    policy.eval()
    return policy


def obs_to_batch(obs, device):
    processed = preprocess_observation(obs)
    return {k: v.to(device) for k, v in processed.items()}


def run_rollout(policy, device, seed, max_steps):
    env = gymnasium.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels_agent_pos")
    obs, info = env.reset(seed=seed)
    policy.reset()

    rewards = []
    latencies = []

    for _ in range(max_steps):
        batch = obs_to_batch(obs, device)
        t0 = time.perf_counter()
        with torch.no_grad():
            action = policy.select_action(batch)
        latencies.append(time.perf_counter() - t0)

        action_np = action[0].cpu().numpy().clip(0, 512)
        obs, reward, terminated, truncated, info = env.step(action_np)
        rewards.append(reward)
        if terminated or truncated:
            break

    env.close()
    return {
        "avg_reward": float(np.mean(rewards)),
        "max_reward": float(np.max(rewards)),
        "success": bool(info.get("is_success", False)),
        "steps": len(rewards),
        "avg_latency_ms": float(np.mean(latencies) * 1000),
    }


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
        policy = make_policy(
            policy_type, stats, features, horizon, n_action_steps, num_inference_steps, checkpoint, device
        )
        r = run_rollout(policy, device, seed + i, max_steps)
        results.append(r)

    return {
        "avg_reward": float(np.mean([r["avg_reward"] for r in results])),
        "std_reward": float(np.std([r["avg_reward"] for r in results])),
        "avg_max_reward": float(np.mean([r["max_reward"] for r in results])),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "avg_steps": float(np.mean([r["steps"] for r in results])),
        "avg_latency_ms": float(np.mean([r["avg_latency_ms"] for r in results])),
    }


def print_table(rows: list[dict], title: str):
    logger.info(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    header = "| {:>15} | {:>12} | {:>12} | {:>12} | {:>12} |".format(
        "Config", "Avg Reward", "Success Rate", "Latency(ms)", "Avg Steps"
    )
    logger.info(header)
    logger.info("|" + "-" * 17 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 14 + "|")
    for row in rows:
        line = "| {:>15} | {:>8.4f}+/-{:<4.4f}| {:>11.0%} | {:>12.2f} | {:>12.1f} |".format(
            row["label"],
            row["avg_reward"],
            row["std_reward"],
            row["success_rate"],
            row["avg_latency_ms"],
            row["avg_steps"],
        )
        logger.info(line)
    logger.info("=" * 80)


def main():
    args = tyro.cli(Args)
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    logger.info(f"Device: {device}")

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, config=vars(args), name=f"sweep-{args.sweep_type}")

    logger.info("Loading dataset...")
    stats, features = load_dataset_info(args.dataset_repo_id)

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
                16,
                8,
                v,
                args.n_rollouts,
                args.max_steps,
                args.seed,
            )
            r["label"] = f"steps={v}"
            r["num_inference_steps"] = v
            rows.append(r)
            if run:
                run.log(
                    {
                        "inference_steps": v,
                        **{f"sweep/{k}": val for k, val in r.items() if isinstance(val, (int, float))},
                    }
                )

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
                16,
                v,
                args.num_inference_steps if hasattr(args, "num_inference_steps") else 1,
                args.n_rollouts,
                args.max_steps,
                args.seed,
            )
            r["label"] = f"chunk={v}"
            r["n_action_steps"] = v
            rows.append(r)
            if run:
                run.log(
                    {
                        "n_action_steps": v,
                        **{f"sweep/{k}": val for k, val in r.items() if isinstance(val, (int, float))},
                    }
                )

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
                1,
                args.n_rollouts,
                args.max_steps,
                args.seed,
            )
            r["label"] = f"H={v}"
            r["horizon"] = v
            rows.append(r)
            if run:
                run.log({"horizon": v, **{f"sweep/{k}": val for k, val in r.items() if isinstance(val, (int, float))}})

    elif args.sweep_type == "policy_compare":
        for ptype in ["flow_matching", "diffusion"]:
            logger.info(f"Running: policy={ptype}")
            n_steps = 1 if ptype == "flow_matching" else 10
            # Resolve checkpoint: if given a flow_matching checkpoint, derive the diffusion one and vice versa
            ckpt = args.checkpoint
            if ckpt:
                ckpt_path = Path(ckpt)
                # Try replacing policy type in checkpoint path (e.g. flow_matching_5000 -> diffusion_5000)
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
                16,
                8,
                n_steps,
                args.n_rollouts,
                args.max_steps,
                args.seed,
            )
            r["label"] = f"{ptype}({n_steps})"
            r["policy_type"] = ptype
            rows.append(r)
            if run:
                run.log(
                    {
                        "policy_type": ptype,
                        **{f"sweep/{k}": val for k, val in r.items() if isinstance(val, (int, float))},
                    }
                )

    print_table(rows, f"Sweep: {args.sweep_type}")

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
