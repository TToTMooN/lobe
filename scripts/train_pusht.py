"""Train Flow Matching / Diffusion on PushT and save checkpoints for eval.

Trains one or both policies, saves checkpoints to checkpoints/pusht/.
Then use eval_pusht.py or sweep_pusht.py with --checkpoint to evaluate.

Usage:
    # Train both (recommended first run — ~10 min on RTX 5090)
    uv run python scripts/train_pusht.py --policy both --steps 5000

    # Train just flow matching
    uv run python scripts/train_pusht.py --policy flow_matching --steps 5000

    # Then eval:
    uv run python scripts/eval_pusht.py --checkpoint checkpoints/pusht/flow_matching_5000
    uv run python scripts/sweep_pusht.py --checkpoint checkpoints/pusht/flow_matching_5000
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from loguru import logger
from torch.utils.data import DataLoader

import lobe.video_compat  # noqa: F401
from lobe.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lobe.policies.flow_matching.modeling_flow_matching import FlowMatchingPolicy


@dataclass
class Args:
    policy: str = "both"  # flow_matching | diffusion | both
    steps: int = 5000
    batch_size: int = 64
    lr: float = 1e-4
    num_inference_steps: int = 1
    horizon: int = 16
    n_action_steps: int = 8
    log_every: int = 100
    save_every: int = 1000
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "checkpoints/pusht"
    dataset_repo_id: str = "lerobot/pusht"
    wandb: bool = False
    wandb_project: str = "lobe-train"


def load_dataset(repo_id: str):
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
    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, video_backend="pyav")
    features = dataset_to_policy_features(dataset.meta.features)
    return dataset, features


def make_policy(policy_type, features, stats, args):
    input_features = {k: v for k, v in features.items() if v.type != FeatureType.ACTION}
    output_features = {k: v for k, v in features.items() if v.type == FeatureType.ACTION}

    if policy_type == "flow_matching":
        config = FlowMatchingConfig(
            n_obs_steps=2,
            horizon=args.horizon,
            n_action_steps=args.n_action_steps,
            num_inference_steps=args.num_inference_steps,
        )
        config.input_features = input_features
        config.output_features = output_features
        return FlowMatchingPolicy(config, dataset_stats=stats)
    elif policy_type == "diffusion":
        config = DiffusionConfig(n_obs_steps=2, horizon=args.horizon, n_action_steps=args.n_action_steps)
        config.input_features = input_features
        config.output_features = output_features
        return DiffusionPolicy(config, dataset_stats=stats)
    else:
        raise ValueError(f"Unknown: {policy_type}")


def save_checkpoint(policy, output_dir: Path, policy_type: str, step: int):
    ckpt_dir = output_dir / f"{policy_type}_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), ckpt_dir / "model.pt")
    return ckpt_dir


def train_one(policy_type: str, args: Args, dataset, features, run=None):
    logger.info(f"{'=' * 60}")
    logger.info(f"Training: {policy_type} for {args.steps} steps")
    logger.info(f"{'=' * 60}")

    torch.manual_seed(args.seed)
    policy = make_policy(policy_type, features, dataset.meta.stats, args)
    policy.to(args.device)
    policy.train()

    n_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"Parameters: {n_params:,}")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
    )
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=args.lr, weight_decay=1e-6)

    data_iter = iter(dataloader)
    losses = []
    output_dir = Path(args.output_dir)
    t0 = time.perf_counter()

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        loss, _ = policy.forward(batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % args.log_every == 0:
            avg_loss = sum(losses[-args.log_every :]) / min(len(losses), args.log_every)
            elapsed = time.perf_counter() - t0
            steps_per_sec = step / elapsed
            eta = (args.steps - step) / steps_per_sec
            logger.info(
                f"[{policy_type}] Step {step}/{args.steps} | loss: {avg_loss:.6f} "
                f"| {steps_per_sec:.1f} steps/s | ETA: {eta:.0f}s"
            )
            if run:
                run.log({f"{policy_type}/loss": avg_loss, f"{policy_type}/step": step})

        if step % args.save_every == 0:
            ckpt_dir = save_checkpoint(policy, output_dir, policy_type, step)
            logger.info(f"Checkpoint: {ckpt_dir}")

    # Final save
    ckpt_dir = save_checkpoint(policy, output_dir, policy_type, args.steps)
    logger.info(f"Final checkpoint: {ckpt_dir}")

    # Save training metadata
    meta = {
        "policy": policy_type,
        "steps": args.steps,
        "final_loss": sum(losses[-100:]) / min(len(losses), 100),
        "n_params": n_params,
        "elapsed_s": time.perf_counter() - t0,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    logger.info(f"Done: {policy_type} | loss: {meta['final_loss']:.6f} | time: {meta['elapsed_s']:.1f}s")
    return ckpt_dir


def main():
    args = tyro.cli(Args)
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    args.device = device
    logger.info(f"Device: {device}")

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, config=vars(args))

    logger.info("Loading dataset...")
    dataset, features = load_dataset(args.dataset_repo_id)
    logger.info(f"Dataset: {len(dataset)} frames")

    policies_to_train = []
    if args.policy == "both":
        policies_to_train = ["flow_matching", "diffusion"]
    else:
        policies_to_train = [args.policy]

    checkpoints = {}
    for policy_type in policies_to_train:
        ckpt_dir = train_one(policy_type, args, dataset, features, run)
        checkpoints[policy_type] = str(ckpt_dir)

    logger.info(f"\n{'=' * 60}")
    logger.info("All checkpoints:")
    for name, path in checkpoints.items():
        logger.info(f"  {name}: {path}")
    logger.info("\nEval with:")
    for name, path in checkpoints.items():
        logger.info(f"  python scripts/eval_pusht.py --policy-type {name} --checkpoint {path} --mode watch")
    logger.info(f"{'=' * 60}")

    if run:
        run.finish()


if __name__ == "__main__":
    main()
