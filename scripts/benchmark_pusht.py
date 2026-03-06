"""Benchmark: Flow Matching vs Diffusion Policy on PushT.

Downloads the PushT dataset, trains both policies with identical hyperparameters,
and compares loss curves + inference speed. Logs to wandb if enabled.

Usage:
    uv run python scripts/benchmark_pusht.py --policy flow_matching --steps 5000
    uv run python scripts/benchmark_pusht.py --policy diffusion --steps 5000
    uv run python scripts/benchmark_pusht.py --policy flow_matching --steps 5000 --wandb
"""

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

import lobe.video_compat  # noqa: F401 — patch video decoding for torch nightly
from lobe.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lobe.policies.flow_matching.modeling_flow_matching import FlowMatchingPolicy


@dataclass
class Args:
    policy: str = "flow_matching"
    dataset_repo_id: str = "lerobot/pusht"
    steps: int = 5000
    batch_size: int = 64
    lr: float = 1e-4
    num_inference_steps: int = 1
    use_optimal_transport: bool = False
    log_every: int = 100
    eval_every: int = 1000
    device: str = "cuda"
    seed: int = 42
    wandb: bool = False
    wandb_project: str = "lobe-benchmark"
    wandb_entity: str = ""
    save_results: str = "results/"


def make_policy(args: Args, dataset: LeRobotDataset):
    features = dataset_to_policy_features(dataset.meta.features)
    input_features = {k: v for k, v in features.items() if v.type != FeatureType.ACTION}
    output_features = {k: v for k, v in features.items() if v.type == FeatureType.ACTION}

    if args.policy == "flow_matching":
        config = FlowMatchingConfig(
            n_obs_steps=2,
            horizon=16,
            n_action_steps=8,
            num_inference_steps=args.num_inference_steps,
            use_optimal_transport=args.use_optimal_transport,
        )
        config.input_features = input_features
        config.output_features = output_features
        config.device = args.device
        policy = FlowMatchingPolicy(config, dataset_stats=dataset.meta.stats)
    elif args.policy == "diffusion":
        config = DiffusionConfig(
            n_obs_steps=2,
            horizon=16,
            n_action_steps=8,
        )
        config.input_features = input_features
        config.output_features = output_features
        config.device = args.device
        policy = DiffusionPolicy(config, dataset_stats=dataset.meta.stats)
    else:
        raise ValueError(f"Unknown policy: {args.policy}")

    return policy


def benchmark_inference(policy, device, n_runs=50):
    """Measure inference latency."""
    policy.eval()

    # Create dummy batch with all input features
    batch = {}
    for key, feat in policy.config.input_features.items():
        batch[key] = torch.randn(1, *feat.shape, device=device)

    # Warmup
    policy.reset()
    for _ in range(3):
        policy.select_action(batch)
        policy.reset()

    # Benchmark
    torch.cuda.synchronize() if device == "cuda" else None
    times = []
    for _ in range(n_runs):
        policy.reset()
        start = time.perf_counter()
        policy.select_action(batch)
        torch.cuda.synchronize() if device == "cuda" else None
        times.append(time.perf_counter() - start)

    avg_ms = sum(times) / len(times) * 1000
    std_ms = (sum((t * 1000 - avg_ms) ** 2 for t in times) / len(times)) ** 0.5
    return avg_ms, std_ms


def main():
    args = tyro.cli(Args)
    torch.manual_seed(args.seed)

    logger.info(f"Policy: {args.policy}")
    logger.info(f"Dataset: {args.dataset_repo_id}")
    logger.info(f"Steps: {args.steps}, batch_size: {args.batch_size}")

    # wandb init
    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=f"{args.policy}-{args.steps}steps",
            config={
                "policy": args.policy,
                "dataset": args.dataset_repo_id,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "num_inference_steps": args.num_inference_steps,
                "use_optimal_transport": args.use_optimal_transport,
                "seed": args.seed,
                "device": args.device,
            },
        )

    # Load dataset with delta_timestamps for multi-step obs/action windows.
    # PushT runs at 10 Hz, so 0.1s = 1 frame.
    # n_obs_steps=2 -> [-0.1, 0.0], horizon=16 -> 16 steps into the future
    logger.info("Loading dataset...")
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
    dataset = LeRobotDataset(args.dataset_repo_id, delta_timestamps=delta_timestamps, video_backend="pyav")
    logger.info(f"Dataset: {len(dataset)} frames, features: {list(dataset.meta.features.keys())}")

    # Create policy
    logger.info("Creating policy...")
    policy = make_policy(args, dataset)
    policy.to(args.device)
    n_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=args.lr, weight_decay=1e-6)

    # Training loop
    logger.info("Starting training...")
    policy.train()
    data_iter = iter(dataloader)
    losses = []

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
            logger.info(f"Step {step}/{args.steps} | loss: {avg_loss:.6f}")
            if run:
                run.log({"train/loss": avg_loss, "train/step": step})

        if step % args.eval_every == 0:
            avg_ms, std_ms = benchmark_inference(policy, args.device)
            logger.info(f"Inference: {avg_ms:.2f} +/- {std_ms:.2f} ms")
            if run:
                run.log({"eval/inference_ms": avg_ms, "eval/inference_std_ms": std_ms, "train/step": step})
            policy.train()

    # Final inference benchmark
    avg_ms, std_ms = benchmark_inference(policy, args.device)
    logger.info(f"Final inference: {avg_ms:.2f} +/- {std_ms:.2f} ms")

    # Summary
    final_loss = sum(losses[-100:]) / min(len(losses), 100)
    logger.info(f"Final loss (last 100 steps): {final_loss:.6f}")
    logger.info(f"Policy: {args.policy} | Steps: {args.steps} | Loss: {final_loss:.6f} | Latency: {avg_ms:.2f}ms")

    # Save results to disk
    results_dir = Path(args.save_results)
    results_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "policy": args.policy,
        "dataset": args.dataset_repo_id,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_inference_steps": args.num_inference_steps,
        "use_optimal_transport": args.use_optimal_transport,
        "seed": args.seed,
        "device": args.device,
        "n_params": n_params,
        "final_loss": final_loss,
        "inference_ms": avg_ms,
        "inference_std_ms": std_ms,
        "losses": losses,
    }
    results_path = results_dir / f"{args.policy}_{args.steps}steps.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved to {results_path}")

    if run:
        run.log({"final/loss": final_loss, "final/inference_ms": avg_ms, "final/n_params": n_params})
        run.finish()


if __name__ == "__main__":
    main()
