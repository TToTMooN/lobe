"""Train Flow Matching / Diffusion on PushT and save checkpoints for eval.

Uses lerobot/pusht_image (pre-decoded images) for fast data loading.
Supports bf16 mixed precision, torch.compile, TF32, and large batch sizes.

Usage:
    # Train both (~8 min for 5000 steps on H100 with defaults)
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

import numpy as np
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
    batch_size: int = 256
    lr: float = 1e-4
    num_inference_steps: int = 10
    horizon: int = 16
    n_action_steps: int = 8
    log_every: int = 100
    save_every: int = 1000
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "checkpoints/pusht"
    dataset_repo_id: str = "lerobot/pusht_image"
    wandb: bool = False
    wandb_project: str = "lobe-train"
    # Performance
    compile: bool = False  # torch.compile the model (big speedup after warmup)
    compile_mode: str = "reduce-overhead"  # max-autotune | reduce-overhead | default
    bf16: bool = True  # bfloat16 mixed precision (H100/5090 tensor cores)
    num_workers: int = 16  # dataloader workers
    prefetch_factor: int = 4  # batches to prefetch per worker
    tf32: bool = True  # TF32 for float32 matmuls (free speedup on Ampere+)
    gradient_accumulation: int = 1  # accumulate gradients over N steps
    warmup_steps: int = 500  # LR warmup steps
    use_cosine_schedule: bool = True  # cosine LR decay after warmup
    # FM architecture (match HRI-EU defaults for PushT)
    fm_down_dims: str = "256,512,1024"  # U-Net channel dims
    fm_embed_dim: int = 256  # timestep embedding dim
    ema_power: float = 0.75  # EMA decay power (0 = disable EMA)
    # Eval during training
    eval_every: int = 0  # run eval every N steps (0 = disable)
    eval_rollouts: int = 10  # number of rollouts per eval


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
    # Image datasets don't need a video backend; video datasets use torchcodec (2-3x faster than pyav)
    is_video = "image" not in repo_id
    kwargs = {"video_backend": "torchcodec"} if is_video else {}
    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, **kwargs)
    features = dataset_to_policy_features(dataset.meta.features)
    return dataset, features


def make_policy(policy_type, features, stats, args):
    input_features = {k: v for k, v in features.items() if v.type != FeatureType.ACTION}
    output_features = {k: v for k, v in features.items() if v.type == FeatureType.ACTION}

    if policy_type == "flow_matching":
        down_dims = tuple(int(x) for x in args.fm_down_dims.split(","))
        config = FlowMatchingConfig(
            n_obs_steps=2,
            horizon=args.horizon,
            n_action_steps=args.n_action_steps,
            num_inference_steps=args.num_inference_steps,
            compile_model=args.compile,
            compile_mode=args.compile_mode,
            down_dims=down_dims,
            diffusion_step_embed_dim=args.fm_embed_dim,
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


def save_checkpoint(policy, output_dir: Path, policy_type: str, step: int, ema=None):
    ckpt_dir = output_dir / f"{policy_type}_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if ema is not None:
        # Save EMA weights (copy EMA -> policy, save, restore original)
        ema.store(policy.parameters())
        ema.copy_to(policy.parameters())
    state_dict = policy.state_dict()
    clean_sd = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    torch.save(clean_sd, ckpt_dir / "model.pt")
    if ema is not None:
        ema.restore(policy.parameters())
    return ckpt_dir


def evaluate_policy(policy, device, n_rollouts=10, max_steps=300, seed=0):
    """Run PushT rollouts and return success rate + avg reward."""
    import gym_pusht  # noqa: F401
    import gymnasium
    from lerobot.envs.utils import preprocess_observation

    policy.eval()
    successes, rewards_all = [], []
    for i in range(n_rollouts):
        env = gymnasium.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels_agent_pos")
        obs, _ = env.reset(seed=seed + i)
        policy.reset()
        rewards = []
        for _ in range(max_steps):
            batch = {k: v.to(device) for k, v in preprocess_observation(obs).items()}
            with torch.no_grad():
                action = policy.select_action(batch)
            obs, reward, term, trunc, info = env.step(action[0].cpu().numpy().clip(0, 512))
            rewards.append(reward)
            if term or trunc:
                break
        successes.append(info.get("is_success", False))
        rewards_all.append(np.mean(rewards))
        env.close()
    policy.train()
    return float(np.mean(successes)), float(np.mean(rewards_all))


def train_one(policy_type: str, args: Args, dataset, features, run=None):
    logger.info(f"{'=' * 60}")
    logger.info(f"Training: {policy_type} for {args.steps} steps")
    logger.info(f"bf16={args.bf16} | compile={args.compile} | tf32={args.tf32} | batch_size={args.batch_size}")
    logger.info(f"workers={args.num_workers} | prefetch={args.prefetch_factor} | accum={args.gradient_accumulation}")
    logger.info(f"{'=' * 60}")

    torch.manual_seed(args.seed)
    policy = make_policy(policy_type, features, dataset.meta.stats, args)
    policy.to(args.device)
    policy.train()

    n_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # EMA for smoother inference weights (used by all working FM implementations)
    ema = None
    if args.ema_power > 0:
        from diffusers.training_utils import EMAModel

        ema = EMAModel(parameters=policy.parameters(), power=args.ema_power)
        ema.to(args.device)
        logger.info(f"EMA enabled (power={args.ema_power})")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=args.lr, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    scheduler = None
    if args.use_cosine_schedule:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=args.steps - args.warmup_steps, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_steps])

    # Mixed precision scaler for bf16/fp16
    use_amp = args.bf16 and args.device == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float32

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

        batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            loss, _ = policy.forward(batch)
            loss = loss / args.gradient_accumulation

        loss.backward()

        if step % args.gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.step(policy.parameters())

        losses.append(loss.item() * args.gradient_accumulation)

        if step % args.log_every == 0:
            avg_loss = sum(losses[-args.log_every :]) / min(len(losses), args.log_every)
            elapsed = time.perf_counter() - t0
            steps_per_sec = step / elapsed
            eta = (args.steps - step) / steps_per_sec
            samples_per_sec = steps_per_sec * args.batch_size
            logger.info(
                f"[{policy_type}] Step {step}/{args.steps} | loss: {avg_loss:.6f} "
                f"| {steps_per_sec:.1f} steps/s ({samples_per_sec:.0f} samples/s) | ETA: {eta:.0f}s"
            )
            if run:
                run.log({f"{policy_type}/loss": avg_loss, f"{policy_type}/step": step})

        if step % args.save_every == 0:
            ckpt_dir = save_checkpoint(policy, output_dir, policy_type, step, ema)
            logger.info(f"Checkpoint: {ckpt_dir}")

        if args.eval_every > 0 and step % args.eval_every == 0:
            if ema is not None:
                ema.store(policy.parameters())
                ema.copy_to(policy.parameters())
            success_rate, avg_reward = evaluate_policy(policy, args.device, args.eval_rollouts)
            if ema is not None:
                ema.restore(policy.parameters())
            logger.info(
                f"[{policy_type}] Eval step {step}: success={success_rate * 100:.0f}%, reward={avg_reward:.3f}"
            )
            if run:
                run.log(
                    {
                        f"{policy_type}/eval_success": success_rate,
                        f"{policy_type}/eval_reward": avg_reward,
                        f"{policy_type}/step": step,
                    }
                )

    # Final save
    ckpt_dir = save_checkpoint(policy, output_dir, policy_type, args.steps, ema)
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

    # Enable TF32 for free speedup on Ampere+ (H100, A100, RTX 5090)
    if args.tf32 and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # Enable cudnn benchmark for fixed-size inputs (picks fastest conv algorithm)
    torch.backends.cudnn.benchmark = True

    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    logger.info(f"Device: {device} ({gpu_name})")

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
