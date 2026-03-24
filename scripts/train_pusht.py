"""Train Flow Matching / Diffusion on PushT and save checkpoints for eval.

Usage:
    uv run python scripts/train_pusht.py --policy both --steps 50000
    uv run python scripts/train_pusht.py --policy flow_matching --steps 50000 --wandb
    uv run python scripts/train_pusht.py --policy flow_matching --eval-every 10000
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from loguru import logger
from torch.utils.data import DataLoader

import lobe.video_compat  # noqa: F401
from lobe import pusht


@dataclass
class Args:
    policy: str = "both"  # flow_matching | diffusion | both
    steps: int = 50000
    batch_size: int = 256
    lr: float = 1e-4
    num_inference_steps: int = 10
    horizon: int = pusht.HORIZON
    n_action_steps: int = pusht.N_ACTION_STEPS
    log_every: int = 100
    save_every: int = 10000
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "checkpoints/pusht"
    dataset_repo_id: str = pusht.DEFAULT_DATASET
    wandb: bool = False
    wandb_project: str = "lobe-train"
    # Performance
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    bf16: bool = True
    num_workers: int = 16
    prefetch_factor: int = 4
    tf32: bool = True
    gradient_accumulation: int = 1
    warmup_steps: int = 500
    use_cosine_schedule: bool = True
    # FM architecture
    fm_down_dims: str = "256,512,1024"
    fm_embed_dim: int = 256
    ema_power: float = 0.75  # 0 = disable
    # Eval during training
    eval_every: int = 0  # 0 = disable
    eval_rollouts: int = 10


def save_checkpoint(policy, output_dir: Path, policy_type: str, step: int, ema=None):
    ckpt_dir = output_dir / f"{policy_type}_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if ema is not None:
        ema.store(policy.parameters())
        ema.copy_to(policy.parameters())
    state_dict = policy.state_dict()
    clean_sd = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    torch.save(clean_sd, ckpt_dir / "model.pt")
    if ema is not None:
        ema.restore(policy.parameters())
    return ckpt_dir


def train_one(policy_type: str, args: Args, dataset, features, run=None):
    logger.info(f"{'=' * 60}")
    logger.info(f"Training: {policy_type} for {args.steps} steps")
    logger.info(f"bf16={args.bf16} | compile={args.compile} | tf32={args.tf32} | batch_size={args.batch_size}")
    logger.info(f"{'=' * 60}")

    torch.manual_seed(args.seed)
    fm_down_dims = tuple(int(x) for x in args.fm_down_dims.split(","))
    policy = pusht.create_policy(
        policy_type,
        features,
        dataset.meta.stats,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        num_inference_steps=args.num_inference_steps,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        fm_down_dims=fm_down_dims,
        fm_embed_dim=args.fm_embed_dim,
    )
    policy.to(args.device)
    policy.train()

    n_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # EMA
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

    scheduler = None
    if args.use_cosine_schedule:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=args.steps - args.warmup_steps, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_steps])

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
            sps = step / elapsed
            logger.info(
                f"[{policy_type}] Step {step}/{args.steps} | loss: {avg_loss:.6f} "
                f"| {sps:.1f} steps/s ({sps * args.batch_size:.0f} samples/s) | ETA: {(args.steps - step) / sps:.0f}s"
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
            success_rate, avg_reward = pusht.evaluate(policy, args.device, args.eval_rollouts)
            if ema is not None:
                ema.restore(policy.parameters())
            policy.train()
            logger.info(f"[{policy_type}] Eval {step}: success={success_rate * 100:.0f}%, reward={avg_reward:.3f}")
            if run:
                run.log({f"{policy_type}/eval_success": success_rate, f"{policy_type}/eval_reward": avg_reward})

    # Final save
    ckpt_dir = save_checkpoint(policy, output_dir, policy_type, args.steps, ema)
    logger.info(f"Final checkpoint: {ckpt_dir}")

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

    if args.tf32 and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    logger.info(f"Device: {device} ({gpu_name})")

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, config=vars(args))

    logger.info("Loading dataset...")
    dataset, features = pusht.load_dataset(args.dataset_repo_id)
    logger.info(f"Dataset: {len(dataset)} frames")

    policies = ["flow_matching", "diffusion"] if args.policy == "both" else [args.policy]
    checkpoints = {}
    for policy_type in policies:
        ckpt_dir = train_one(policy_type, args, dataset, features, run)
        checkpoints[policy_type] = str(ckpt_dir)

    logger.info(f"\n{'=' * 60}")
    for name, path in checkpoints.items():
        logger.info(f"  {name}: {path}")
    logger.info(f"{'=' * 60}")

    if run:
        run.finish()


if __name__ == "__main__":
    main()
