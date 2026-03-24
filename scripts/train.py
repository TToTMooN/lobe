"""Generic policy training — works with any registered environment.

Usage:
    # PushT (default)
    uv run python scripts/train.py
    uv run python scripts/train.py --env.name pusht --wandb.enable

    # PushT with GPU preload
    uv run python scripts/train.py --performance.gpu-preload

    # YAM bimanual (future)
    uv run python scripts/train.py --env.name yam --env.dataset-repo-id yourname/yam-red-cube

    # Diffusion policy
    uv run python scripts/train.py --policy.type diffusion

    # All config options
    uv run python scripts/train.py --help
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import tyro
from loguru import logger
from torch.utils.data import DataLoader

import lobe.video_compat  # noqa: F401
from lobe.configs import TrainPipelineConfig
from lobe.envs import get_env
from lobe.policies.factory import create_policy


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


def train_one(policy_type: str, cfg: TrainPipelineConfig, dataset, features, env_module):
    logger.info(f"{'=' * 60}")
    logger.info(f"Training: {policy_type} | env: {cfg.env.name} | steps: {cfg.train.steps}")
    logger.info(f"bf16={cfg.performance.bf16} | compile={cfg.performance.compile} | batch={cfg.train.batch_size}")
    logger.info(f"{'=' * 60}")

    torch.manual_seed(cfg.train.seed)
    fm_down_dims = tuple(int(x) for x in cfg.policy.down_dims.split(","))
    policy = create_policy(
        policy_type,
        features,
        dataset.meta.stats,
        n_obs_steps=cfg.env.n_obs_steps,
        horizon=cfg.env.horizon,
        n_action_steps=cfg.env.n_action_steps,
        num_inference_steps=cfg.policy.num_inference_steps,
        compile_model=cfg.performance.compile,
        compile_mode=cfg.performance.compile_mode,
        fm_down_dims=fm_down_dims,
        fm_embed_dim=cfg.policy.embed_dim,
    )
    policy.to(cfg.device)
    policy.train()

    n_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # EMA
    ema = None
    if cfg.train.ema_power > 0:
        from diffusers.training_utils import EMAModel

        ema = EMAModel(parameters=policy.parameters(), power=cfg.train.ema_power)
        ema.to(cfg.device)
        logger.info(f"EMA enabled (power={cfg.train.ema_power})")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.performance.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=cfg.performance.prefetch_factor,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=cfg.train.lr, weight_decay=1e-6)

    scheduler = None
    if cfg.train.use_cosine_schedule:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=cfg.train.warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=cfg.train.steps - cfg.train.warmup_steps, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[cfg.train.warmup_steps])

    use_amp = cfg.performance.bf16 and cfg.device == "cuda"
    amp_dtype = torch.bfloat16 if cfg.performance.bf16 else torch.float32

    # wandb
    run = None
    if cfg.wandb.enable:
        import wandb

        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.env.name}_{policy_type}",
            config={
                "policy_type": policy_type,
                "env": vars(cfg.env),
                "train": vars(cfg.train),
                "policy": vars(cfg.policy),
                "performance": vars(cfg.performance),
            },
        )

    # Check if env supports evaluation
    has_eval = hasattr(env_module, "evaluate")

    data_iter = iter(dataloader)
    losses = []
    output_dir = Path(cfg.logging.output_dir)
    t0 = time.perf_counter()
    accum = cfg.performance.gradient_accumulation

    for step in range(1, cfg.train.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(cfg.device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            loss, _ = policy.forward(batch)
            loss = loss / accum

        loss.backward()

        if step % accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.step(policy.parameters())

        losses.append(loss.item() * accum)

        if step % cfg.logging.log_every == 0:
            avg_loss = sum(losses[-cfg.logging.log_every :]) / min(len(losses), cfg.logging.log_every)
            elapsed = time.perf_counter() - t0
            sps = step / elapsed
            logger.info(
                f"[{policy_type}] Step {step}/{cfg.train.steps} | loss: {avg_loss:.6f} "
                f"| {sps:.1f} steps/s ({sps * cfg.train.batch_size:.0f} samples/s) "
                f"| ETA: {(cfg.train.steps - step) / sps:.0f}s"
            )
            if run:
                run.log({"loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]}, step=step)

        if step % cfg.logging.save_every == 0:
            ckpt_dir = save_checkpoint(policy, output_dir, policy_type, step, ema)
            logger.info(f"Checkpoint: {ckpt_dir}")

        if has_eval and cfg.logging.eval_every > 0 and step % cfg.logging.eval_every == 0:
            if ema is not None:
                ema.store(policy.parameters())
                ema.copy_to(policy.parameters())
            success_rate, avg_reward = env_module.evaluate(policy, cfg.device, cfg.logging.eval_rollouts)
            if ema is not None:
                ema.restore(policy.parameters())
            policy.train()
            logger.info(f"[{policy_type}] Eval {step}: success={success_rate * 100:.0f}%, reward={avg_reward:.3f}")
            if run:
                run.log({"eval/success_rate": success_rate, "eval/avg_reward": avg_reward}, step=step)

    # Final save
    ckpt_dir = save_checkpoint(policy, output_dir, policy_type, cfg.train.steps, ema)
    logger.info(f"Final checkpoint: {ckpt_dir}")

    meta = {
        "policy": policy_type,
        "env": cfg.env.name,
        "steps": cfg.train.steps,
        "final_loss": sum(losses[-100:]) / min(len(losses), 100),
        "n_params": n_params,
        "elapsed_s": time.perf_counter() - t0,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Done: {policy_type} | loss: {meta['final_loss']:.6f} | time: {meta['elapsed_s']:.1f}s")

    if run:
        run.finish()
    return ckpt_dir


def main():
    cfg = tyro.cli(TrainPipelineConfig)
    device = cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    cfg.device = device

    if cfg.performance.tf32 and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    logger.info(f"Device: {device} ({gpu_name})")

    # Load env module
    env_module = get_env(cfg.env.name)
    logger.info(f"Environment: {cfg.env.name}")

    # Load dataset
    logger.info(f"Loading dataset: {cfg.env.dataset_repo_id}")
    dataset, features = env_module.load_dataset(cfg.env.dataset_repo_id)
    logger.info(f"Dataset: {len(dataset)} frames")

    if cfg.performance.gpu_preload:
        from lobe.data.preload import preload_dataset_to_gpu

        dataset = preload_dataset_to_gpu(dataset, device)

    # Train
    policies = [cfg.policy.type]
    if cfg.policy.type == "both":
        policies = ["flow_matching", "diffusion"]

    for policy_type in policies:
        train_one(policy_type, cfg, dataset, features, env_module)


if __name__ == "__main__":
    main()
