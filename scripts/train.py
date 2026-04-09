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


def train_one(cfg: TrainPipelineConfig, dataset, features, env_module):
    from lobe.configs import FMPolicyConfig

    is_fm = isinstance(cfg.policy, FMPolicyConfig)
    policy_type = "flow_matching" if is_fm else "diffusion"

    logger.info(f"{'=' * 60}")
    logger.info(f"Training: {policy_type} | env: {cfg.env.name} | steps: {cfg.train.steps}")
    logger.info(f"bf16={cfg.performance.bf16} | compile={cfg.performance.compile} | batch={cfg.train.batch_size}")
    logger.info(f"{'=' * 60}")

    torch.manual_seed(cfg.train.seed)

    # Build policy from config variant
    fm_kwargs = {}
    resize_shape = None
    if is_fm:
        fm_kwargs = {
            "fm_backbone": cfg.policy.backbone,
            "fm_vision_encoder": cfg.policy.vision_encoder,
            "fm_down_dims": tuple(int(x) for x in cfg.policy.down_dims.split(",")),
            "fm_embed_dim": cfg.policy.embed_dim,
            "fm_delta_actions": cfg.policy.delta_actions,
        }
        resize_shape = tuple(int(x) for x in cfg.policy.resize_shape.split(",")) if cfg.policy.resize_shape else None

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
        resize_shape=resize_shape,
        **fm_kwargs,
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

    # wandb — init BEFORE dataloader to avoid fork conflicts with persistent workers
    run = None
    if cfg.wandb.enable:
        import wandb

        run_name = cfg.wandb.name or f"{cfg.env.name}_{policy_type}_{cfg.train.steps // 1000}k"
        tags = [t.strip() for t in cfg.wandb.tags.split(",") if t.strip()] if cfg.wandb.tags else []
        tags += [cfg.env.name, policy_type]
        run = wandb.init(
            project=cfg.wandb.project,
            name=run_name,
            group=cfg.wandb.group or cfg.env.name,
            tags=tags,
            config={
                "policy_type": policy_type,
                "n_params": n_params,
                "env": vars(cfg.env),
                "train": vars(cfg.train),
                "policy": vars(cfg.policy),
                "performance": vars(cfg.performance),
            },
        )

    nw = cfg.performance.num_workers
    gpu_resident = isinstance(dataset, __import__("lobe.data.fast_dataset", fromlist=["FastDataset"]).FastDataset)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0 if gpu_resident else nw,
        pin_memory=not gpu_resident,
        persistent_workers=(not gpu_resident) and nw > 0,
        prefetch_factor=cfg.performance.prefetch_factor if (not gpu_resident) and nw > 0 else None,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=cfg.train.lr, weight_decay=1e-6)

    scheduler = None
    if cfg.train.use_cosine_schedule:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=cfg.train.warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=cfg.train.steps - cfg.train.warmup_steps, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[cfg.train.warmup_steps])

    # Multi-GPU: wrap with accelerate if available
    accelerator = None
    try:
        from accelerate import Accelerator

        accelerator = Accelerator(mixed_precision="bf16" if cfg.performance.bf16 else "no")
        policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
        if scheduler is not None:
            scheduler = accelerator.prepare(scheduler)
        cfg.device = str(accelerator.device)
        if accelerator.is_main_process:
            logger.info(f"Accelerate: {accelerator.num_processes} GPUs, device={accelerator.device}")
    except ImportError:
        pass

    use_amp = cfg.performance.bf16 and cfg.device == "cuda" and accelerator is None
    amp_dtype = torch.bfloat16 if cfg.performance.bf16 else torch.float32

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

        if gpu_resident:
            batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        elif accelerator is None:
            batch = {k: v.to(cfg.device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        else:
            batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp) if accelerator is None else accelerator.autocast():
            loss, _ = policy.forward(batch)
            loss = loss / accum

        if accelerator is not None:
            accelerator.backward(loss)
        else:
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

    elapsed = time.perf_counter() - t0
    final_loss = sum(losses[-100:]) / min(len(losses), 100)
    meta = {
        "policy": policy_type,
        "env": cfg.env.name,
        "steps": cfg.train.steps,
        "final_loss": final_loss,
        "n_params": n_params,
        "elapsed_s": elapsed,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Done: {policy_type} | loss: {final_loss:.6f} | time: {elapsed:.1f}s")

    # Log to experiments.tsv
    from lobe.experiment_log import log_experiment

    gpu_name = torch.cuda.get_device_name(0) if cfg.device == "cuda" else "CPU"
    backbone = cfg.policy.backbone if is_fm else ""
    log_experiment(
        env=cfg.env.name,
        policy=policy_type,
        backbone=backbone,
        norm="MEAN_STD" if is_fm else "MIN_MAX",
        steps=cfg.train.steps,
        batch_size=cfg.train.batch_size,
        n_params=n_params,
        final_loss=final_loss,
        train_s=elapsed,
        gpu=gpu_name,
    )

    if run:
        run.finish()
    return ckpt_dir


def main():
    from lobe.configs import PRESETS

    cfg = tyro.extras.overridable_config_cli(PRESETS)
    device = cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    cfg.device = device

    if cfg.performance.tf32 and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    logger.info(f"Device: {device} ({gpu_name})")

    # Load env module and apply env-specific defaults (only where cfg value is 0 = "use default")
    env_module = get_env(cfg.env.name)
    if hasattr(env_module, "FPS"):
        cfg.env.fps = env_module.FPS
    if cfg.env.n_obs_steps == 0 and hasattr(env_module, "N_OBS_STEPS"):
        cfg.env.n_obs_steps = env_module.N_OBS_STEPS
    if cfg.env.horizon == 0 and hasattr(env_module, "HORIZON"):
        cfg.env.horizon = env_module.HORIZON
    if cfg.env.n_action_steps == 0 and hasattr(env_module, "N_ACTION_STEPS"):
        cfg.env.n_action_steps = env_module.N_ACTION_STEPS
    if cfg.env.max_steps == 0 and hasattr(env_module, "MAX_STEPS"):
        cfg.env.max_steps = env_module.MAX_STEPS
    logger.info(f"Environment: {cfg.env.name} (horizon={cfg.env.horizon}, action_steps={cfg.env.n_action_steps})")

    # Load dataset — use FastDataset if .pt cache exists, else standard LeRobot
    repo_id = cfg.env.dataset_repo_id
    if repo_id.endswith(".pt") and Path(repo_id).exists():
        from lobe.data.fast_dataset import FastDataset

        # Build delta_timestamps from cfg values (respecting CLI overrides for horizon/obs_steps)
        if hasattr(env_module, "delta_timestamps"):
            dt = env_module.delta_timestamps()
            # Override action timestamps with cfg horizon
            fps = cfg.env.fps
            act_ts = [i / fps for i in range(1 - cfg.env.n_obs_steps, 1 - cfg.env.n_obs_steps + cfg.env.horizon)]
            dt["action"] = act_ts
            # Override obs timestamps with cfg n_obs_steps
            obs_ts = [i / fps for i in range(1 - cfg.env.n_obs_steps, 1)]
            for k in list(dt.keys()):
                if k != "action":
                    dt[k] = obs_ts
        else:
            dt = None
        dataset = FastDataset(repo_id, device=device, delta_timestamps=dt)
        # Build features from the cache metadata
        from lerobot.configs.types import FeatureType, PolicyFeature

        features = {}
        # Use raw per-frame shapes (without temporal window dim) — policy handles stacking
        meta = dataset.meta_info
        for key, shape in meta.get("features", {}).items():
            raw_shape = tuple(shape[1:])  # strip N dimension
            if "image" in key:
                features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=raw_shape)
            elif key == "action":
                features[key] = PolicyFeature(type=FeatureType.ACTION, shape=raw_shape)
            elif "state" in key:
                features[key] = PolicyFeature(type=FeatureType.STATE, shape=raw_shape)
        logger.info(f"Fast dataset: {len(dataset)} frames from {repo_id}")
    else:
        logger.info(f"Loading dataset: {repo_id}")
        dataset, features = env_module.load_dataset(repo_id)
        logger.info(f"Dataset: {len(dataset)} frames")

    # Train
    train_one(cfg, dataset, features, env_module)


if __name__ == "__main__":
    main()
