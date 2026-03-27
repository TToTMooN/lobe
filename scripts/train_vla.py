"""VLA fine-tuning via LeRobot — wraps lerobot-train with sensible defaults.

Supports pi0, SmolVLA, and any LeRobot-registered VLA policy.
For FM/Diffusion baselines, use scripts/train.py instead.

Usage:
    # SmolVLA on YAM data (recommended first experiment — smallest VLA)
    uv run python scripts/train_vla.py --model smolvla --dataset yourname/yam-red-cube

    # pi0 on YAM data
    uv run python scripts/train_vla.py --model pi0 --dataset yourname/yam-red-cube

    # SmolVLA on PushT (for testing the pipeline)
    uv run python scripts/train_vla.py --model smolvla --dataset lerobot/pusht

    # Custom settings
    uv run python scripts/train_vla.py --model pi0 --dataset yourname/data \\
        --steps 50000 --batch-size 4 --output-dir checkpoints/pi0-yam
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

import tyro
from loguru import logger

# Model presets — pretrained weights + recommended training settings
MODEL_PRESETS = {
    "smolvla": {
        "policy_path": "lerobot/smolvla_base",
        "batch_size": 8,
        "steps": 50000,
        "description": "SmolVLA — lightweight VLA, flow matching, fast iteration",
    },
    "pi0": {
        "policy_path": "lerobot/pi0",
        "batch_size": 4,
        "steps": 50000,
        "description": "pi0 — 3B VLA, flow matching action expert, strong baseline",
    },
}


@dataclass
class VLATrainArgs:
    """VLA fine-tuning configuration."""

    model: str = "smolvla"  # smolvla | pi0
    dataset: str = ""  # HuggingFace dataset repo_id (required)
    steps: int = 0  # 0 = use model preset
    batch_size: int = 0  # 0 = use model preset
    output_dir: str = ""  # auto-generated if empty
    # Advanced
    policy_path: str = ""  # override pretrained weights path
    lr: float = 0.0  # 0 = use lerobot default
    eval_freq: int = 5000
    save_freq: int = 5000
    num_workers: int = 4
    wandb: bool = True
    wandb_project: str = "lobe-train"
    resume: bool = False
    extra_args: str = ""  # additional args passed to lerobot-train


def main():
    args = tyro.cli(VLATrainArgs)

    if not args.dataset:
        logger.error("Must provide --dataset (e.g. yourname/yam-red-cube or lerobot/pusht)")
        sys.exit(1)

    # Get model preset
    if args.model not in MODEL_PRESETS:
        logger.error(f"Unknown model: {args.model}. Available: {list(MODEL_PRESETS.keys())}")
        sys.exit(1)

    preset = MODEL_PRESETS[args.model]
    logger.info(f"Model: {args.model} — {preset['description']}")

    policy_path = args.policy_path or preset["policy_path"]
    batch_size = args.batch_size or preset["batch_size"]
    steps = args.steps or preset["steps"]
    output_dir = args.output_dir or f"checkpoints/{args.model}-{args.dataset.split('/')[-1]}-{steps // 1000}k"

    # Build lerobot-train command
    # Use -c wrapper to apply video_compat patch (PyAV) before lerobot imports.
    # This patches torchvision VideoReader (removed in nightly) with PyAV.
    cmd = [
        sys.executable,
        "-c",
        "import sys; sys.argv[0] = 'lerobot-train'; import lobe.video_compat; "
        "from lerobot.scripts.lerobot_train import main; main()",
        f"--dataset.repo_id={args.dataset}",
        f"--policy.path={policy_path}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--output_dir={output_dir}",
        f"--eval_freq={args.eval_freq}",
        f"--save_freq={args.save_freq}",
        f"--num_workers={args.num_workers}",
        f"--policy.repo_id={args.dataset.split('/')[-1]}-{args.model}",
        "--save_checkpoint=true",
    ]

    # SmolVLA expects camera1/camera2/camera3 — remap dataset image keys
    if args.model == "smolvla":
        if "libero" in args.dataset.lower():
            # LIBERO: 2 cameras (image + image2) → camera1 + camera2, 1 empty
            cmd.append(
                '--rename_map={"observation.images.image": "observation.images.camera1",'
                ' "observation.images.image2": "observation.images.camera2"}'
            )
            cmd.append("--policy.empty_cameras=1")
        elif "pusht" in args.dataset.lower():
            cmd.append('--rename_map={"observation.image": "observation.images.camera1"}')
            cmd.append("--policy.empty_cameras=2")
        else:
            # Generic single-camera fallback
            cmd.append("--policy.empty_cameras=2")

    if args.lr > 0:
        cmd.append(f"--optimizer.lr={args.lr}")

    if args.wandb:
        cmd.extend(
            [
                "--wandb.enable=True",
                f"--wandb.project={args.wandb_project}",
            ]
        )

    if args.resume:
        cmd.append("--resume=True")

    if args.extra_args:
        cmd.extend(args.extra_args.split())

    logger.info(f"Output: {output_dir}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("=" * 60)

    # Run lerobot-train
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
