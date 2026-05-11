"""Train a YAM policy by preset name.

Picks a preset from lobe/configs/yam.py, assembles the CLI flags, and
launches via accelerate. Saves you from copy-pasting 20-flag commands.

Usage:
    uv run python scripts/train_yam.py diffusion
    uv run python scripts/train_yam.py flow_matching
    uv run python scripts/train_yam.py xvla
    uv run python scripts/train_yam.py smolvla
    uv run python scripts/train_yam.py --list            # show available presets
    uv run python scripts/train_yam.py xvla --gpus 4     # override GPU count
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from loguru import logger

PRESET_ALIASES = {
    "diffusion": "yam_grey_cube_diffusion",
    "dp": "yam_grey_cube_diffusion",
    "flow_matching": "yam_grey_cube_flow_matching",
    "fm": "yam_grey_cube_flow_matching",
    "xvla": "yam_grey_cube_xvla",
    "smolvla": "yam_grey_cube_smolvla",
    "dp-vial": "yam_place_vial_diffusion",
    "fm-vial": "yam_place_vial_flow_matching",
    "xvla-vial": "yam_place_vial_xvla",
    "xvla-vial-30fps": "yam_8ml_vial_xvla",
    "fm-vial-30fps-h32": "yam_8ml_vial_flow_matching_h32",
    "fm-v2": "yam_8ml_vial_flow_matching_h32",
}

_XVLA_RENAME = (
    '{"observation.images.head_camera": "observation.images.image",'
    ' "observation.images.left_wrist_camera": "observation.images.image2",'
    ' "observation.images.right_wrist_camera": "observation.images.image3"}'
)
_SMOLVLA_RENAME = (
    '{"observation.images.head_camera": "observation.images.camera1",'
    ' "observation.images.left_wrist_camera": "observation.images.camera2",'
    ' "observation.images.right_wrist_camera": "observation.images.camera3"}'
)
RENAME_MAPS = {
    "yam_grey_cube_xvla": _XVLA_RENAME,
    "yam_grey_cube_smolvla": _SMOLVLA_RENAME,
    "yam_place_vial_xvla": _XVLA_RENAME,
    "yam_8ml_vial_xvla": _XVLA_RENAME,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("preset", nargs="?", help="Preset name or alias (diffusion, fm, xvla, smolvla)")
    parser.add_argument("--list", action="store_true", help="List available presets")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs (default: auto from nvidia-smi)")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running")
    args = parser.parse_args()

    from lobe.configs.yam import PRESETS

    if args.list or args.preset is None:
        logger.info("Available presets:")
        for name, cfg in PRESETS.items():
            aliases = [a for a, p in PRESET_ALIASES.items() if p == name]
            logger.info(f"  {name} (aliases: {', '.join(aliases)})")
            logger.info(f"    steps={cfg.steps} batch={cfg.batch_size} output={cfg.output_dir}")
        return 0

    preset_name = PRESET_ALIASES.get(args.preset, args.preset)
    if preset_name not in PRESETS:
        logger.error(f"Unknown preset '{args.preset}'. Use --list to see options.")
        return 1

    config = PRESETS[preset_name]
    launch_args = config.to_launch_args()

    if preset_name in RENAME_MAPS:
        launch_args.append(f"--rename_map={RENAME_MAPS[preset_name]}")

    n_gpus = args.gpus
    if n_gpus is None:
        import torch
        n_gpus = torch.cuda.device_count()

    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        f"--num_processes={n_gpus}",
        "--mixed_precision=bf16",
        "scripts/_lobe_train_entry.py",
        *launch_args,
    ]

    logger.info(f"Preset: {preset_name}")
    logger.info(f"GPUs: {n_gpus}")
    logger.info(f"Command:\n  {' '.join(cmd)}")

    if args.dry_run:
        return 0

    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main() or 0)
