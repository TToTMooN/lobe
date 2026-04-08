"""Thin entry point for lerobot-train with video_compat patch.

Used by train_vla.py — supports both single-GPU and accelerate launch.
"""
import lobe.video_compat  # noqa: F401 — patches PyAV before lerobot imports
from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()
