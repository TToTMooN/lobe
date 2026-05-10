"""Thin entry point for accelerate launch — imports lobe then calls lerobot-train."""
import lobe  # noqa: F401 — registers custom policies and applies patches
import lobe.video_compat  # noqa: F401 — patches PyAV

from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()
