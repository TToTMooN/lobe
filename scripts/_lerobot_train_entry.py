"""Entry point for accelerate launch — registers LOBE policies before lerobot-train."""
import lobe  # noqa: F401 — registers custom policies
import lobe.video_compat  # noqa: F401 — patches PyAV
from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()
