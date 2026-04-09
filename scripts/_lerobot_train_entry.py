"""Thin entry point for lerobot-train with video_compat patch + custom policies.

Used by train_vla.py — supports both single-GPU and accelerate launch.
Registers LOBE custom policies (flow_matching) so lerobot-train can use them.
"""
import lobe.video_compat  # noqa: F401 — patches PyAV before lerobot imports
import lobe.policies.flow_matching.configuration_flow_matching  # noqa: F401 — registers flow_matching policy type
import lobe.policies.flow_matching.modeling_flow_matching  # noqa: F401
from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()
