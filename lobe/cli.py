"""CLI entry points that register LOBE policies before calling lerobot.

Usage:
    lobe-train --policy.type=flow_matching ...   (instead of lerobot-train)
    lobe-eval --policy.path=<checkpoint> ...     (instead of lerobot-eval)

These are thin wrappers that import lobe (registering custom policies)
then delegate to lerobot's CLI.
"""

import lobe  # noqa: F401 — registers custom policies
import lobe.video_compat  # noqa: F401 — patches PyAV


def train():
    from lerobot.scripts.lerobot_train import main

    main()


def eval():
    from lerobot.scripts.lerobot_eval import main

    main()


def serve():
    from lobe.serve import main

    main()
