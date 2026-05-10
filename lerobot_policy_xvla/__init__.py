"""LeRobot plugin: registers X-VLA policy type.

This package is auto-discovered by lerobot via the `lerobot_policy_*` naming convention.
It registers XVLAConfig and XVLAPolicy so that:
  lerobot-train --policy.type=xvla ...
  lerobot-eval --policy.path=<xvla-checkpoint> ...
both work without custom entry points.
"""

from lerobot.policies.xvla.configuration_xvla import XVLAConfig  # noqa: F401
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy  # noqa: F401
