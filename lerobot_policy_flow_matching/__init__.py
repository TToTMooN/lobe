"""LeRobot plugin: registers Flow Matching policy type.

This package is auto-discovered by lerobot via the `lerobot_policy_*` naming convention.
It registers FlowMatchingConfig and FlowMatchingPolicy so that:
  lerobot-train --policy.type=flow_matching ...
  lerobot-eval --policy.path=<fm-checkpoint> ...
both work without custom entry points.
"""
from lobe.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig  # noqa: F401
from lobe.policies.flow_matching.modeling_flow_matching import FlowMatchingPolicy  # noqa: F401
