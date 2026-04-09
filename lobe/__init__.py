# Auto-register LOBE custom policies with lerobot when this package is imported.
# This makes `--policy.type=flow_matching` work in lerobot-train and lerobot-eval.
import lobe.policies.flow_matching.configuration_flow_matching  # noqa: F401
import lobe.policies.flow_matching.modeling_flow_matching  # noqa: F401
