# Auto-register LOBE custom policies + apply lerobot patches when this package is imported.
# This makes `--policy.type=flow_matching` work in lerobot-train and lerobot-eval,
# and applies our data loading optimizations to lerobot.
from lobe.patches import apply_patches  # noqa: E402

apply_patches()

# X-VLA is implemented in lerobot — just import to ensure registration
from lerobot.policies.xvla.configuration_xvla import XVLAConfig  # noqa: F401, E402
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy  # noqa: F401, E402

# Register LOBE custom X-VLA processors (LIBERO dataset → absolute EE6D adapter)
import lobe.policies.xvla.libero_xvla_adapter  # noqa: F401, E402

import lobe.policies.flow_matching.configuration_flow_matching  # noqa: F401, E402
import lobe.policies.flow_matching.modeling_flow_matching  # noqa: F401, E402
