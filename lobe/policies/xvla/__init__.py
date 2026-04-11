"""Local copy of X-VLA policy from lerobot (for reading/modification).

Currently LOBE uses the lerobot-bundled X-VLA (registered in lobe/__init__.py).
This local copy is here so you can read and modify the code.

To switch to using this local copy:
1. In lobe/__init__.py, replace:
     from lerobot.policies.xvla.configuration_xvla import XVLAConfig
     from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
   with:
     from lobe.policies.xvla.configuration_xvla import XVLAConfig
     from lobe.policies.xvla.modeling_xvla import XVLAPolicy
2. Uncomment @PreTrainedConfig.register_subclass("xvla") in configuration_xvla.py
"""
