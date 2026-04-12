"""Adapter from HuggingFaceVLA/libero raw format → X-VLA EE6D training format.

The HuggingFaceVLA/libero dataset stores LIBERO demos in its native format:
- observation.state (8-D): [eef_xyz(3), eef_axis_angle(3), gripper_qpos(2)]
- action (7-D): [delta_xyz(3), delta_axis_angle(3), gripper(±1)] — normalized OSC_POSE commands
- observation.images.* (3, 256, 256): float32 in [0, 1]

X-VLA was trained on a 10-D absolute EE6D target (padded to 20-D dual-arm):
- observation.state (20-D): [eef_xyz(3), eef_rot6d(6), extra(1), zeros(10)]
- action (20-D): [abs_xyz(3), abs_rot6d(6), gripper_binary(1), zeros(10)]
- Images: ImageNet-normalized

This module provides a single ProcessorStep that converts from the native format to the
X-VLA format. It's designed to be idempotent — safe to run on already-converted data —
so the same step can be used at training time (where data is raw) and during eval
(where data has already been converted by env_preprocessor).

At inference/eval, LiberoProcessorStep and XVLAImageNetNormalizeProcessorStep already
produce the X-VLA format from the env, so this step is a no-op there.

At training, we inject this step into the policy preprocessor pipeline so the dataset's
raw format gets converted before the model sees it. See lobe/patches.py for the injection.

Action conversion (per chunk step):
  delta_xyz_m = action[:3] * OSC_POS_SCALE     # 0.05 m per unit
  delta_aa    = action[3:6] * OSC_ROT_SCALE    # 0.5 rad per unit
  abs_xyz[t]  = eef_xyz_0 + cumsum(delta_xyz_m)[t]
  R_abs[t]    = R_delta_aa[t] ∘ R_delta_aa[t-1] ∘ ... ∘ R_eef_0
  abs_rot6d[t] = first 2 columns of R_abs[t]
  gripper[t]  = (action[6] > 0).float()  # {0, 1} for BCEWithLogitsLoss

Note: eef_xyz_0 and R_eef_0 are the state at chunk start (single observation).
The cumulative integration approximates absolute EE pose at each chunk step
assuming perfect delta tracking (idealized; real robot lags).
"""

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.pipeline import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

# OSC_POSE controller scales (from robosuite/controllers/config/osc_pose.json)
OSC_POS_SCALE = 0.05  # meters per unit [-1, 1]
OSC_ROT_SCALE = 0.5  # radians per unit [-1, 1]

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (..., 3) to rotation matrices (..., 3, 3) via Rodrigues formula."""
    theta = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / theta
    cos_t = theta.cos().unsqueeze(-1)
    sin_t = theta.sin().unsqueeze(-1)
    one_minus_cos = 1.0 - cos_t

    # Skew-symmetric matrix K for each axis
    zero = torch.zeros_like(axis[..., :1])
    K = torch.stack(
        [
            torch.cat([zero, -axis[..., 2:3], axis[..., 1:2]], dim=-1),
            torch.cat([axis[..., 2:3], zero, -axis[..., 0:1]], dim=-1),
            torch.cat([-axis[..., 1:2], axis[..., 0:1], zero], dim=-1),
        ],
        dim=-2,
    )  # (..., 3, 3)

    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    return eye + sin_t * K + one_minus_cos * (K @ K)


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices (..., 3, 3) to 6D representation (..., 6)
    by taking the first two columns flattened (col1, col2)."""
    # R[..., :, 0] = first column, R[..., :, 1] = second column
    col1 = R[..., :, 0]  # (..., 3)
    col2 = R[..., :, 1]  # (..., 3)
    return torch.cat([col1, col2], dim=-1)  # (..., 6)


@dataclass
@ProcessorStepRegistry.register(name="libero_xvla_adapter")
class LiberoXVLAAdapterStep(ProcessorStep):
    """Idempotent adapter from HuggingFaceVLA/libero raw format to X-VLA training format.

    Transforms (all no-op if already in target format):
        1. State 8-D → 20-D EE6D
        2. Action 7-D delta → 20-D absolute EE6D (requires state, uses cumulative integration)
        3. Images [0, 1] → ImageNet-normalized

    Usage at TRAINING: inject at position 0 of policy preprocessor to convert raw dataset batches.
    Usage at EVAL: safe to leave in — all transforms are idempotent and no-op on already-converted data.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = dict(transition)

        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return new_transition
        observation = dict(observation)

        # Capture raw state BEFORE any conversion — action conversion needs it.
        raw_state_8d = None
        state = observation.get(OBS_STATE)
        if state is not None and isinstance(state, torch.Tensor) and state.shape[-1] == 8:
            raw_state_8d = state

        # === 1. STATE: 8-D → 20-D EE6D ===
        if raw_state_8d is not None:
            # 8-D: [eef_xyz(3), eef_axis_angle(3), gripper_qpos_l, gripper_qpos_r]
            eef_xyz = raw_state_8d[..., :3]  # (..., 3)
            eef_aa = raw_state_8d[..., 3:6]  # (..., 3)
            R = axis_angle_to_matrix(eef_aa)  # (..., 3, 3)
            eef_rot6d = matrix_to_rot6d(R)  # (..., 6)
            extra = torch.zeros_like(eef_xyz[..., :1])  # (..., 1)
            left_arm_10d = torch.cat([eef_xyz, eef_rot6d, extra], dim=-1)  # (..., 10)
            right_arm_10d = torch.zeros_like(left_arm_10d)
            state_20d = torch.cat([left_arm_10d, right_arm_10d], dim=-1)  # (..., 20)
            observation[OBS_STATE] = state_20d

        # === 2. ACTION: 7-D delta → 20-D absolute EE6D ===
        action = new_transition.get(TransitionKey.ACTION)
        if (
            action is not None
            and isinstance(action, torch.Tensor)
            and action.shape[-1] == 7
            and raw_state_8d is not None
        ):
            # action shape: (..., T, 7) for chunks, or (..., 7) for single step
            # raw_state_8d shape: (..., 8) — single state for the chunk
            eef_xyz_0 = raw_state_8d[..., :3]  # (..., 3)
            eef_aa_0 = raw_state_8d[..., 3:6]  # (..., 3)
            R_eef_0 = axis_angle_to_matrix(eef_aa_0)  # (..., 3, 3)

            # Scale normalized deltas to physical units
            delta_xyz = action[..., :3] * OSC_POS_SCALE  # (..., T, 3) meters
            delta_aa = action[..., 3:6] * OSC_ROT_SCALE  # (..., T, 3) radians
            gripper = (action[..., 6] > 0).float().unsqueeze(-1)  # (..., T, 1) binary {0, 1}

            # Absolute xyz: state[0] + cumulative delta_xyz
            # action shape (..., T, 3): cumsum along T axis (axis=-2 since 3 is last)
            if action.ndim >= 2:
                cum_delta_xyz = delta_xyz.cumsum(dim=-2)  # (..., T, 3)
                abs_xyz = eef_xyz_0.unsqueeze(-2) + cum_delta_xyz  # (..., T, 3)
            else:
                abs_xyz = eef_xyz_0 + delta_xyz

            # Absolute rotation: R_abs[t] = R_delta[t] @ R_delta[t-1] @ ... @ R_eef_0
            # (accumulate delta rotations in world frame; approximate since we don't know exact EE frame at each step)
            R_delta = axis_angle_to_matrix(delta_aa)  # (..., T, 3, 3)
            if R_delta.ndim >= 3:
                # Cumulative rotation product along time axis
                T = R_delta.shape[-3]
                R_prev = R_eef_0
                abs_rot_list = []
                for t in range(T):
                    R_prev = R_delta[..., t, :, :] @ R_prev
                    abs_rot_list.append(R_prev)
                R_abs = torch.stack(abs_rot_list, dim=-3)  # (..., T, 3, 3)
                abs_rot6d = matrix_to_rot6d(R_abs)  # (..., T, 6)
            else:
                R_abs = R_delta @ R_eef_0
                abs_rot6d = matrix_to_rot6d(R_abs)

            left_arm_action = torch.cat([abs_xyz, abs_rot6d, gripper], dim=-1)  # (..., T, 10)
            right_arm_action = torch.zeros_like(left_arm_action)
            action_20d = torch.cat([left_arm_action, right_arm_action], dim=-1)  # (..., T, 20)
            new_transition[TransitionKey.ACTION] = action_20d

        # === 3. IMAGES: [0, 1] → ImageNet-normalized ===
        # Image tensors have shape (..., C, H, W) — the channel dim is at index -3.
        # Broadcast mean/std to (1, ..., 1, C, 1, 1) so they apply per-channel regardless of
        # leading batch/time dims.
        for key in list(observation.keys()):
            if not key.startswith(f"{OBS_IMAGES}."):
                continue
            img = observation[key]
            if not isinstance(img, torch.Tensor) or img.ndim < 3:
                continue
            # Idempotency check: if values exceed [0, 1] range, assume already normalized
            img_max = img.max().item() if img.numel() > 0 else 0.0
            img_min = img.min().item() if img.numel() > 0 else 0.0
            if img_max > 1.5 or img_min < -0.5:
                continue

            # Broadcast shape: put channels at dim -3, ones everywhere else
            bcast_shape = [1] * img.ndim
            bcast_shape[-3] = 3
            mean = torch.tensor(IMAGENET_MEAN, device=img.device, dtype=img.dtype).view(bcast_shape)
            std = torch.tensor(IMAGENET_STD, device=img.device, dtype=img.dtype).view(bcast_shape)
            observation[key] = (img - mean) / std

        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def transform_features(self, features):
        """Update feature shapes for downstream steps that check them."""
        from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature

        new_features = {ft: {k: v for k, v in feats.items()} for ft, feats in features.items()}

        # State becomes 20-D
        state_features = new_features.get(PipelineFeatureType.STATE, {})
        if OBS_STATE in state_features:
            state_features[OBS_STATE] = PolicyFeature(key=OBS_STATE, shape=(20,), dtype="float32")

        # Action becomes 20-D
        action_features = new_features.get(PipelineFeatureType.ACTION, {})
        if "action" in action_features:
            action_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(20,), dtype="float32")

        return new_features

    def get_config(self) -> dict[str, Any]:
        return {}
