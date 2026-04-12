"""Custom LIBERO env preprocessors for V3 (auto mode, raw delta actions).

The default lerobot XVLA pipeline assumes the dataset actions are in 20-D absolute EE6D format.
HuggingFaceVLA/libero actually stores raw 7-D delta actions with 8-D state. This module provides
preprocessors that match the dataset's native format, so V3 (trained on raw 7-D delta via
action_mode=auto) evaluates in the same distribution it was trained on.

Training data format (from HuggingFaceVLA/libero):
- observation.state: 8-D [eef_xyz(3), eef_axis_angle(3), gripper_qpos(2)]
- action: 7-D [delta_xyz(3), delta_axis_angle(3), gripper(-1 or 1)]  (OSC_POSE normalized)
- observation.images: raw [0, 1] float32

At eval, the env produces robot_state as a nested dict. We extract the same 8-D state and
DO NOT apply ImageNet normalization (training didn't either).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.pipeline import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_IMAGES, OBS_PREFIX, OBS_STATE


@dataclass
@ProcessorStepRegistry.register(name="libero_v3_state_extractor")
class LiberoV3StateExtractorStep(ProcessorStep):
    """Extract 8-D state from LIBERO env to match HuggingFaceVLA/libero dataset format.

    Dataset state: [eef_xyz(3), eef_axis_angle(3), gripper_qpos(2)] = 8-D
    Env provides: robot_state.eef.{pos,mat}, robot_state.gripper.qpos

    Also flips observation.images.image 180° to match dataset camera orientation.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if obs is None:
            return new_transition
        obs = dict(obs)  # copy

        # Flip image1 (first camera) 180° for dataset camera convention
        for key in list(obs.keys()):
            if key == f"{OBS_IMAGES}.image":
                img = obs[key]
                if isinstance(img, torch.Tensor):
                    obs[key] = torch.flip(img, dims=[-2, -1])

        # Extract 8-D state from nested robot_state
        robot_state_key = OBS_PREFIX + "robot_state"
        if robot_state_key in obs:
            rs = obs.pop(robot_state_key)
            eef_pos = rs["eef"]["pos"]  # (B, 3)
            eef_mat = rs["eef"]["mat"]  # (B, 3, 3)
            gripper_qpos = rs["gripper"]["qpos"]  # (B, 2)

            if not isinstance(eef_pos, torch.Tensor):
                eef_pos = torch.as_tensor(eef_pos, dtype=torch.float32)
                eef_mat = torch.as_tensor(eef_mat, dtype=torch.float32)
                gripper_qpos = torch.as_tensor(gripper_qpos, dtype=torch.float32)

            # Ensure batched
            if eef_pos.ndim == 1:
                eef_pos = eef_pos.unsqueeze(0)
                eef_mat = eef_mat.unsqueeze(0)
                gripper_qpos = gripper_qpos.unsqueeze(0)

            # Convert rotation matrix -> axis-angle (3D)
            eef_aa = self._mat_to_axis_angle(eef_mat)  # (B, 3)

            # Concatenate into 8-D state
            state_8d = torch.cat([eef_pos.float(), eef_aa.float(), gripper_qpos.float()], dim=-1)
            obs[OBS_STATE] = state_8d

        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    @staticmethod
    def _mat_to_axis_angle(rot_mats: torch.Tensor) -> torch.Tensor:
        """Convert batched rotation matrices (B, 3, 3) to axis-angle (B, 3).

        Uses the standard formula: angle = arccos((trace - 1) / 2),
        axis from (R - R^T) / (2 sin(angle)).
        """
        eps = 1e-6
        # trace
        trace = rot_mats[:, 0, 0] + rot_mats[:, 1, 1] + rot_mats[:, 2, 2]
        cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + eps, 1.0 - eps)
        theta = torch.acos(cos_theta)  # (B,)

        sin_theta = torch.sin(theta)
        # axis from skew-symmetric part
        rx = rot_mats[:, 2, 1] - rot_mats[:, 1, 2]
        ry = rot_mats[:, 0, 2] - rot_mats[:, 2, 0]
        rz = rot_mats[:, 1, 0] - rot_mats[:, 0, 1]
        axis = torch.stack([rx, ry, rz], dim=-1)  # (B, 3)

        # For theta near 0, return zero
        # For theta near π, use eigenvalue decomposition (simplified: just use the skew axis)
        denom = (2.0 * sin_theta).unsqueeze(-1).clamp(min=eps)
        axis_normalized = axis / denom  # unit axis
        axis_angle = axis_normalized * theta.unsqueeze(-1)  # magnitude = theta

        # For theta ≈ π (sin ≈ 0), use a different formula
        near_pi = (sin_theta.abs() < 1e-3) & (theta > 1.0)
        if near_pi.any():
            # R = I + 2 * axis * axis^T (for theta=π)
            # axis^2 = (R[ii] + 1) / 2
            d0 = torch.sqrt(torch.clamp((rot_mats[:, 0, 0] + 1.0) / 2.0, min=0.0))
            d1 = torch.sqrt(torch.clamp((rot_mats[:, 1, 1] + 1.0) / 2.0, min=0.0))
            d2 = torch.sqrt(torch.clamp((rot_mats[:, 2, 2] + 1.0) / 2.0, min=0.0))
            # Pick signs from off-diagonals
            sign_01 = torch.sign(rot_mats[:, 0, 1])
            sign_02 = torch.sign(rot_mats[:, 0, 2])
            # assume d0 is positive (largest of the three ideally)
            axis_pi = torch.stack([d0, d1 * sign_01, d2 * sign_02], dim=-1)
            # Normalize
            axis_pi = axis_pi / (axis_pi.norm(dim=-1, keepdim=True).clamp(min=eps))
            axis_angle_pi = axis_pi * np.pi
            axis_angle = torch.where(near_pi.unsqueeze(-1), axis_angle_pi, axis_angle)

        return axis_angle

    def transform_features(self, features):
        """State is now 8-D."""
        from lerobot.configs.types import PolicyFeature, FeatureType, PipelineFeatureType

        new_features = {ft: feats.copy() for ft, feats in features.items() if ft != PipelineFeatureType.STATE}
        new_features[PipelineFeatureType.STATE] = {
            OBS_STATE: PolicyFeature(key=OBS_STATE, shape=(8,), dtype="float32")
        }
        return new_features

    def get_config(self) -> dict[str, Any]:
        return {}


def make_xvla_libero_v3_pre_post_processors():
    """Build preprocessor that matches HuggingFaceVLA/libero training data format.
    
    - Uses LiberoV3StateExtractorStep (8-D state, no ImageNet normalization)
    - Adds domain_id=3 (LIBERO)
    - Empty postprocessor (V3 outputs native 7-D delta actions)
    """
    from lerobot.policies.xvla.processor_xvla import XVLAAddDomainIdProcessorStep
    from lerobot.processor import PolicyProcessorPipeline

    pre_steps = [
        LiberoV3StateExtractorStep(),
        XVLAAddDomainIdProcessorStep(domain_id=3),
    ]
    post_steps = []  # V3 auto mode: model outputs 7-D directly, no conversion needed
    return PolicyProcessorPipeline(steps=pre_steps), PolicyProcessorPipeline(steps=post_steps)
