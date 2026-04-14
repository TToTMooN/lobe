"""Convert 2toINF/Libero-XVLA-format HDF5 demos → LeRobot v3.0 dataset.

Source (downloaded separately with `hf download`):
  /mnt/localssd/sunlingfeng/datasets/libero_xvla_format/{suite}/{task_dir}/demo_*.hdf5

Each HDF5 demo has `abs_action_6d` (T, 10) **already correct** — computed by upstream X-VLA
via `rel2abs.py` over OpenVLA-regenerated LIBERO demos, which preserve init_state_id. We
use it verbatim, no sim replay and no nearest-neighbor matching. This eliminates the
5-30 mm xyz offsets that V10's parquet carries.

Uses LeRobotDataset.create(use_videos=False) so the v3.0 layout (info.json, tasks.parquet
with pandas index format, episodes parquet with data/chunk_index + stats/*, etc.) is
produced natively by lerobot instead of us reverse-engineering the schema.

State conversion (matches V10's rewrite_libero_state_body_to_site.py):
    eef_quat (xyzw, robosuite body frame) → 3x3 matrix → rot6d (col0, col1) [body frame]
    → rotate to grip-site frame via R_z(-90°):
        rot6d_site = [-body_col1; +body_col0]
    → concat [eef_pos(3), rot6d_site(6), 0(extra), zeros(10)] = 20-D (matches
      LiberoProcessorStep output at eval time).
"""

from __future__ import annotations

import io
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import h5py
import numpy as np
import tyro
from loguru import logger
from PIL import Image

import torch

_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    """(N, 4) xyzw → (N, 3, 3). robosuite convention."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    out = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    out[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    out[:, 0, 1] = 2.0 * (xy - wz)
    out[:, 0, 2] = 2.0 * (xz + wy)
    out[:, 1, 0] = 2.0 * (xy + wz)
    out[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    out[:, 1, 2] = 2.0 * (yz - wx)
    out[:, 2, 0] = 2.0 * (xz - wy)
    out[:, 2, 1] = 2.0 * (yz + wx)
    out[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return out


def build_state_20d(eef_pos: np.ndarray, eef_quat_xyzw: np.ndarray) -> np.ndarray:
    """Build 20-D EE6D state in grip_site frame."""
    R_body = quat_xyzw_to_mat(eef_quat_xyzw)  # (T, 3, 3)
    body_col0 = R_body[:, :, 0]  # (T, 3)
    body_col1 = R_body[:, :, 1]  # (T, 3)
    # R_site = R_body @ R_z(-90°) → col0_site = -body_col1, col1_site = body_col0
    rot6d_site = np.concatenate([-body_col1, body_col0], axis=-1).astype(np.float32)
    extra = np.zeros((eef_pos.shape[0], 1), dtype=np.float32)
    proprio_10d = np.concatenate([eef_pos.astype(np.float32), rot6d_site, extra], axis=-1)
    zeros_10d = np.zeros_like(proprio_10d)
    return np.concatenate([proprio_10d, zeros_10d], axis=-1)


def build_task_index() -> dict[str, tuple[str, int]]:
    """Map task.language.lower() → (suite_name, task_id)."""
    from libero.libero.benchmark import get_benchmark

    idx: dict[str, tuple[str, int]] = {}
    for suite in SUITES:
        bm = get_benchmark(suite)()
        for tid in range(len(bm.tasks)):
            idx[bm.get_task(tid).language.strip().lower()] = (suite, tid)
    return idx


def resolve_benchmark_language(task_base: str, suite: str, task_idx: dict[str, tuple[str, int]]) -> str | None:
    """Given a HDF5 task_base like 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it',
    find the matching benchmark task.language via suffix match."""
    normalized = task_base.replace("_", " ").lower()
    for lang, (s, _) in task_idx.items():
        if s != suite:
            continue
        if normalized.endswith(lang.strip()):
            return lang
    return None


def main(
    src_root: str = "/mnt/localssd/sunlingfeng/datasets/libero_xvla_format",
    out_root: str = "/mnt/localssd/sunlingfeng/datasets/local/libero_xvla_v12",
    repo_id: str = "local/libero_xvla_v12",
) -> None:
    # Clear any previous output
    import shutil
    if Path(out_root).exists():
        shutil.rmtree(out_root)

    # lerobot import AFTER torch.load patch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: F401

    # Feature spec matching V10's schema.
    # LeRobotDataset.create() parses these into the v3.0 info.json format for us.
    features = {
        "observation.images.image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.image2": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (20,),
            "names": [
                "eef_x", "eef_y", "eef_z",
                "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5",
                "extra",
                "pad_0", "pad_1", "pad_2", "pad_3", "pad_4", "pad_5", "pad_6", "pad_7", "pad_8", "pad_9",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (10,),
            "names": [
                "abs_xyz_x", "abs_xyz_y", "abs_xyz_z",
                "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5",
                "gripper",
            ],
        },
    }

    logger.info("Creating empty LeRobot v3.0 dataset...")
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        features=features,
        root=out_root,
        robot_type="libero_panda",
        use_videos=False,
    )

    task_idx = build_task_index()
    logger.info(f"Indexed {len(task_idx)} LIBERO tasks")

    # Enumerate demo HDF5s
    src = Path(src_root)
    demos: list[tuple[Path, str]] = []  # (path, task_language)
    for suite in SUITES:
        suite_dir = src / suite
        if not suite_dir.exists():
            logger.warning(f"Missing suite dir: {suite_dir}")
            continue
        for task_dir in sorted(suite_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_base = task_dir.name.removesuffix("_demo")
            lang = resolve_benchmark_language(task_base, suite, task_idx)
            if lang is None:
                logger.warning(f"No benchmark match for {suite}/{task_dir.name}")
                continue
            for demo_file in sorted(task_dir.glob("demo_*.hdf5")):
                demos.append((demo_file, lang))
    logger.info(f"Found {len(demos)} demo files")

    # Process each demo
    import time as _time

    t0 = _time.perf_counter()
    for demo_idx, (demo_path, task_lang) in enumerate(demos):
        with h5py.File(demo_path, "r") as f:
            abs_action_6d = f["abs_action_6d"][()].astype(np.float32)  # (T, 10)
            eef_pos = f["eef_pos"][()].astype(np.float32)  # (T, 3)
            eef_quat = f["eef_quat"][()].astype(np.float32)  # (T, 4) xyzw
            third_raw = f["observation/third_image"][()]  # (T,) JPEG bytes
            wrist_raw = f["observation/wrist_image"][()]  # (T,) JPEG bytes

        T = abs_action_6d.shape[0]
        state_20d = build_state_20d(eef_pos, eef_quat)  # (T, 20)

        for t in range(T):
            # Decode JPEG → numpy array (HWC, uint8). LeRobotDataset.add_frame
            # expects images as numpy arrays or torch tensors in HWC uint8 format
            # (it will handle the JPEG/PNG encoding internally when writing parquet).
            img_third = np.asarray(Image.open(io.BytesIO(third_raw[t])).convert("RGB"), dtype=np.uint8)
            img_wrist = np.asarray(Image.open(io.BytesIO(wrist_raw[t])).convert("RGB"), dtype=np.uint8)
            frame = {
                "observation.images.image": img_third,
                "observation.images.image2": img_wrist,
                "observation.state": state_20d[t],
                "action": abs_action_6d[t],
                "task": task_lang,
            }
            ds.add_frame(frame)

        ds.save_episode()

        if (demo_idx + 1) % 50 == 0:
            rate = (demo_idx + 1) / max(_time.perf_counter() - t0, 1e-6)
            eta = (len(demos) - demo_idx - 1) / max(rate, 1e-6)
            logger.info(
                f"[{demo_idx+1}/{len(demos)}] rate={rate:.1f} demos/s eta={eta/60:.1f}min"
            )

    logger.success(f"Finished: {ds.num_episodes} eps, {ds.num_frames} frames")


if __name__ == "__main__":
    tyro.cli(main)
