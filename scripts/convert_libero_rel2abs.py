"""Port of upstream X-VLA rel2abs.py for HuggingFaceVLA/libero.

Upstream X-VLA trains on `abs_action_6d` (10-D absolute EE6D per frame):
    [abs_xyz(3), abs_rot6d(6), gripper(1)]

These are extracted by REPLAYING each demo in the LIBERO simulator and reading the
OSC controller's `goal_pos` / `goal_ori` after each step — these are the absolute
targets X-VLA was pretrained to predict. HuggingFaceVLA/libero only ships raw delta
actions. This script replays every demo and caches `abs_action_6d` per episode.

Optimizations vs a naive port:
- Load the LeRobot source dataset ONCE (not per-episode).
- Group episodes by task; reuse env across all episodes in a task.
- Pre-compute settled eef positions for all 50 init states per task (one-time cost).
- Match episodes to init states by nearest-neighbor on the pre-computed positions.

Reference: https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/rel2abs.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import torch
import tyro
from loguru import logger

import lobe  # noqa: F401 — apply lerobot patches

# weights_only=False needed for LIBERO init state pickle files
_orig_torch_load = torch.load
torch.load = lambda *a, **k: _orig_torch_load(*a, **{**k, "weights_only": False})

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

BENCHMARK_NAMES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
NUM_STEPS_WAIT = 10  # same as upstream rel2abs.py
NOOP_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D (first 2 columns flattened)."""
    return np.concatenate([mat[:3, 0], mat[:3, 1]], axis=-1)


def build_task_index(benchmark_names: list[str]) -> dict[str, tuple[str, int]]:
    """Map task description → (benchmark_name, task_id)."""
    index: dict[str, tuple[str, int]] = {}
    for bm_name in benchmark_names:
        bm = get_benchmark(bm_name)()
        for task_id in range(len(bm.tasks)):
            task = bm.get_task(task_id)
            key = task.language.strip().lower()
            index[key] = (bm_name, task_id)
    return index


def load_init_states(bm_name: str, task_id: int) -> np.ndarray:
    bm = get_benchmark(bm_name)()
    return bm.get_task_init_states(task_id)


def make_env(bm_name: str, task_id: int) -> OffScreenRenderEnv:
    bm = get_benchmark(bm_name)()
    task = bm.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    return env


def compute_settled_eef_positions(env, init_states: np.ndarray) -> np.ndarray:
    """Settle each of the N init states and return the eef_pos after NUM_STEPS_WAIT no-ops.

    Returns:
        (N, 3) settled eef positions — used for fast episode→init_state matching.
    """
    settled = np.zeros((init_states.shape[0], 3), dtype=np.float32)
    for i in range(init_states.shape[0]):
        env.reset()
        env.set_init_state(init_states[i])
        obs = None
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(NOOP_ACTION)
        settled[i] = obs["robot0_eef_pos"]
    return settled


def find_nearest_init(settled: np.ndarray, target_xyz: np.ndarray) -> tuple[int, float]:
    """Return (index, distance) of the init state whose settled eef is nearest to target."""
    dists = np.linalg.norm(settled - target_xyz[None, :], axis=1)
    idx = int(np.argmin(dists))
    return idx, float(dists[idx])


def replay_episode(env, init_state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Replay delta actions; return (T, 10) abs_action_6d."""
    env.reset()
    env.set_init_state(init_state)
    for _ in range(NUM_STEPS_WAIT):
        env.step(NOOP_ACTION)

    out = np.empty((actions.shape[0], 10), dtype=np.float32)
    for t, action in enumerate(actions):
        env.step(action)
        ctrl = env.env.robots[0].controller
        out[t, :3] = ctrl.goal_pos
        out[t, 3:9] = mat_to_rot6d(ctrl.goal_ori)
        out[t, 9] = action[-1]
    return out


def main(
    cache_dir: str = "/mnt/localssd/sunlingfeng/datasets/libero_abs_action_6d_cache",
    start: int = 0,
    end: int | None = None,
    skip_existing: bool = True,
):
    """Convert HuggingFaceVLA/libero raw delta actions → abs_action_6d via LIBERO replay.

    Args:
        cache_dir: where to save per-episode .npz files
        start: first episode index to process
        end: last episode index (exclusive); defaults to all episodes
        skip_existing: skip episodes whose .npz already exists
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    logger.info("Building task index...")
    task_index = build_task_index(BENCHMARK_NAMES)
    logger.info(f"Indexed {len(task_index)} tasks")

    logger.info("Loading HuggingFaceVLA/libero dataset...")
    full_ds = LeRobotDataset("HuggingFaceVLA/libero")
    num_episodes = full_ds.num_episodes
    if end is None:
        end = num_episodes
    logger.info(f"Source: {num_episodes} episodes total; processing [{start}, {end})")

    # Get per-episode metadata via meta.episodes (HF Dataset)
    # Each row has: episode_index, dataset_from_index, dataset_to_index, tasks, length
    meta_eps = full_ds.meta.episodes
    ep_info: dict[int, tuple[str, int, int]] = {}
    for row in meta_eps:
        ep_idx = int(row["episode_index"])
        if ep_idx < start or ep_idx >= end:
            continue
        frame_start = int(row["dataset_from_index"])
        frame_end = int(row["dataset_to_index"])
        task_desc = row["tasks"][0] if isinstance(row["tasks"], list) else row["tasks"]
        ep_info[ep_idx] = (task_desc, frame_start, frame_end)

    # Group episodes by task description
    task_to_episodes: dict[str, list[int]] = {}
    for ep_idx, (task_desc, _, _) in ep_info.items():
        task_to_episodes.setdefault(task_desc, []).append(ep_idx)
    logger.info(f"Episodes grouped into {len(task_to_episodes)} tasks in range")

    t0 = time.perf_counter()
    ok = 0
    skipped = 0
    failed = 0

    for task_idx, (task_desc, ep_indices) in enumerate(task_to_episodes.items()):
        key = task_desc.strip().lower()
        if key not in task_index:
            logger.warning(f"Task '{task_desc}' not in any benchmark, skipping {len(ep_indices)} episodes")
            failed += len(ep_indices)
            continue

        bm_name, task_id = task_index[key]
        logger.info(
            f"[task {task_idx+1}/{len(task_to_episodes)}] {bm_name}/{task_id} — "
            f"{len(ep_indices)} episodes — '{task_desc[:50]}...'"
        )

        # Check which episodes actually need conversion
        needed = []
        for ep_idx in ep_indices:
            out_path = cache_path / f"episode_{ep_idx:06d}.npz"
            if skip_existing and out_path.exists():
                ok += 1
            else:
                needed.append(ep_idx)
        if not needed:
            continue

        # Create env and pre-compute settled positions for all 50 init states (one-time)
        env = make_env(bm_name, task_id)
        init_states = load_init_states(bm_name, task_id)
        settled = compute_settled_eef_positions(env, init_states)

        # Process each episode for this task
        for ep_idx in needed:
            _, frame_start, frame_end = ep_info[ep_idx]
            first = full_ds[frame_start]
            target_xyz = first["observation.state"][:3].numpy()

            init_idx, dist = find_nearest_init(settled, target_xyz)
            if dist > 0.05:
                logger.warning(f"Episode {ep_idx}: init match dist={dist:.3f} (>5cm)")

            # Gather actions
            n_frames = frame_end - frame_start
            actions = np.stack(
                [full_ds[frame_start + t]["action"].numpy() for t in range(n_frames)]
            )

            try:
                abs_action_6d = replay_episode(env, init_states[init_idx], actions)
            except Exception as e:
                logger.error(f"Episode {ep_idx}: replay failed: {e}")
                failed += 1
                continue

            out_path = cache_path / f"episode_{ep_idx:06d}.npz"
            np.savez(
                out_path,
                abs_action_6d=abs_action_6d,
                task_description=task_desc,
                benchmark_name=bm_name,
                task_id=task_id,
                init_state_idx=init_idx,
                match_dist=dist,
            )
            ok += 1

        env.close()

        # Progress log
        elapsed = time.perf_counter() - t0
        total_range = end - start
        rate = ok / max(elapsed, 1e-6)
        eta = (total_range - ok) / max(rate, 1e-6)
        logger.info(
            f"Progress: {ok}/{total_range} ok, {failed} failed — "
            f"{rate:.2f} ep/s, eta {eta/60:.1f} min"
        )

    logger.success(
        f"Done: {ok}/{end-start} converted, {failed} failed, took {(time.perf_counter()-t0)/60:.1f} min"
    )


if __name__ == "__main__":
    tyro.cli(main)
