"""Port of upstream X-VLA rel2abs.py for HuggingFaceVLA/libero.

Upstream X-VLA trains on `abs_action_6d` (10-D absolute EE6D actions per frame):
    [abs_xyz(3), abs_rot6d(6), gripper(1)]

These are extracted by REPLAYING each raw LIBERO demo in the simulator and reading the
OSC controller's `goal_pos` / `goal_ori` after each step — these are the controller's
absolute targets, which is what X-VLA was pretrained to predict.

HuggingFaceVLA/libero only ships the raw delta actions. This script replays every demo
in LIBERO and produces the corresponding `abs_action_6d` per frame. Results are cached
as `.npz` files per episode under `--cache-dir`. A follow-up script (`build_libero_abs_dataset.py`)
writes a new LeRobot dataset using these cached absolute actions.

Reference: https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/rel2abs.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import torch
import tyro
from loguru import logger

import lobe  # noqa: F401 — ensures lerobot patches apply

# Monkey-patch torch.load for LIBERO init-state files (weights_only=False needed)
_orig_torch_load = torch.load
torch.load = lambda *a, **k: _orig_torch_load(*a, **{**k, "weights_only": False})

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

BENCHMARK_NAMES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
NUM_STEPS_WAIT = 10  # same as rel2abs.py — let sim settle before executing actions


def mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D representation (first 2 columns flattened)."""
    return np.concatenate([mat[:3, 0], mat[:3, 1]], axis=-1)


def build_task_index(benchmark_names: list[str]) -> dict[str, tuple[str, int]]:
    """Map task description → (benchmark_name, task_id)."""
    index: dict[str, tuple[str, int]] = {}
    for bm_name in benchmark_names:
        bm = get_benchmark(bm_name)()
        for task_id in range(len(bm.tasks)):
            task = bm.get_task(task_id)
            key = task.language.strip().lower()
            if key in index:
                logger.warning(f"Duplicate task '{key}' in {bm_name} and {index[key][0]}")
            index[key] = (bm_name, task_id)
    return index


def load_init_states_cache(benchmark_names: list[str]) -> dict[tuple[str, int], np.ndarray]:
    """Pre-load init_states for all tasks (50 states × 123 dims each)."""
    cache: dict[tuple[str, int], np.ndarray] = {}
    for bm_name in benchmark_names:
        bm = get_benchmark(bm_name)()
        for task_id in range(len(bm.tasks)):
            cache[(bm_name, task_id)] = bm.get_task_init_states(task_id)
    return cache


def make_env(bm_name: str, task_id: int) -> OffScreenRenderEnv:
    """Instantiate a single-task LIBERO env."""
    bm = get_benchmark(bm_name)()
    task = bm.get_task(task_id)
    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=256, camera_widths=256)
    env.seed(0)
    return env


def find_init_state_index(env, init_states: np.ndarray, target_xyz: np.ndarray) -> tuple[int, float]:
    """Find the init_state whose settled eef_xyz is closest to `target_xyz`.

    Early-exits if a state matches within 1cm.
    """
    best_idx, best_dist = -1, float("inf")
    noop = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    for i in range(init_states.shape[0]):
        env.reset()
        env.set_init_state(init_states[i])
        obs = None
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(noop)
        dist = float(np.linalg.norm(obs["robot0_eef_pos"] - target_xyz))
        if dist < best_dist:
            best_dist, best_idx = dist, i
        if dist < 0.01:
            return i, dist
    return best_idx, best_dist


def replay_episode_and_capture(
    env, init_state: np.ndarray, actions: np.ndarray
) -> np.ndarray:
    """Replay delta actions and capture abs_action_6d per step.

    Returns:
        [T, 10] — [abs_xyz(3), rot6d(6), gripper(1)] per frame.
    """
    env.reset()
    env.set_init_state(init_state)
    noop = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    for _ in range(NUM_STEPS_WAIT):
        env.step(noop)

    abs_list = []
    for action in actions:
        env.step(action)
        ctrl = env.env.robots[0].controller
        goal_pos = ctrl.goal_pos.copy()
        goal_ori = ctrl.goal_ori.copy()
        rot6d = mat_to_rot6d(goal_ori)
        abs_list.append(np.concatenate([goal_pos, rot6d, action[-1:]]))
    return np.stack(abs_list).astype(np.float32)


def convert_episodes(
    cache_dir: Path,
    episodes: list[int] | None = None,
    skip_existing: bool = True,
) -> None:
    """Run rel2abs conversion for specified episode indices (or all)."""
    logger.info(f"Building task index across {BENCHMARK_NAMES}...")
    task_index = build_task_index(BENCHMARK_NAMES)
    logger.info(f"Indexed {len(task_index)} tasks")

    logger.info("Loading init-states cache...")
    init_states_cache = load_init_states_cache(BENCHMARK_NAMES)

    logger.info("Loading HuggingFaceVLA/libero dataset metadata...")
    full_ds = LeRobotDataset("HuggingFaceVLA/libero", episodes=episodes)
    total_episodes = full_ds.num_episodes if episodes is None else len(episodes)
    logger.info(f"Will process {total_episodes} episodes")

    cache_dir.mkdir(parents=True, exist_ok=True)
    current_bm_task: tuple[str, int] | None = None
    env: OffScreenRenderEnv | None = None

    episode_list = episodes if episodes is not None else list(range(full_ds.num_episodes))
    t0 = time.perf_counter()
    ok_count = 0
    fail_count = 0

    for ep_idx in episode_list:
        out_path = cache_dir / f"episode_{ep_idx:06d}.npz"
        if skip_existing and out_path.exists():
            ok_count += 1
            continue

        # Load this episode's frames: need task, observation.state[:3] for init matching, actions
        ep_ds = LeRobotDataset("HuggingFaceVLA/libero", episodes=[ep_idx])
        if len(ep_ds) == 0:
            logger.warning(f"Episode {ep_idx} is empty, skipping")
            fail_count += 1
            continue

        first = ep_ds[0]
        task_desc = first["task"]
        key = task_desc.strip().lower()
        if key not in task_index:
            logger.warning(f"Episode {ep_idx}: task '{task_desc}' not in any benchmark, skipping")
            fail_count += 1
            continue

        bm_name, task_id = task_index[key]
        target_xyz = first["observation.state"][:3].numpy()

        # Recreate env only when task changes (expensive)
        if (bm_name, task_id) != current_bm_task:
            if env is not None:
                env.close()
            env = make_env(bm_name, task_id)
            current_bm_task = (bm_name, task_id)

        init_states = init_states_cache[(bm_name, task_id)]
        init_idx, dist = find_init_state_index(env, init_states, target_xyz)
        if dist > 0.05:
            logger.warning(
                f"Episode {ep_idx}: init-state match poor (dist={dist:.3f}); continuing anyway"
            )

        # Gather all actions for this episode
        actions = np.stack([ep_ds[i]["action"].numpy() for i in range(len(ep_ds))])

        try:
            abs_action_6d = replay_episode_and_capture(env, init_states[init_idx], actions)
        except Exception as e:
            logger.error(f"Episode {ep_idx}: replay failed: {e}")
            fail_count += 1
            continue

        np.savez(
            out_path,
            abs_action_6d=abs_action_6d,
            task_description=task_desc,
            benchmark_name=bm_name,
            task_id=task_id,
            init_state_idx=init_idx,
            match_dist=dist,
        )
        ok_count += 1
        if ok_count % 20 == 0:
            elapsed = time.perf_counter() - t0
            rate = ok_count / elapsed
            eta = (total_episodes - ok_count) / max(rate, 1e-6)
            logger.info(
                f"[{ok_count}/{total_episodes}] ({100*ok_count/total_episodes:.1f}%) "
                f"rate={rate:.2f} ep/s eta={eta/60:.1f} min fails={fail_count}"
            )

    if env is not None:
        env.close()
    logger.success(
        f"Done: {ok_count}/{total_episodes} converted, {fail_count} failed, "
        f"took {(time.perf_counter()-t0)/60:.1f} min"
    )


def main(
    cache_dir: str = "/mnt/localssd/sunlingfeng/datasets/libero_abs_action_6d_cache",
    episodes: int | None = None,
    start: int = 0,
    skip_existing: bool = True,
):
    """Convert HuggingFaceVLA/libero raw delta actions → abs_action_6d via LIBERO replay.

    Args:
        cache_dir: where to save per-episode .npz files
        episodes: if set, process only the first N episodes (for testing)
        start: start from episode index (use with `episodes` for chunking)
        skip_existing: skip episodes whose .npz already exists
    """
    episode_list: list[int] | None = None
    if episodes is not None:
        episode_list = list(range(start, start + episodes))

    convert_episodes(Path(cache_dir), episodes=episode_list, skip_existing=skip_existing)


if __name__ == "__main__":
    tyro.cli(main)
