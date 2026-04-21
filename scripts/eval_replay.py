"""Replay-based offline evaluation for trained policies on YAM data.

For each held-out episode, feeds each frame's observations to the policy,
collects predicted actions, and compares them to ground-truth demonstrated
actions. Reports per-joint MSE, L_inf, and aggregated metrics.

This is the primary automated quality gate before on-robot evaluation
(no sim available for YAM). See docs/milestones/yam_multibackbone.md Phase 3.

Usage:
    uv run python scripts/eval_replay.py \
        --policy.path=checkpoints/yam-grey-cube-dp-v0/checkpoints/050000/pretrained_model \
        --dataset.repo_id=ttotmoon/yam_pick_up_grey_cube \
        --eval_episodes 8 9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from loguru import logger

import lobe  # noqa: F401 — register policies + patches
import lobe.video_compat  # noqa: F401


def load_policy(policy_path: str, ds_meta, device: str = "cuda"):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path

    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    policy.eval()
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={
            "device_processor": {"device": device},
        },
    )
    return policy, preprocessor, postprocessor


def load_dataset(repo_id: str, root: str | None = None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    kwargs = {"root": Path(root)} if root else {}
    return LeRobotDataset(repo_id, **kwargs)


def _decode_all_video_frames(video_path: Path) -> torch.Tensor:
    """Decode all frames of a video at once. Returns (T, C, H, W) float32 [0, 1]."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.codec_context.thread_type = "AUTO"

    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        frames.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
    container.close()

    return torch.stack(frames)


def preload_episode(dataset, ep_idx: int, camera_keys: list[str]) -> dict:
    """Bulk-load an entire episode: parquet for state/action + video decode for cameras.

    Returns dict with keys:
      - observation.state: (T, state_dim) tensor
      - action: (T, action_dim) tensor
      - observation.images.<cam>: (T, C, H, W) tensor
      - task: str
      - n_frames: int
    """
    episodes = dataset.meta.episodes
    from_idx = int(episodes["dataset_from_index"][ep_idx])
    to_idx = int(episodes["dataset_to_index"][ep_idx])
    n_frames = to_idx - from_idx

    data_chunk = episodes["data/chunk_index"][ep_idx]
    data_file = episodes["data/file_index"][ep_idx]

    ds_root = Path(dataset.meta.root)

    parquet_path = ds_root / f"data/chunk-{data_chunk:03d}/file-{data_file:03d}.parquet"
    df = pq.read_table(parquet_path).to_pandas()

    state = torch.tensor(np.stack(df["observation.state"].to_list()), dtype=torch.float32)
    action = torch.tensor(np.stack(df["action"].to_list()), dtype=torch.float32)

    task_list = episodes["tasks"][ep_idx]
    task = task_list[0] if task_list else ""

    result = {
        "observation.state": state,
        "action": action,
        "task": task,
        "n_frames": n_frames,
    }

    for cam_key in camera_keys:
        cam_name = cam_key  # e.g. "observation.images.head_camera"
        chunk_col = f"videos/{cam_name}/chunk_index"
        file_col = f"videos/{cam_name}/file_index"
        v_chunk = episodes[chunk_col][ep_idx]
        v_file = episodes[file_col][ep_idx]
        video_path = ds_root / f"videos/{cam_name}/chunk-{v_chunk:03d}/file-{v_file:03d}.mp4"

        logger.info(f"  Decoding {cam_name} ({video_path.name})...")
        frames = _decode_all_video_frames(video_path)
        if len(frames) != n_frames:
            logger.warning(f"  {cam_name}: video has {len(frames)} frames, parquet has {n_frames}")
            frames = frames[:n_frames]
        result[cam_key] = frames

    return result


@torch.no_grad()
def eval_episode(
    policy, preprocessor, postprocessor, episode_data: dict, ep_idx: int, camera_keys: list[str]
) -> dict:
    """Run policy frame-by-frame on a preloaded episode and compute error metrics."""
    n_frames = episode_data["n_frames"]
    logger.info(f"Running inference on {n_frames} frames...")

    policy.reset()
    predicted_actions = []

    for t in range(n_frames):
        obs: dict = {"observation.state": episode_data["observation.state"][t].unsqueeze(0)}
        for cam_key in camera_keys:
            obs[cam_key] = episode_data[cam_key][t].unsqueeze(0)
        obs["task"] = episode_data["task"]

        obs = preprocessor(obs)
        action = policy.select_action(obs)
        action = postprocessor(action)

        predicted_actions.append(action.squeeze(0).cpu().float().numpy())

    predicted = np.stack(predicted_actions)
    ground_truth = episode_data["action"].numpy()

    per_joint_mse = np.mean((predicted - ground_truth) ** 2, axis=0)
    per_joint_linf = np.max(np.abs(predicted - ground_truth), axis=0)
    mse = float(np.mean(per_joint_mse))
    linf = float(np.max(per_joint_linf))

    return {
        "episode": ep_idx,
        "n_frames": n_frames,
        "mse": mse,
        "linf": linf,
        "per_joint_mse": per_joint_mse.tolist(),
        "per_joint_linf": per_joint_linf.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy.path", dest="policy_path", required=True)
    parser.add_argument("--dataset.repo_id", dest="repo_id", default="ttotmoon/yam_pick_up_grey_cube")
    parser.add_argument("--dataset.root", dest="dataset_root", default=None)
    parser.add_argument("--eval_episodes", nargs="+", type=int, default=[8, 9])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    logger.info(f"Loading dataset {args.repo_id}")
    dataset = load_dataset(args.repo_id, args.dataset_root)

    logger.info(f"Loading policy from {args.policy_path}")
    policy, preprocessor, postprocessor = load_policy(args.policy_path, dataset.meta, args.device)

    camera_keys = [k for k in dataset.meta.features if k.startswith("observation.images.")]
    logger.info(f"Camera keys: {camera_keys}")

    results = []
    for ep_idx in args.eval_episodes:
        if ep_idx >= dataset.num_episodes:
            logger.warning(f"Skipping episode {ep_idx} — dataset only has {dataset.num_episodes}")
            continue

        logger.info(f"Episode {ep_idx}: preloading...")
        episode_data = preload_episode(dataset, ep_idx, camera_keys)
        r = eval_episode(policy, preprocessor, postprocessor, episode_data, ep_idx, camera_keys)
        results.append(r)
        logger.info(
            f"  Episode {ep_idx}: MSE={r['mse']:.6f}  L_inf={r['linf']:.4f}  "
            f"per-joint MSE: {[f'{x:.5f}' for x in r['per_joint_mse']]}"
        )

    if results:
        avg_mse = float(np.mean([r["mse"] for r in results]))
        avg_linf = float(np.mean([r["linf"] for r in results]))
        logger.info(f"AGGREGATE: avg_MSE={avg_mse:.6f}  avg_L_inf={avg_linf:.4f}")
    else:
        avg_mse = avg_linf = float("nan")
        logger.error("No episodes evaluated.")

    report = {
        "policy_path": args.policy_path,
        "repo_id": args.repo_id,
        "eval_episodes": args.eval_episodes,
        "episodes": results,
        "avg_mse": avg_mse,
        "avg_linf": avg_linf,
    }

    if args.output_json:
        args.output_json.write_text(json.dumps(report, indent=2, default=str))
        logger.info(f"Report written to {args.output_json}")

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
