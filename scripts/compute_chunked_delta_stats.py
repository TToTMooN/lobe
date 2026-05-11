"""Compute delta_stats.json over the CHUNKED mixed-delta distribution.

This is the input file for FM v2's openpi-style mixed-delta + Q01-Q99 normalization.
The result matches what `openpi/scripts/compute_norm_stats.py` produces — each
`(frame_t, chunk_offset i)` pair in [0, H-1] contributes ONE delta point:

    delta[i, dim] = action[t+i, dim] - state[t, dim]   if mask[dim] == 1 (joint)
    delta[i, dim] = action[t+i, dim]                    if mask[dim] == 0 (gripper)

Stats (mean / std / min / max / q01 / q99) are computed over all collected points.
This matches OpenPI's RunningStats which flattens (B, H, A) → (B*H, A) and treats
every (sample, in-chunk timestep) as an independent observation.

Common pitfall: computing stats over i=0 only (single-step deltas) yields q01/q99
that are 1.5–2× too narrow. Multi-step deltas (i > 0) grow larger as i increases.

Usage:
    uv run python scripts/compute_chunked_delta_stats.py \\
        --dataset-root /path/to/dataset_image \\
        --horizon 32 \\
        --action-mask 1,1,1,1,1,1,0,1,1,1,1,1,1,0
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from loguru import logger


def collect_all_deltas(data_dir: str, horizon: int, mask: np.ndarray):
    """Iterate all parquet files, compute mixed-delta for every (t, i) pair.

    Returns: ndarray of shape (TOTAL_STEPS, action_dim).
    """
    all_files = sorted(glob.glob(f"{data_dir}/*.parquet"))
    logger.info(f"Found {len(all_files)} parquet file(s) under {data_dir}")

    all_deltas = []
    total_chunks = 0
    for f in all_files:
        df = pq.read_table(f, columns=["observation.state", "action", "episode_index"]).to_pandas()
        states = np.stack(df["observation.state"].to_list()).astype(np.float32)
        actions = np.stack(df["action"].to_list()).astype(np.float32)
        eps = df["episode_index"].to_numpy()
        N = len(df)
        logger.debug(f"  {os.path.basename(f)}: {N} frames, {len(np.unique(eps))} eps")

        for t in range(N - 1):
            ep_t = eps[t]
            max_i = 0
            for i in range(horizon):
                if t + i < N and eps[t + i] == ep_t:
                    max_i = i
                else:
                    break
            if max_i == 0:
                continue
            chunk_actions = actions[t : t + max_i + 1]
            state_t = states[t]
            chunk_delta = chunk_actions - state_t[None, :] * mask[None, :]
            all_deltas.append(chunk_delta)
            total_chunks += 1

    arr = np.concatenate(all_deltas, axis=0)
    logger.info(f"Total starting frames: {total_chunks}")
    logger.info(f"Total delta points: {len(arr)}")
    return arr


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, help="Root of LeRobot dataset (contains data/ and meta/)")
    parser.add_argument("--horizon", type=int, required=True, help="Action chunk horizon used by the policy")
    parser.add_argument("--action-mask", type=str, required=True, help="Comma-separated 0/1 mask, 1=joint(delta), 0=gripper(absolute)")
    parser.add_argument("--out", default=None, help="Output path (default: <root>/meta/delta_stats.json)")
    args = parser.parse_args()

    mask = np.asarray([int(x) for x in args.action_mask.split(",")], dtype=np.float32)
    data_dir = f"{args.dataset_root}/data/chunk-000"
    out_path = args.out or f"{args.dataset_root}/meta/delta_stats.json"

    deltas = collect_all_deltas(data_dir, args.horizon, mask)

    stats = {
        "mean": deltas.mean(axis=0).tolist(),
        "std": deltas.std(axis=0).tolist(),
        "min": deltas.min(axis=0).tolist(),
        "max": deltas.max(axis=0).tolist(),
        "q01": np.quantile(deltas, 0.01, axis=0).tolist(),
        "q99": np.quantile(deltas, 0.99, axis=0).tolist(),
    }

    above = float((deltas > np.asarray(stats["q99"])[None, :]).sum()) / deltas.size
    below = float((deltas < np.asarray(stats["q01"])[None, :]).sum()) / deltas.size
    logger.info(f"Fraction above q99: {above*100:.2f}% (expect ~1%)")
    logger.info(f"Fraction below q01: {below*100:.2f}% (expect ~1%)")

    logger.info("Final stats:")
    for key in ["q01", "q99", "min", "max", "mean", "std"]:
        logger.info(f"  {key}: {np.asarray(stats[key]).round(3).tolist()}")

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
