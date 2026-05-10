"""Validate a limb-collected YAM LeRobot v3.0 dataset.

Runs the two checks we need before Phase 1 training:

1. **LeRobotDataset loadability**: the path LOBE training takes. If this fails,
   lerobot-train will fail the same way, so nothing else matters until it passes.
2. **Semantic/structural sanity**: state/action per-dim ranges, step-to-step
   smoothness, action-vs-state correlation, video/parquet frame-count alignment.
   This was previously `/tmp/eval_yam.py`; the logic is promoted here verbatim.

Usage:
    uv run python scripts/validate_yam_dataset.py ttotmoon/yam_pick_up_grey_cube
    uv run python scripts/validate_yam_dataset.py --local /path/to/dataset
    uv run python scripts/validate_yam_dataset.py ttotmoon/yam_pick_up_grey_cube --create-tag
    uv run python scripts/validate_yam_dataset.py ttotmoon/yam_pick_up_grey_cube --rebuild-stats
    uv run python scripts/validate_yam_dataset.py ttotmoon/yam_pick_up_grey_cube --output-json report.json

Fixer flags (opt-in, only applied on failure):
- `--create-tag` creates the `v<codebase_version>` tag on the HF repo when
  LeRobotDataset rejects the dataset for lack of a version tag. Shared-state
  write.
- `--rebuild-stats` aggregates per-episode stats (from the episodes parquet)
  into a fresh meta/stats.json in the local cache. Needed because limb's
  exporter currently writes a top-level stats.json without image/video
  features, which makes lerobot-train KeyError on any camera key. Local-only
  write; does not push to HF.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from loguru import logger

# lobe.video_compat patches lerobot's video decoding path. LeRobotDataset
# instantiation can touch video metadata before we ever sample a frame, so
# import this before anything from lerobot.
import lobe.video_compat  # noqa: F401

STATE_NAMES = [
    "left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5",
    "left_gripper",
    "right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5",
    "right_gripper",
]
EXPECTED_FPS = 30
EXPECTED_STATE_DIM = 14
EXPECTED_ACTION_DIM = 14
EXPECTED_CAMERAS = ["head_camera", "left_wrist_camera", "right_wrist_camera"]
EXPECTED_CODEBASE_VERSION = "v3.0"


def _try_lerobot_load(repo_id: str, local_root: Path | None) -> tuple[bool, str, object | None]:
    """Try loading via LeRobotDataset — the path actual training uses."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        kwargs = {"root": local_root} if local_root is not None else {}
        ds = LeRobotDataset(repo_id, **kwargs)
        return True, "ok", ds
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None


def _create_version_tag(repo_id: str, tag: str) -> None:
    """Create a git tag on an HF dataset repo so LeRobotDataset accepts it."""
    from huggingface_hub import HfApi

    api = HfApi()
    logger.info(f"Creating tag '{tag}' on HF dataset {repo_id}")
    api.create_tag(repo_id, tag=tag, repo_type="dataset")
    logger.info("Tag created.")


def _check_stats_completeness(info: dict, stats: dict | None) -> list[str]:
    """Top-level stats.json must have an entry for every dataset feature,
    including video features. Lerobot uses VISUAL: MEAN_STD normalization and
    will KeyError if a camera feature is missing."""
    if stats is None:
        return ["meta/stats.json missing entirely"]
    missing = []
    for key, ft in info.get("features", {}).items():
        if key in {"timestamp", "frame_index", "episode_index", "index", "task_index"}:
            continue
        if key not in stats:
            missing.append(f"{key} ({ft.get('dtype')})")
    return missing


def _rebuild_stats_from_episodes(ds_root: Path, info: dict) -> dict:
    """Aggregate per-episode stats (stored in meta/episodes/...parquet under
    stats/<feature>/<stat>) into a top-level stats dict.

    Mirrors the aggregation done during v2.1→v3.0 conversion in
    lerobot.datasets.dataset_tools (~line 848). Needed when limb's data writer
    produces a top-level stats.json without image/video features.
    """
    from lerobot.datasets.compute_stats import aggregate_stats

    episodes_df = pq.read_table(ds_root / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    features = info["features"]

    all_stats: list[dict[str, dict]] = []
    for _, row in episodes_df.iterrows():
        episode_stats: dict[str, dict] = {}
        for col in episodes_df.columns:
            if not col.startswith("stats/"):
                continue
            parts = col[len("stats/") :].split("/")
            if len(parts) != 2:
                continue
            feature_name, stat_name = parts
            value = row[col]

            if feature_name in features:
                dtype = features[feature_name]["dtype"]
                if dtype in ("image", "video") and stat_name != "count":
                    # Same nested-array unnesting pattern as lerobot's v2→v3 converter.
                    if isinstance(value, np.ndarray) and value.dtype == object:
                        flat: list[float] = []
                        for item in value:
                            while isinstance(item, np.ndarray):
                                item = item.flatten()[0]
                            flat.append(item)
                        value = np.array(flat, dtype=np.float64).reshape(3, 1, 1)
                    elif isinstance(value, np.ndarray) and value.shape == (3,):
                        value = value.reshape(3, 1, 1)
                else:
                    if isinstance(value, np.ndarray) and value.dtype == object:
                        value = np.array([float(v) for v in value], dtype=np.float64)
                    elif not isinstance(value, np.ndarray):
                        value = np.array(value, dtype=np.float64)

            episode_stats.setdefault(feature_name, {})[stat_name] = value
        all_stats.append(episode_stats)

    aggregated = aggregate_stats(all_stats)
    return {k: v for k, v in aggregated.items() if k in features}


def _write_stats(stats: dict, ds_root: Path) -> None:
    from lerobot.datasets.io_utils import write_stats

    write_stats(stats, ds_root)


def _fetch_info(repo_id: str, local_root: Path | None) -> dict:
    if local_root is not None:
        return json.loads((local_root / "meta/info.json").read_text())
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
    return json.loads(Path(path).read_text())


def _check_info_schema(info: dict, report: dict) -> None:
    """Match design-doc dataset contract."""
    problems = []
    if info.get("codebase_version") != EXPECTED_CODEBASE_VERSION:
        problems.append(f"codebase_version={info.get('codebase_version')} (expected {EXPECTED_CODEBASE_VERSION})")
    if info.get("fps") != EXPECTED_FPS:
        problems.append(f"fps={info.get('fps')} (expected {EXPECTED_FPS})")

    features = info.get("features", {})
    state = features.get("observation.state", {})
    if state.get("dtype") != "float32" or state.get("shape") != [EXPECTED_STATE_DIM]:
        problems.append(f"observation.state dtype/shape = {state.get('dtype')}/{state.get('shape')}")
    if state.get("names") != STATE_NAMES:
        problems.append("observation.state.names != canonical YAM names")

    action = features.get("action", {})
    if action.get("dtype") != "float32" or action.get("shape") != [EXPECTED_ACTION_DIM]:
        problems.append(f"action dtype/shape = {action.get('dtype')}/{action.get('shape')}")

    for cam in EXPECTED_CAMERAS:
        key = f"observation.images.{cam}"
        if key not in features:
            problems.append(f"missing camera feature {key}")
        elif features[key].get("dtype") != "video":
            problems.append(f"{key} dtype = {features[key].get('dtype')} (expected 'video')")

    report["info_schema"] = {"pass": not problems, "problems": problems}


def _download_one_episode(repo_id: str) -> tuple[Path, Path, list[Path]]:
    """Download episode 0's parquet + 3 videos from HF for low-level checks."""
    from huggingface_hub import hf_hub_download

    parquet_path = Path(
        hf_hub_download(repo_id, "data/chunk-000/file-000.parquet", repo_type="dataset")
    )
    episodes_parquet = Path(
        hf_hub_download(repo_id, "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset")
    )
    video_paths = [
        Path(
            hf_hub_download(
                repo_id,
                f"videos/observation.images.{cam}/chunk-000/file-000.mp4",
                repo_type="dataset",
            )
        )
        for cam in EXPECTED_CAMERAS
    ]
    return parquet_path, episodes_parquet, video_paths


def _check_trajectory_semantics(parquet_path: Path, report: dict) -> None:
    """Mirror /tmp/eval_yam.py's state/action checks against episode 0."""
    df = pq.read_table(parquet_path).to_pandas()
    if "observation.state" in df.columns:
        state = np.stack(df["observation.state"].to_list()).astype(np.float32)
        action = np.stack(df["action"].to_list()).astype(np.float32)
    else:
        state = np.stack([df[f"observation.state.{i}"].values for i in range(EXPECTED_STATE_DIM)], axis=1)
        action = np.stack([df[f"action.{i}"].values for i in range(EXPECTED_ACTION_DIM)], axis=1)

    per_dim = []
    for i, name in enumerate(STATE_NAMES):
        per_dim.append(
            {
                "dim": i,
                "name": name,
                "state_range": [float(state[:, i].min()), float(state[:, i].max())],
                "action_range": [float(action[:, i].min()), float(action[:, i].max())],
            }
        )

    daction = np.diff(action, axis=0)
    daction_norm = float(np.linalg.norm(daction, axis=1).mean())

    lag_corrs = []
    for i in range(EXPECTED_STATE_DIM):
        a = action[:-1, i]
        s_next = state[1:, i]
        if a.std() > 1e-6 and s_next.std() > 1e-6:
            lag_corrs.append(float(np.corrcoef(a, s_next)[0, 1]))

    report["trajectory"] = {
        "pass": bool(daction_norm < 1.0 and min(lag_corrs) > 0.5),
        "episode_0_frames": int(len(df)),
        "per_dim": per_dim,
        "action_stepwise_l2_mean": daction_norm,
        "action_state_lag1_corr_min": float(min(lag_corrs)),
        "action_state_lag1_corr_mean": float(np.mean(lag_corrs)),
    }


def _check_video_alignment(parquet_path: Path, video_paths: list[Path], report: dict) -> None:
    df = pq.read_table(parquet_path).to_pandas()
    parquet_frames = len(df)
    cameras = {}
    all_aligned = True
    for vp, cam in zip(video_paths, EXPECTED_CAMERAS):
        container = av.open(str(vp))
        stream = container.streams.video[0]
        v_frames = stream.frames
        codec = stream.codec_context.name
        rate = float(stream.average_rate)
        container.close()
        aligned = v_frames == parquet_frames
        all_aligned = all_aligned and aligned
        cameras[cam] = {
            "video_frames": int(v_frames),
            "parquet_frames": int(parquet_frames),
            "aligned": aligned,
            "codec": codec,
            "fps": rate,
        }
    report["video_alignment"] = {"pass": all_aligned, "cameras": cameras}


def _check_episodes_meta(episodes_parquet: Path, report: dict) -> None:
    df = pq.read_table(episodes_parquet).to_pandas()
    n = len(df)
    lengths = df["length"].tolist() if "length" in df.columns else []
    total = int(sum(lengths))
    report["episodes_meta"] = {
        "pass": n > 0 and total > 0,
        "num_episodes": n,
        "total_frames": total,
        "min_length": int(min(lengths)) if lengths else 0,
        "max_length": int(max(lengths)) if lengths else 0,
        "columns": list(df.columns),
    }


def _check_lerobot_sample(ds, report: dict) -> None:
    """If LeRobotDataset loaded, also sanity-check a sample."""
    sample = ds[0]
    keys = sorted(sample.keys())
    has_state = "observation.state" in sample
    has_action = "action" in sample
    camera_keys = [k for k in keys if k.startswith("observation.images.")]
    report["lerobot_sample"] = {
        "pass": has_state and has_action and len(camera_keys) == len(EXPECTED_CAMERAS),
        "num_episodes": ds.num_episodes,
        "num_frames": ds.num_frames,
        "fps": ds.fps,
        "sample_keys": keys,
        "camera_keys": camera_keys,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo_id",
        nargs="?",
        default="ttotmoon/yam_pick_up_grey_cube",
        help="HF repo id (default: %(default)s)",
    )
    parser.add_argument("--local", type=Path, default=None, help="Local dataset root (skip HF download)")
    parser.add_argument("--create-tag", action="store_true", help="If version tag is missing, create it on HF")
    parser.add_argument(
        "--rebuild-stats",
        action="store_true",
        help="If meta/stats.json is missing camera stats, rebuild it locally from per-episode stats",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Write structured report to JSON")
    args = parser.parse_args()

    report: dict = {"repo_id": args.repo_id, "local": str(args.local) if args.local else None}

    logger.info(f"Validating {args.repo_id} (local={args.local})")

    info = _fetch_info(args.repo_id, args.local)
    _check_info_schema(info, report)
    logger.info(f"info.json schema: {'PASS' if report['info_schema']['pass'] else 'FAIL'}")
    for p in report["info_schema"]["problems"]:
        logger.warning(f"  {p}")

    ok, msg, ds = _try_lerobot_load(args.repo_id, args.local)
    report["lerobot_load"] = {"pass": ok, "message": msg}
    if not ok:
        logger.error(f"LeRobotDataset load failed: {msg}")
        codebase_version = info.get("codebase_version", EXPECTED_CODEBASE_VERSION)
        missing_tag = False
        if args.local is None:
            from huggingface_hub import HfApi

            refs = HfApi().list_repo_refs(args.repo_id, repo_type="dataset")
            tags = [t.name for t in refs.tags]
            report["lerobot_load"]["hf_tags"] = tags
            missing_tag = codebase_version not in tags
        if missing_tag:
            logger.error(f"HF repo has no '{codebase_version}' tag (tags: {report['lerobot_load']['hf_tags']})")
            if args.create_tag and args.local is None:
                _create_version_tag(args.repo_id, codebase_version)
                ok, msg, ds = _try_lerobot_load(args.repo_id, args.local)
                report["lerobot_load"] = {"pass": ok, "message": msg, "fixed_by_creating_tag": True}
                if ok:
                    logger.info("LeRobotDataset load now succeeds after tag creation.")
                else:
                    logger.error(f"Still failing after tag creation: {msg}")
            else:
                logger.warning(
                    f"Fix: create a '{codebase_version}' tag on {args.repo_id}. Rerun with --create-tag, or:\n"
                    f"    from huggingface_hub import HfApi\n"
                    f"    HfApi().create_tag('{args.repo_id}', tag='{codebase_version}', repo_type='dataset')"
                )
        else:
            logger.warning("Tag exists — load failure has a different root cause.")
    if ok:
        logger.info(f"LeRobotDataset load: PASS (num_episodes={ds.num_episodes}, num_frames={ds.num_frames})")
        _check_lerobot_sample(ds, report)

        missing_stats = _check_stats_completeness(info, ds.meta.stats)
        report["stats_completeness"] = {"pass": not missing_stats, "missing": missing_stats}
        if missing_stats:
            logger.error(f"stats.json missing entries: {missing_stats}")
            if args.rebuild_stats:
                ds_root = Path(ds.meta.root)
                logger.info(f"Rebuilding {ds_root}/meta/stats.json from per-episode stats...")
                rebuilt = _rebuild_stats_from_episodes(ds_root, info)
                _write_stats(rebuilt, ds_root)
                logger.info(f"Rebuilt stats for {len(rebuilt)} features.")
                # Re-verify.
                ok2, msg2, ds2 = _try_lerobot_load(args.repo_id, args.local)
                if ok2:
                    missing2 = _check_stats_completeness(info, ds2.meta.stats)
                    report["stats_completeness"] = {
                        "pass": not missing2,
                        "missing": missing2,
                        "rebuilt": True,
                    }
                    logger.info(f"After rebuild: missing={missing2}")
            else:
                logger.warning("Rerun with --rebuild-stats to fix locally.")
        else:
            logger.info("stats_completeness: PASS")

    if args.local is not None:
        root = args.local
        parquet_path = root / "data/chunk-000/file-000.parquet"
        episodes_parquet = root / "meta/episodes/chunk-000/file-000.parquet"
        video_paths = [
            root / f"videos/observation.images.{cam}/chunk-000/file-000.mp4" for cam in EXPECTED_CAMERAS
        ]
    else:
        parquet_path, episodes_parquet, video_paths = _download_one_episode(args.repo_id)

    _check_trajectory_semantics(parquet_path, report)
    _check_video_alignment(parquet_path, video_paths, report)
    _check_episodes_meta(episodes_parquet, report)

    for section in ("trajectory", "video_alignment", "episodes_meta"):
        status = "PASS" if report[section]["pass"] else "FAIL"
        logger.info(f"{section}: {status}")

    overall = all(
        report[k].get("pass", True)
        for k in (
            "info_schema",
            "lerobot_load",
            "trajectory",
            "video_alignment",
            "episodes_meta",
            "stats_completeness",
        )
        if k in report
    )
    report["overall_pass"] = overall
    logger.info(f"overall: {'PASS' if overall else 'FAIL'}")

    if args.output_json:
        args.output_json.write_text(json.dumps(report, indent=2, default=str))
        logger.info(f"Report written to {args.output_json}")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
