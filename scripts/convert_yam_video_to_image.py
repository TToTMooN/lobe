"""Fast video→image conversion for YAM LeRobot datasets.

Bulk-decodes all video frames at once per episode (O(n)) and writes a new
LeRobot-compatible image-format dataset. This eliminates the ~20× video
decode bottleneck in training.

Usage:
    uv run python scripts/convert_yam_video_to_image.py \
        --repo_id ttotmoon/yam_pick_up_grey_cube \
        --output local/yam_pick_up_grey_cube_image
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

import av
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from PIL import Image

import lobe.video_compat  # noqa: F401


def decode_all_frames(video_path: Path) -> list[bytes]:
    """Decode all frames from a video and return as JPEG bytes."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.codec_context.thread_type = "AUTO"

    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=95)
        frames.append(buf.getvalue())
    container.close()
    return frames


def convert(repo_id: str, output_repo_id: str, resize: tuple[int, int] | None = None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    logger.info(f"Loading source dataset: {repo_id}")
    ds = LeRobotDataset(repo_id)
    src_root = Path(ds.meta.root)
    info = json.loads((src_root / "meta/info.json").read_text())

    camera_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    logger.info(f"Camera features to convert: {camera_keys}")

    dst_root = Path.home() / ".cache/huggingface/lerobot" / output_repo_id
    dst_root.mkdir(parents=True, exist_ok=True)
    (dst_root / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (dst_root / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)

    episodes = ds.meta.episodes
    n_eps = len(episodes)

    for ep_idx in range(n_eps):
        data_chunk = episodes["data/chunk_index"][ep_idx]
        data_file = episodes["data/file_index"][ep_idx]
        src_parquet = src_root / f"data/chunk-{data_chunk:03d}/file-{data_file:03d}.parquet"

        logger.info(f"Episode {ep_idx}: reading parquet + decoding videos...")
        df = pq.read_table(src_parquet).to_pandas()

        for cam_key in camera_keys:
            v_chunk = episodes[f"videos/{cam_key}/chunk_index"][ep_idx]
            v_file = episodes[f"videos/{cam_key}/file_index"][ep_idx]
            video_path = src_root / f"videos/{cam_key}/chunk-{v_chunk:03d}/file-{v_file:03d}.mp4"

            logger.info(f"  {cam_key}: decoding {video_path.name}...")
            jpeg_frames = decode_all_frames(video_path)

            if len(jpeg_frames) != len(df):
                logger.warning(f"  {cam_key}: video={len(jpeg_frames)} vs parquet={len(df)} frames")
                jpeg_frames = jpeg_frames[: len(df)]

            if resize:
                resized = []
                for jpg in jpeg_frames:
                    img = Image.open(io.BytesIO(jpg))
                    img = img.resize((resize[1], resize[0]), Image.BILINEAR)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=95)
                    resized.append(buf.getvalue())
                jpeg_frames = resized

            df[cam_key] = [{"bytes": b, "path": None} for b in jpeg_frames]

        dst_parquet = dst_root / f"data/chunk-000/file-{ep_idx:03d}.parquet"
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, dst_parquet)
        logger.info(f"  Wrote {dst_parquet} ({dst_parquet.stat().st_size / 1e6:.1f} MB)")

    # Copy and update meta
    shutil.copy2(src_root / "meta/tasks.parquet", dst_root / "meta/tasks.parquet")
    shutil.copy2(
        src_root / "meta/episodes/chunk-000/file-000.parquet",
        dst_root / "meta/episodes/chunk-000/file-000.parquet",
    )

    # Update episodes parquet: remove video columns
    ep_df = pq.read_table(dst_root / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    video_cols = [c for c in ep_df.columns if c.startswith("videos/")]
    ep_df = ep_df.drop(columns=video_cols)
    pq.write_table(pa.Table.from_pandas(ep_df, preserve_index=False),
                    dst_root / "meta/episodes/chunk-000/file-000.parquet")

    # Update info.json
    new_info = dict(info)
    shape = info["features"][camera_keys[0]]["shape"]
    if resize:
        shape = [resize[0], resize[1], 3]
    for cam_key in camera_keys:
        new_info["features"][cam_key] = {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }
    new_info.pop("video_path", None)
    (dst_root / "meta/info.json").write_text(json.dumps(new_info, indent=2))

    # Copy stats
    if (src_root / "meta/stats.json").exists():
        shutil.copy2(src_root / "meta/stats.json", dst_root / "meta/stats.json")

    logger.info(f"Done! Image dataset at: {dst_root}")
    logger.info(f"Use with: --dataset.repo_id={output_repo_id} --dataset.root={dst_root}")

    # Verify
    logger.info("Verifying load...")
    try:
        ds2 = LeRobotDataset(output_repo_id, root=dst_root)
        s = ds2[0]
        for cam_key in camera_keys:
            logger.info(f"  {cam_key}: shape={s[cam_key].shape}")
        logger.info(f"  Verified: {ds2.num_episodes} episodes, {ds2.num_frames} frames")
    except Exception as e:
        logger.error(f"  Verification failed: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", default="ttotmoon/yam_pick_up_grey_cube")
    parser.add_argument("--output", default="local/yam_pick_up_grey_cube_image")
    parser.add_argument("--resize", type=int, nargs=2, default=None, help="H W to resize images")
    args = parser.parse_args()
    convert(args.repo_id, args.output, resize=tuple(args.resize) if args.resize else None)


if __name__ == "__main__":
    main()
