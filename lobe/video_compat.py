"""Video decoding compatibility patch for PyTorch nightly.

torchvision nightly removed `VideoReader`, breaking lerobot's video decoding.
This patches lerobot to use PyAV directly for video frame decoding.

Import this module early (before lerobot dataset loading) to apply the patch:
    import lobe.video_compat  # noqa: F401
"""

import av
import numpy as np
import torch
from loguru import logger


def decode_video_frames_pyav(video_path, timestamps, tolerance_s):
    """Decode video frames using PyAV directly, matching lerobot's interface."""
    video_path = str(video_path)

    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.codec_context.thread_type = "AUTO"

    time_base = float(stream.time_base)

    # Collect all frames
    all_frames = []
    all_pts = []
    for frame in container.decode(video=0):
        all_frames.append(frame)
        all_pts.append(frame.pts * time_base)

    container.close()

    # Match requested timestamps to nearest frames
    all_pts = np.array(all_pts)
    matched_frames = []
    for ts in timestamps:
        idx = np.argmin(np.abs(all_pts - ts))
        if abs(all_pts[idx] - ts) > tolerance_s:
            raise ValueError(f"No frame within tolerance {tolerance_s}s of timestamp {ts}s")
        frame = all_frames[idx]
        img = frame.to_ndarray(format="rgb24")
        matched_frames.append(torch.from_numpy(img).permute(2, 0, 1))  # HWC -> CHW

    return torch.stack(matched_frames)


def _apply_patch():
    """Monkey-patch lerobot's video_utils to use our PyAV decoder."""
    try:
        import lerobot.datasets.video_utils as video_utils

        video_utils.decode_video_frames_torchcodec = decode_video_frames_pyav

        def _torchvision_compat(video_path, timestamps, tolerance_s, *args, **kw):
            return decode_video_frames_pyav(video_path, timestamps, tolerance_s)

        video_utils.decode_video_frames_torchvision = _torchvision_compat

        # Patch get_safe_default_codec to always return "torchcodec" (which now points to our pyav impl)
        video_utils.get_safe_default_codec = lambda: "torchcodec"

        logger.debug("Patched lerobot video decoding to use PyAV directly")
    except ImportError:
        pass


_apply_patch()
