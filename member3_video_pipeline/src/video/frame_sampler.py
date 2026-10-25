"""
TrustShield AI — Member 3 — Frame Sampler
src/video/frame_sampler.py

Samples frames from a validated video at a configurable interval.

Design:
  - Default: ~1 frame per second
  - MAX_FRAMES hard cap prevents runaway processing
  - Every sampled frame carries frame_index + timestamp_seconds
  - The image array is a numpy ndarray in BGR format (OpenCV default)

Why sample instead of processing all frames?
  At 30fps, a 5-minute video = 9,000 frames.
  Processing all 9,000 is slow and wasteful — most frames are very similar.
  Sampling at 1fps gives ~300 frames — sufficient for detection while fast.
"""

from __future__ import annotations

from typing import Generator, List

import cv2

from src.config import PipelineConfig, DEFAULT_CONFIG
from src.schemas.result_schema import SampledFrame


class FrameSampler:
    """
    Samples frames from a video file at a configurable time interval.

    Usage:
        sampler = FrameSampler(config)
        for frame in sampler.sample(video_path, fps):
            # frame.image is the BGR numpy array
            # frame.frame_index is the original index
            # frame.timestamp_seconds is the time in the video
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.config = config

    def sample(self, video_path: str, fps: float) -> Generator[SampledFrame, None, None]:
        """
        Yield SampledFrame objects from the video.

        Algorithm:
          - Compute frame_step = fps * SAMPLING_INTERVAL_SECONDS
          - Seek to each target frame index using CAP_PROP_POS_FRAMES
          - Stop after MAX_FRAMES frames or end of video

        Args:
            video_path: Path to a validated video file.
            fps:        FPS from metadata (used to compute frame step).

        Yields:
            SampledFrame with frame_index, timestamp_seconds, and image.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"FrameSampler: Cannot open video: {video_path}")

        try:
            frame_step = max(1, int(round(fps * self.config.SAMPLING_INTERVAL_SECONDS)))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_index = 0
            sampled_count = 0

            while frame_index < total_frames and sampled_count < self.config.MAX_FRAMES:
                # Seek directly to the target frame index (fast seek)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, image = cap.read()

                if not ret or image is None:
                    # Frame decode failure — skip this frame and continue
                    frame_index += frame_step
                    continue

                timestamp_seconds = frame_index / fps if fps > 0 else 0.0

                yield SampledFrame(
                    frame_index=frame_index,
                    timestamp_seconds=round(timestamp_seconds, 4),
                    image=image,
                )

                sampled_count += 1
                frame_index += frame_step

        finally:
            cap.release()

    def sample_all(self, video_path: str, fps: float) -> List[SampledFrame]:
        """
        Return all sampled frames as a list.
        Only use this for short videos — prefer the generator for memory efficiency.
        """
        return list(self.sample(video_path, fps))


# ─── Convenience function ─────────────────────────────────────────────────────

def sample_frames(
    video_path: str,
    fps: float,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> Generator[SampledFrame, None, None]:
    """Sample frames from a validated video file."""
    return FrameSampler(config).sample(video_path, fps)
