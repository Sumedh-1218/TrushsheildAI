"""
TrustShield AI — Member 3 — Video Reader
src/video/video_reader.py

Opens a validated video file and extracts its metadata using OpenCV.
Does NOT perform frame-by-frame reading here — that is the frame sampler's job.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2

from src.config import PipelineConfig, DEFAULT_CONFIG
from src.schemas.result_schema import VideoMetadata


class VideoReader:
    """
    Opens a video file with OpenCV and extracts metadata.

    Always call validate_video() before using this class.
    This class assumes the video has already passed validation.
    """

    def __init__(self, video_path: str, config: PipelineConfig = DEFAULT_CONFIG):
        self.video_path = video_path
        self.config = config

    def extract_metadata(self) -> VideoMetadata:
        """
        Extract metadata from the video file.

        Returns a VideoMetadata dataclass.
        Raises RuntimeError if the video cannot be opened.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(
                f"VideoReader: Cannot open video: {self.video_path}"
            )

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Compute duration from frame_count / fps (reliable for most containers)
            duration_seconds = frame_count / fps if fps > 0 else 0.0

            # Codec fourcc → human-readable string
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = _fourcc_to_str(fourcc_int)

            # Container from extension
            _, ext = os.path.splitext(self.video_path)
            container = ext.lstrip(".").upper() if ext else "UNKNOWN"

        finally:
            cap.release()

        return VideoMetadata(
            width=width,
            height=height,
            fps=round(fps, 4),
            frame_count=frame_count,
            duration_seconds=round(duration_seconds, 4),
            codec=codec,
            container=container,
        )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fourcc_to_str(fourcc_int: int) -> str:
    """Convert OpenCV fourcc integer to a 4-character codec string."""
    if fourcc_int == 0:
        return "UNKNOWN"
    try:
        chars = [chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]
        return "".join(c if c.isprintable() else "?" for c in chars).strip()
    except Exception:
        return "UNKNOWN"


# ─── Convenience function ─────────────────────────────────────────────────────

def extract_metadata(video_path: str, config: PipelineConfig = DEFAULT_CONFIG) -> VideoMetadata:
    """Extract metadata from a validated video file."""
    return VideoReader(video_path, config).extract_metadata()
