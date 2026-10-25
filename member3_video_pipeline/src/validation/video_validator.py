"""
TrustShield AI — Member 3 — Video Validator
src/validation/video_validator.py

Validates a video file before it enters the pipeline.
Security principle: treat all uploaded files as untrusted input.

Validation checks (in order):
  1. File exists
  2. File is readable
  3. File size <= MAX_FILE_SIZE_MB
  4. Extension is in allowed list (first-pass only)
  5. File is actually openable by OpenCV (proves it's a real video container)
  6. FPS is valid
  7. Duration <= MAX_DURATION_SECONDS
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2

from src.config import PipelineConfig, DEFAULT_CONFIG


# ─── Validation Result ────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of video file validation."""
    is_valid: bool
    error: Optional[str] = None
    error_code: Optional[str] = None

    @classmethod
    def ok(cls) -> "ValidationResult":
        return cls(is_valid=True)

    @classmethod
    def fail(cls, error: str, error_code: str) -> "ValidationResult":
        return cls(is_valid=False, error=error, error_code=error_code)


# ─── Validator ────────────────────────────────────────────────────────────────

class VideoValidator:
    """
    Validates a video file before the pipeline processes it.

    All checks are explicit and ordered from cheapest to most expensive.
    The file is NEVER executed — only opened with OpenCV for metadata reading.
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.config = config

    def validate(self, video_path: str) -> ValidationResult:
        """
        Run all validation checks on the given path.

        Returns a ValidationResult — check .is_valid before proceeding.
        """
        # 1. File existence
        if not os.path.exists(video_path):
            return ValidationResult.fail(
                f"File not found: {video_path}",
                "FILE_NOT_FOUND"
            )

        # 2. Is a file (not a directory)
        if not os.path.isfile(video_path):
            return ValidationResult.fail(
                f"Path is not a file: {video_path}",
                "NOT_A_FILE"
            )

        # 3. File size check
        size_bytes = os.path.getsize(video_path)
        if size_bytes == 0:
            return ValidationResult.fail(
                "File is empty (0 bytes).",
                "EMPTY_FILE"
            )
        max_bytes = self.config.MAX_FILE_SIZE_MB * 1024 * 1024
        if size_bytes > max_bytes:
            size_mb = size_bytes / (1024 * 1024)
            return ValidationResult.fail(
                f"File size {size_mb:.1f} MB exceeds limit of {self.config.MAX_FILE_SIZE_MB} MB.",
                "FILE_TOO_LARGE"
            )

        # 4. Extension check (first-pass; not trusted alone)
        _, ext = os.path.splitext(video_path)
        if ext.lower() not in self.config.SUPPORTED_EXTENSIONS:
            return ValidationResult.fail(
                f"Unsupported file extension '{ext}'. "
                f"Supported: {self.config.SUPPORTED_EXTENSIONS}",
                "UNSUPPORTED_EXTENSION"
            )

        # 5. Attempt to open with OpenCV (proves it's a decodable video)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return ValidationResult.fail(
                "Video cannot be opened. File may be corrupt or unsupported format.",
                "CANNOT_OPEN"
            )

        # 6. FPS check
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 500:
            cap.release()
            return ValidationResult.fail(
                f"Invalid FPS value: {fps}. Video may be corrupt.",
                "INVALID_FPS"
            )

        # 7. Duration check
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if frame_count <= 0:
            cap.release()
            return ValidationResult.fail(
                "Video reports no frames. File may be corrupt.",
                "NO_FRAMES"
            )

        duration = frame_count / fps
        if duration > self.config.MAX_DURATION_SECONDS:
            cap.release()
            return ValidationResult.fail(
                f"Video duration {duration:.1f}s exceeds limit of "
                f"{self.config.MAX_DURATION_SECONDS}s.",
                "DURATION_EXCEEDED"
            )

        cap.release()
        return ValidationResult.ok()


# ─── Convenience function ─────────────────────────────────────────────────────

def validate_video(video_path: str, config: PipelineConfig = DEFAULT_CONFIG) -> ValidationResult:
    """Validate a video file using the given configuration."""
    return VideoValidator(config).validate(video_path)
