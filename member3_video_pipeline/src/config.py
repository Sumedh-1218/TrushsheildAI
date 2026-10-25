"""
TrustShield AI — Member 3 — Deepfake Video Pipeline
src/config.py

Central configuration for the entire pipeline.
All limits are defined here — never hardcoded in individual modules.
"""

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """
    Central configuration for the deepfake video pipeline.

    All values are configurable. The defaults are reasonable hackathon-ready values.
    Do NOT scatter these values throughout the codebase — import this config instead.
    """

    # ─── File limits ───────────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: float = 500.0
    """Maximum allowed video file size in megabytes."""

    # ─── Duration & frame limits ───────────────────────────────────────────────
    MAX_DURATION_SECONDS: float = 300.0
    """Maximum allowed video duration in seconds (5 minutes)."""

    MAX_FRAMES: int = 300
    """Hard cap on the number of frames the pipeline will sample and analyze."""

    SAMPLING_INTERVAL_SECONDS: float = 1.0
    """Sample approximately one frame every N seconds."""

    # ─── Detection threshold ───────────────────────────────────────────────────
    DETECTION_THRESHOLD: float = 0.5
    """
    Score threshold above which a frame is considered high-scoring.
    This does NOT automatically mean the video is fake — see responsible AI notes.
    """

    # ─── Evidence ─────────────────────────────────────────────────────────────
    TOP_K_EVIDENCE: int = 5
    """Number of top high-scoring frames to include in evidence output."""

    # ─── Face processing ──────────────────────────────────────────────────────
    MIN_FACE_SIZE: int = 30
    """Minimum face dimension (pixels). Smaller faces are rejected as unscorable."""

    FACE_SCALE_FACTOR: float = 1.1
    """Haar cascade scale factor for face detection."""

    FACE_MIN_NEIGHBORS: int = 5
    """Haar cascade min-neighbors parameter for face detection."""

    # ─── Supported video formats (magic-byte validated) ──────────────────────
    SUPPORTED_EXTENSIONS: tuple = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    """
    Allowed file extensions. Used as a first-pass filter only.
    Actual format is verified by attempting to open with OpenCV.
    """

    # ─── Score semantics label ────────────────────────────────────────────────
    SCORE_TYPE: str = "model_score"
    """
    Label used to describe the individual frame score.
    Use 'model_score' — NOT 'probability'. A raw model output is not a calibrated probability.
    """

    AGGREGATE_SCORE_TYPE: str = "aggregate_model_score"
    """Label for the video-level aggregated score."""


# Default configuration instance — import this in all modules
DEFAULT_CONFIG = PipelineConfig()
