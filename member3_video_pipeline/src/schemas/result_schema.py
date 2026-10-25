"""
TrustShield AI — Member 3 — Schemas
src/schemas/result_schema.py

All data structures for the pipeline — typed dataclasses, JSON serializable.
Every module produces and consumes these types.

Design principle:
    Use dataclasses.asdict() to convert to dict/JSON anywhere in the codebase.
    This avoids tight coupling to any specific serialization library.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional


# ─── Video Metadata ───────────────────────────────────────────────────────────

@dataclass
class VideoMetadata:
    """Extracted metadata from a video file."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: str
    container: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ─── Sampled Frame ────────────────────────────────────────────────────────────

@dataclass
class SampledFrame:
    """A single frame sampled from the video, with position metadata."""
    frame_index: int
    """Original frame index in the video (0-based)."""
    timestamp_seconds: float
    """Precise timestamp within the video in seconds."""
    image: Any  # numpy ndarray — not serializable, excluded from JSON output

    def to_dict(self) -> dict:
        """Returns a JSON-serializable dict (excludes raw image data)."""
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
        }


# ─── Detector Output ──────────────────────────────────────────────────────────

@dataclass
class DetectorOutput:
    """
    Raw output from a detector (mock or real).
    The pipeline always validates this structure before using the score.
    """
    model_score: float
    """Raw score from the model. Semantics depend on model contract."""
    model_name: str
    """Name of the detector that produced this result."""
    model_version: str
    """Version string of the detector."""
    score_type: str
    """Semantic label for the score — e.g. 'model_score', never 'probability'."""
    is_mock: bool = True
    """True if this result came from the mock detector."""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ─── Frame Result ─────────────────────────────────────────────────────────────

FRAME_STATUS_SCORED = "scored"
FRAME_STATUS_UNSCORABLE = "unscorable"
FRAME_STATUS_FAILED = "failed"


@dataclass
class FrameResult:
    """
    Analysis result for a single sampled frame.

    Status values:
        scored      — detector successfully returned a score
        unscorable  — frame could not be scored (e.g. no face found)
        failed      — detector raised an error
    """
    frame_index: int
    timestamp_seconds: float
    status: str  # "scored" | "unscorable" | "failed"
    model_score: Optional[float] = None
    """Score from detector. None if status != 'scored'."""
    score_type: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    is_mock: bool = True
    unscorable_reason: Optional[str] = None
    """Explains why the frame was unscorable (e.g., 'no_face_detected')."""
    error: Optional[str] = None
    """Error message if status == 'failed'."""

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "status": self.status,
            "model_score": round(self.model_score, 4) if self.model_score is not None else None,
            "score_type": self.score_type,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "is_mock": self.is_mock,
            "unscorable_reason": self.unscorable_reason,
            "error": self.error,
        }


# ─── Aggregation Result ───────────────────────────────────────────────────────

@dataclass
class AggregationResult:
    """
    Video-level analytical summary derived from all frame results.

    IMPORTANT: This is an analytical summary, NOT a calibrated probability
    that the video is fake. Do not present it as such.
    """
    mean_score: Optional[float]
    """Mean score across all scored frames. None if no frames were scored."""
    max_score: Optional[float]
    """Highest score seen across all frames."""
    scored_frames: int
    unscorable_frames: int
    failed_frames: int
    frames_above_threshold: int
    """Frames with model_score > DETECTION_THRESHOLD."""
    high_score_ratio: Optional[float]
    """Fraction of scored frames above threshold. None if no frames scored."""
    highest_score_timestamp: Optional[float]
    """Timestamp of the frame with the highest model score."""
    top_frames: list[FrameResult] = field(default_factory=list)
    """Top-K highest-scoring frames, ordered by score descending."""

    def to_dict(self) -> dict:
        return {
            "mean_score": round(self.mean_score, 4) if self.mean_score is not None else None,
            "max_score": round(self.max_score, 4) if self.max_score is not None else None,
            "scored_frames": self.scored_frames,
            "unscorable_frames": self.unscorable_frames,
            "failed_frames": self.failed_frames,
            "frames_above_threshold": self.frames_above_threshold,
            "high_score_ratio": round(self.high_score_ratio, 4) if self.high_score_ratio is not None else None,
            "highest_score_timestamp": self.highest_score_timestamp,
            "top_frames": [f.to_dict() for f in self.top_frames],
        }


# ─── Evidence Result ──────────────────────────────────────────────────────────

@dataclass
class EvidenceResult:
    """
    Timestamped evidence for why a video was flagged.
    This is the data that the frontend/Member 1 uses to explain the result.
    """
    primary_evidence: Optional[FrameResult]
    """The single highest-scoring frame."""
    top_frames: list[FrameResult]
    """Top-K highest-scoring frames."""
    frames_above_threshold: int
    high_score_ratio: Optional[float]
    unscorable_frames: int

    def to_dict(self) -> dict:
        return {
            "primary_evidence": self.primary_evidence.to_dict() if self.primary_evidence else None,
            "top_frames": [f.to_dict() for f in self.top_frames],
            "frames_above_threshold": self.frames_above_threshold,
            "high_score_ratio": round(self.high_score_ratio, 4) if self.high_score_ratio is not None else None,
            "unscorable_frames": self.unscorable_frames,
        }


# ─── Video Label ──────────────────────────────────────────────────────────────

LABEL_FLAGGED = "flagged_for_review"
LABEL_SUSPICIOUS = "suspicious"
LABEL_INCONCLUSIVE = "inconclusive"
LABEL_ERROR = "error"

LIMITATIONS = [
    "Results are indicative only — not a forensic determination.",
    "Mock detector in use — DEVELOPMENT MODE ONLY.",
    "Frame sampling may miss brief manipulations between sampled frames.",
    "Face-based detection depends on face visibility and quality.",
    "Highly compressed or low-resolution video reduces reliability.",
    "Score is not a calibrated probability.",
]


# ─── Final Video Analysis Result ─────────────────────────────────────────────

@dataclass
class VideoAnalysisResult:
    """
    The final JSON-serializable result produced by analyze_video().
    This is the contract between Member 3 and Member 1.
    """
    status: str
    """'completed' | 'failed'"""
    module: str = "member3_video_pipeline"
    label: Optional[str] = None
    """flagged_for_review | suspicious | inconclusive | error"""
    score: Optional[float] = None
    """Aggregate model score (mean across scored frames)."""
    score_type: Optional[str] = None
    video: Optional[VideoMetadata] = None
    frames_analyzed: int = 0
    frames_unscorable: int = 0
    frames_failed: int = 0
    aggregation: Optional[AggregationResult] = None
    evidence: Optional[EvidenceResult] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    is_mock: bool = True
    limitations: list[str] = field(default_factory=list)
    error: Optional[str] = None
    runtime_seconds: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "module": self.module,
            "label": self.label,
            "score": round(self.score, 4) if self.score is not None else None,
            "score_type": self.score_type,
            "video": self.video.to_dict() if self.video else None,
            "frames_analyzed": self.frames_analyzed,
            "frames_unscorable": self.frames_unscorable,
            "frames_failed": self.frames_failed,
            "aggregation": self.aggregation.to_dict() if self.aggregation else None,
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "model": {
                "name": self.model_name,
                "version": self.model_version,
                "is_mock": self.is_mock,
            },
            "limitations": self.limitations,
            "error": self.error,
            "runtime_seconds": round(self.runtime_seconds, 3) if self.runtime_seconds is not None else None,
        }
