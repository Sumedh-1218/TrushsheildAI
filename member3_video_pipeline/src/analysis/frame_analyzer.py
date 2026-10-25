"""
TrustShield AI — Member 3 — Frame Analyzer
src/analysis/frame_analyzer.py

Orchestrates the per-frame analysis pipeline:
  SampledFrame → face extraction → detector → validation → FrameResult

Rules:
  - No silent failures. Every error is captured in the FrameResult.
  - Unexpected detector output raises immediately (not silently passed through).
  - No face detected → status='unscorable', NOT status='real'.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.config import PipelineConfig, DEFAULT_CONFIG
from src.detector.detector_adapter import BaseDetector, DetectorError
from src.face.face_processor import BaseFaceProcessor
from src.schemas.result_schema import (
    FrameResult,
    SampledFrame,
    FRAME_STATUS_SCORED,
    FRAME_STATUS_UNSCORABLE,
    FRAME_STATUS_FAILED,
)


class FrameAnalyzer:
    """
    Analyzes a single sampled frame through the full analysis chain.

    Dependencies injected through constructor — no global state.
    """

    def __init__(
        self,
        detector: BaseDetector,
        face_processor: BaseFaceProcessor,
        config: PipelineConfig = DEFAULT_CONFIG,
    ):
        self.detector = detector
        self.face_processor = face_processor
        self.config = config

    def analyze(self, frame: SampledFrame) -> FrameResult:
        """
        Analyze a single sampled frame.

        Pipeline:
          1. Extract face from frame image
          2. If no face → mark unscorable (NOT real)
          3. Call detector on face crop
          4. Validate detector output (score must be float in [0, 1])
          5. Return structured FrameResult

        Args:
            frame: A SampledFrame with frame_index, timestamp_seconds, image.

        Returns:
            FrameResult with status 'scored', 'unscorable', or 'failed'.
        """
        # ── Step 1: Face extraction ────────────────────────────────────────────
        try:
            face_image: Optional[np.ndarray] = self.face_processor.extract_face(frame.image)
        except Exception as exc:
            return FrameResult(
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                status=FRAME_STATUS_FAILED,
                error=f"Face extraction error: {exc}",
                is_mock=self.detector.is_mock,
            )

        # ── Step 2: No face → unscorable ──────────────────────────────────────
        if face_image is None:
            return FrameResult(
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                status=FRAME_STATUS_UNSCORABLE,
                unscorable_reason="no_face_detected",
                model_name=self.detector.model_name,
                model_version=self.detector.model_version,
                is_mock=self.detector.is_mock,
            )

        # ── Step 3: Run detector ───────────────────────────────────────────────
        try:
            detector_output = self.detector.analyze(face_image)
        except DetectorError as exc:
            return FrameResult(
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                status=FRAME_STATUS_FAILED,
                error=f"Detector error [{exc.error_code}]: {exc}",
                model_name=self.detector.model_name,
                model_version=self.detector.model_version,
                is_mock=self.detector.is_mock,
            )
        except Exception as exc:
            return FrameResult(
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                status=FRAME_STATUS_FAILED,
                error=f"Unexpected detector exception: {exc}",
                model_name=self.detector.model_name,
                model_version=self.detector.model_version,
                is_mock=self.detector.is_mock,
            )

        # ── Step 4: Validate detector output ──────────────────────────────────
        validation_error = _validate_detector_output(detector_output.model_score)
        if validation_error:
            return FrameResult(
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                status=FRAME_STATUS_FAILED,
                error=f"Invalid detector output: {validation_error}",
                model_name=detector_output.model_name,
                model_version=detector_output.model_version,
                is_mock=detector_output.is_mock,
            )

        # ── Step 5: Success ────────────────────────────────────────────────────
        return FrameResult(
            frame_index=frame.frame_index,
            timestamp_seconds=frame.timestamp_seconds,
            status=FRAME_STATUS_SCORED,
            model_score=detector_output.model_score,
            score_type=detector_output.score_type,
            model_name=detector_output.model_name,
            model_version=detector_output.model_version,
            is_mock=detector_output.is_mock,
        )


# ─── Validation helper ────────────────────────────────────────────────────────

def _validate_detector_output(score: object) -> Optional[str]:
    """
    Returns an error message if the score is invalid, or None if it's fine.

    Valid score: float or int in [0.0, 1.0].
    """
    if score is None:
        return "Score is None."
    if not isinstance(score, (int, float)):
        return f"Score type {type(score).__name__} is not numeric."
    if not (0.0 <= float(score) <= 1.0):
        return f"Score {score} is outside expected range [0.0, 1.0]."
    return None
