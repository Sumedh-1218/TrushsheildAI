"""
TrustShield AI — Member 3 — Detector Adapter
src/detector/detector_adapter.py

Abstract interface that ALL detector implementations must follow.

ARCHITECTURAL PRINCIPLE:
  The pipeline NEVER directly calls a specific detector class.
  It always calls BaseDetector.analyze(image).

  This means MockDetector (now) and Member2Detector (later) are
  interchangeable without modifying any pipeline code.

  To integrate Member 2's real model:
    1. Create Member2Detector(BaseDetector) in this package
    2. Implement analyze()
    3. Pass it to FrameAnalyzer instead of MockDetector
    ← That's it. Zero pipeline rewrites.

MODEL INTEGRATION CONTRACT (request this info from Member 2 / Tejas):
  - Model name and version
  - Input image size (width × height)
  - Color format (BGR/RGB)
  - Normalization (mean/std or custom)
  - Face crop required or full frame
  - Output format (float score / logit / class label)
  - Score semantics: does higher score = more likely fake?
  - Threshold and direction
  - Calibration status
  - Hardware requirements (CPU/GPU)
"""

from __future__ import annotations

import abc

import numpy as np

from src.schemas.result_schema import DetectorOutput


class BaseDetector(abc.ABC):
    """
    Abstract base class for all deepfake detectors.

    Any concrete detector (mock, real, future) must implement analyze().
    The pipeline only ever calls this interface.
    """

    @abc.abstractmethod
    def analyze(self, image: np.ndarray) -> DetectorOutput:
        """
        Analyze an image (full frame or face crop) and return a structured result.

        Args:
            image: BGR numpy array (OpenCV format).
                   Do NOT assume a specific size here — each implementation
                   handles its own resizing/preprocessing.

        Returns:
            DetectorOutput with model_score, model_name, model_version,
            score_type, and is_mock flag.

        Raises:
            DetectorError: If inference fails in a structured way.
            Exception: Other unexpected errors are caught by FrameAnalyzer.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Name of this detector/model."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def model_version(self) -> str:
        """Version of this detector/model."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_mock(self) -> bool:
        """True if this is a mock/development detector, False if real."""
        raise NotImplementedError


# ─── Structured error ─────────────────────────────────────────────────────────

class DetectorError(Exception):
    """
    Raised when a detector encounters a structured, expected failure.
    The pipeline catches this and marks the frame as 'failed'.
    """

    def __init__(self, message: str, error_code: str = "DETECTOR_ERROR"):
        super().__init__(message)
        self.error_code = error_code
