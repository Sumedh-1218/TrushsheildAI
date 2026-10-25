"""
TrustShield AI — Member 3 — Mock Detector
src/detector/mock_detector.py

⚠️  MOCK / DEVELOPMENT ONLY — NOT A REAL DEEPFAKE DETECTOR ⚠️

Purpose:
  - Allows the full pipeline to run and be tested before Member 2's real model.
  - Produces deterministic scores so tests can make reliable assertions.
  - Clearly labelled as MOCK throughout all outputs.

Score behavior:
  Score is derived from a hash of the frame's pixel data.
  This means:
    - Same frame → always same score (deterministic)
    - Different frames → meaningfully different scores
    - Score is in [0.0, 1.0]
    - Score carries no real deepfake detection meaning

NEVER deploy this as a real detector. The is_mock flag prevents misuse.
"""

from __future__ import annotations

import hashlib

import numpy as np

from src.detector.detector_adapter import BaseDetector, DetectorError
from src.schemas.result_schema import DetectorOutput


class MockDetector(BaseDetector):
    """
    ⚠️  MOCK DETECTOR — FOR DEVELOPMENT AND TESTING ONLY ⚠️

    Generates a deterministic score based on the hash of the input image.
    This allows the pipeline to run end-to-end without a real model.

    The score has NO deepfake detection validity.
    All outputs from this detector carry is_mock=True.
    """

    MODEL_NAME = "MockDetector"
    MODEL_VERSION = "dev-1.0-MOCK"

    def analyze(self, image: np.ndarray) -> DetectorOutput:
        """
        Compute a deterministic score from the image content.

        Algorithm:
          1. Take the first 4096 bytes of the image buffer (fast)
          2. SHA-256 hash → 64-character hex string
          3. Take first 8 hex chars → integer → divide by 0xFFFFFFFF
          Result: float in [0.0, 1.0], deterministic per unique image content.

        Args:
            image: BGR numpy array from OpenCV.

        Returns:
            DetectorOutput with deterministic model_score and is_mock=True.
        """
        if image is None or image.size == 0:
            raise DetectorError(
                "MockDetector received empty image.",
                error_code="EMPTY_INPUT"
            )

        # Hash first 4096 bytes of the array's raw bytes for speed
        raw_bytes = image.tobytes()[:4096]
        sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
        # Take first 8 hex chars → integer → normalize to [0.0, 1.0]
        score_int = int(sha256_hex[:8], 16)
        model_score = score_int / 0xFFFFFFFF

        return DetectorOutput(
            model_score=round(model_score, 6),
            model_name=self.MODEL_NAME,
            model_version=self.MODEL_VERSION,
            score_type="model_score",
            is_mock=True,
        )

    @property
    def model_name(self) -> str:
        return self.MODEL_NAME

    @property
    def model_version(self) -> str:
        return self.MODEL_VERSION

    @property
    def is_mock(self) -> bool:
        return True
