"""
Tests for src/detector/mock_detector.py

Contract:
  - analyze(image) returns DetectorOutput
  - Same image → same score (deterministic)
  - Score in [0.0, 1.0]
  - is_mock=True, model_name contains "mock"
"""

import numpy as np
import pytest

_D_ERROR = None
try:
    from src.detector.mock_detector import MockDetector
    from src.schemas.result_schema import DetectorOutput
    _OK = True
except Exception as exc:
    _OK = False
    _D_ERROR = exc

pytestmark = pytest.mark.skipif(
    not _OK,
    reason=f"Cannot import MockDetector ({_D_ERROR}). Paste src/detector/mock_detector.py to fix.",
)


def test_detector_exposes_identity(detector):
    assert hasattr(detector, "model_name")
    assert hasattr(detector, "model_version")
    assert hasattr(detector, "is_mock")


def test_detector_flags_itself_as_mock(detector):
    assert detector.is_mock is True
    assert "mock" in detector.model_name.lower()


def test_score_in_unit_range(detector, noise_image):
    out = detector.analyze(noise_image)
    assert 0.0 <= out.model_score <= 1.0


def test_deterministic_across_calls(detector, noise_image):
    a = detector.analyze(noise_image).model_score
    b = detector.analyze(noise_image.copy()).model_score
    assert a == b


def test_different_images_give_different_scores(detector, blank_image, noise_image):
    a = detector.analyze(blank_image).model_score
    b = detector.analyze(noise_image).model_score
    assert a != b


def test_returns_detector_output(detector, noise_image):
    out = detector.analyze(noise_image)
    assert isinstance(out, DetectorOutput)


def test_output_preserves_mock_identity(detector, noise_image):
    out = detector.analyze(noise_image)
    assert out.is_mock is True
    assert out.model_name == detector.model_name
    assert out.model_version == detector.model_version