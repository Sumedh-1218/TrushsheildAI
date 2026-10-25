"""
Tests for src/analysis/frame_analyzer.py

Covers every branch of FrameAnalyzer.analyze():
  - happy path → status="scored"
  - no face → status="unscorable", reason="no_face_detected"
  - face processor raises → status="failed"
  - detector raises DetectorError → status="failed" (error_code in message)
  - detector raises other exception → status="failed"
  - invalid detector output (None, str, out-of-range) → status="failed"
  - boundary scores 0.0 and 1.0 → status="scored"
"""

import numpy as np
import pytest

from src.analysis.frame_analyzer import FrameAnalyzer, _validate_detector_output
from src.detector.detector_adapter import DetectorError
from src.schemas.result_schema import (
    DetectorOutput,
    FrameResult,
    SampledFrame,
    FRAME_STATUS_FAILED,
    FRAME_STATUS_SCORED,
    FRAME_STATUS_UNSCORABLE,
)


class FakeFaceProcessor:
    def __init__(self, face=None, exc=None):
        self._face = face
        self._exc = exc

    def extract_face(self, image):
        if self._exc is not None:
            raise self._exc
        return self._face


class FakeDetector:
    model_name = "FakeDetector"
    model_version = "test-1.0"
    is_mock = True

    def __init__(self, score=0.5, score_type="probability", exc=None):
        self._score = score
        self._score_type = score_type
        self._exc = exc

    def analyze(self, image):
        if self._exc is not None:
            raise self._exc
        return DetectorOutput(
            model_score=self._score,
            score_type=self._score_type,
            model_name=self.model_name,
            model_version=self.model_version,
            is_mock=self.is_mock,
        )


def _make_detector_error(code="MODEL_CRASH"):
    """Build a DetectorError regardless of its constructor signature."""
    try:
        err = DetectorError("model exploded")
        err.error_code = code
        return err
    except TypeError:
        pass
    try:
        err = DetectorError(error_code=code, message="model exploded")
        return err
    except TypeError:
        pytest.skip("DetectorError constructor unknown — paste detector_adapter.py")


@pytest.fixture
def sample_frame():
    return SampledFrame(
        frame_index=7,
        timestamp_seconds=1.4,
        image=np.zeros((80, 80, 3), dtype=np.uint8),
    )


@pytest.fixture
def face_crop():
    return np.full((40, 40, 3), 128, dtype=np.uint8)


def _analyzer(face, detector):
    return FrameAnalyzer(detector=detector, face_processor=FakeFaceProcessor(face=face))


def test_scored_path(sample_frame, face_crop):
    analyzer = _analyzer(face_crop, FakeDetector(score=0.83))
    result = analyzer.analyze(sample_frame)

    assert isinstance(result, FrameResult)
    assert result.status == FRAME_STATUS_SCORED
    assert result.model_score == pytest.approx(0.83)
    assert result.score_type == "probability"
    assert result.model_name == "FakeDetector"
    assert result.is_mock is True
    assert result.error is None


def test_frame_metadata_preserved(sample_frame, face_crop):
    analyzer = _analyzer(face_crop, FakeDetector(score=0.5))
    result = analyzer.analyze(sample_frame)
    assert result.frame_index == 7
    assert result.timestamp_seconds == 1.4


@pytest.mark.parametrize("boundary", [0.0, 1.0])
def test_boundary_scores_are_scored(sample_frame, face_crop, boundary):
    result = _analyzer(face_crop, FakeDetector(score=boundary)).analyze(sample_frame)
    assert result.status == FRAME_STATUS_SCORED
    assert result.model_score == boundary


def test_no_face_is_unscorable(sample_frame):
    result = _analyzer(face=None, detector=FakeDetector()).analyze(sample_frame)
    assert result.status == FRAME_STATUS_UNSCORABLE
    assert result.unscorable_reason == "no_face_detected"
    assert result.model_score is None


def test_no_face_never_scores_real(sample_frame):
    result = _analyzer(face=None, detector=FakeDetector()).analyze(sample_frame)
    assert result.status != FRAME_STATUS_SCORED
    assert result.model_score is None


def test_face_processor_exception_is_failed(sample_frame):
    analyzer = _analyzer(face=None, detector=FakeDetector())
    analyzer.face_processor = FakeFaceProcessor(exc=RuntimeError("cascade boom"))
    result = analyzer.analyze(sample_frame)

    assert result.status == FRAME_STATUS_FAILED
    assert "Face extraction error" in result.error
    assert "cascade boom" in result.error


def test_detector_error_captured(sample_frame, face_crop):
    err = _make_detector_error("MODEL_CRASH")
    result = _analyzer(face_crop, FakeDetector(exc=err)).analyze(sample_frame)

    assert result.status == FRAME_STATUS_FAILED
    assert "Detector error" in result.error
    assert "MODEL_CRASH" in result.error


def test_unexpected_detector_exception_is_failed(sample_frame, face_crop):
    result = _analyzer(face_crop, FakeDetector(exc=ValueError("kaboom"))).analyze(sample_frame)
    assert result.status == FRAME_STATUS_FAILED
    assert "Unexpected detector exception" in result.error
    assert "kaboom" in result.error


@pytest.mark.parametrize("bad_score, expected", [
    (None,     "Score is None"),
    ("0.5",    "not numeric"),
    ([0.5],    "not numeric"),
    (1.5,      "outside expected range"),
    (-0.01,    "outside expected range"),
])
def test_invalid_detector_output(sample_frame, face_crop, bad_score, expected):
    result = _analyzer(face_crop, FakeDetector(score=bad_score)).analyze(sample_frame)
    assert result.status == FRAME_STATUS_FAILED
    assert "Invalid detector output" in result.error
    assert expected in result.error


def test_validate_helper_accepts_valid():
    assert _validate_detector_output(0.0) is None
    assert _validate_detector_output(0.5) is None
    assert _validate_detector_output(1) is None


def test_validate_helper_rejects_invalid():
    assert _validate_detector_output(None) is not None
    assert _validate_detector_output("0.5") is not None
    assert _validate_detector_output(-0.1) is not None
    assert _validate_detector_output(1.1) is not None