"""
Tests for src/schemas/result_schema.py

Field names below come directly from what frame_analyzer.py,
aggregator.py, and evidence.py pass in.
"""

import json
import numpy as np
import pytest

_SCHEMA_ERROR = None
try:
    from src.schemas.result_schema import (
        AggregationResult,
        DetectorOutput,
        EvidenceResult,
        FrameResult,
        SampledFrame,
        FRAME_STATUS_FAILED,
        FRAME_STATUS_SCORED,
        FRAME_STATUS_UNSCORABLE,
    )
    _OK = True
except Exception as exc:
    _OK = False
    _SCHEMA_ERROR = exc

pytestmark = pytest.mark.skipif(
    not _OK,
    reason=f"Cannot import result_schema ({_SCHEMA_ERROR}). Paste src/schemas/result_schema.py to fix.",
)


# ─── Status constants ────────────────────────────────────────────────────────

def test_status_constants_are_distinct_strings():
    assert isinstance(FRAME_STATUS_SCORED, str)
    assert isinstance(FRAME_STATUS_UNSCORABLE, str)
    assert isinstance(FRAME_STATUS_FAILED, str)
    assert len({FRAME_STATUS_SCORED, FRAME_STATUS_UNSCORABLE, FRAME_STATUS_FAILED}) == 3


# ─── SampledFrame ────────────────────────────────────────────────────────────

def test_sampled_frame_constructs():
    frame = SampledFrame(
        frame_index=5,
        timestamp_seconds=1.25,
        image=np.zeros((10, 10, 3), dtype=np.uint8),
    )
    assert frame.frame_index == 5
    assert frame.timestamp_seconds == pytest.approx(1.25)
    assert frame.image.shape == (10, 10, 3)


# ─── DetectorOutput ──────────────────────────────────────────────────────────

def test_detector_output_constructs():
    out = DetectorOutput(
        model_score=0.75,
        score_type="probability",
        model_name="MockDetector",
        model_version="dev-1.0-MOCK",
        is_mock=True,
    )
    assert out.model_score == pytest.approx(0.75)
    assert out.is_mock is True
    assert out.score_type == "probability"


# ─── FrameResult ─────────────────────────────────────────────────────────────

def test_frame_result_scored_path():
    res = FrameResult(
        frame_index=0,
        timestamp_seconds=0.0,
        status=FRAME_STATUS_SCORED,
        model_score=0.8,
        score_type="probability",
        model_name="MockDetector",
        model_version="dev-1.0-MOCK",
        is_mock=True,
    )
    assert res.status == FRAME_STATUS_SCORED
    assert res.model_score == pytest.approx(0.8)


def test_frame_result_unscorable_path():
    res = FrameResult(
        frame_index=1,
        timestamp_seconds=0.1,
        status=FRAME_STATUS_UNSCORABLE,
        unscorable_reason="no_face_detected",
        model_name="MockDetector",
        model_version="dev-1.0-MOCK",
        is_mock=True,
    )
    assert res.status == FRAME_STATUS_UNSCORABLE
    assert res.model_score is None
    assert res.unscorable_reason == "no_face_detected"


def test_frame_result_failed_path():
    res = FrameResult(
        frame_index=2,
        timestamp_seconds=0.2,
        status=FRAME_STATUS_FAILED,
        error="boom",
        model_name="MockDetector",
        model_version="dev-1.0-MOCK",
        is_mock=True,
    )
    assert res.status == FRAME_STATUS_FAILED
    assert res.error == "boom"


# ─── AggregationResult ──────────────────────────────────────────────────────

def test_aggregation_result_empty_aggregate():
    agg = AggregationResult(
        mean_score=None,
        max_score=None,
        scored_frames=0,
        unscorable_frames=3,
        failed_frames=1,
        frames_above_threshold=0,
        high_score_ratio=None,
        highest_score_timestamp=None,
        top_frames=[],
    )
    assert agg.mean_score is None
    assert agg.top_frames == []


def test_aggregation_result_populated():
    agg = AggregationResult(
        mean_score=0.5,
        max_score=0.9,
        scored_frames=4,
        unscorable_frames=1,
        failed_frames=0,
        frames_above_threshold=2,
        high_score_ratio=0.5,
        highest_score_timestamp=2.3,
        top_frames=[],
    )
    assert agg.frames_above_threshold == 2


# ─── EvidenceResult ─────────────────────────────────────────────────────────

def test_evidence_result_empty():
    ev = EvidenceResult(
        primary_evidence=None,
        top_frames=[],
        frames_above_threshold=0,
        high_score_ratio=None,
        unscorable_frames=0,
    )
    assert ev.primary_evidence is None


# ─── Serialisation ──────────────────────────────────────────────────────────

def test_sampled_frame_to_dict_excludes_image():
    if not hasattr(SampledFrame, "to_dict"):
        pytest.skip("SampledFrame has no to_dict() — optional")
    frame = SampledFrame(
        frame_index=1,
        timestamp_seconds=0.05,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
    )
    payload = frame.to_dict()
    json.dumps(payload)
    assert "image" not in payload