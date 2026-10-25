"""
Tests for src/analysis/evidence.py

Contract:
  - primary_evidence = top_frames[0] (or None if empty).
  - Counts/ratio pass through untouched.
  - top_frames list is passed by identity (not copied).
"""

import pytest

from src.analysis.evidence import generate_evidence
from src.schemas.result_schema import (
    AggregationResult,
    EvidenceResult,
    FrameResult,
    FRAME_STATUS_SCORED,
)


def make_scored(idx, ts, score):
    return FrameResult(
        frame_index=idx, timestamp_seconds=ts,
        status=FRAME_STATUS_SCORED, model_score=score,
        score_type="probability",
        model_name="FakeDetector", model_version="t", is_mock=True,
    )


def make_agg(top_frames, *, above=0, ratio=None, unscorable=0):
    return AggregationResult(
        mean_score=0.5, max_score=0.9,
        scored_frames=len(top_frames),
        unscorable_frames=unscorable, failed_frames=0,
        frames_above_threshold=above,
        high_score_ratio=ratio,
        highest_score_timestamp=top_frames[0].timestamp_seconds if top_frames else None,
        top_frames=top_frames,
    )


def test_primary_is_first_top_frame():
    frames = [make_scored(0, 0.0, 0.95), make_scored(1, 0.1, 0.7)]
    ev = generate_evidence(make_agg(frames, above=1, ratio=0.5))
    assert isinstance(ev, EvidenceResult)
    assert ev.primary_evidence is frames[0]


def test_empty_top_frames_no_primary():
    ev = generate_evidence(make_agg([], above=0, ratio=None, unscorable=5))
    assert ev.primary_evidence is None
    assert ev.top_frames == []


def test_counts_and_ratio_passthrough():
    frames = [make_scored(0, 0.0, 0.9)]
    ev = generate_evidence(make_agg(frames, above=3, ratio=0.6, unscorable=4))
    assert ev.frames_above_threshold == 3
    assert ev.high_score_ratio == pytest.approx(0.6)
    assert ev.unscorable_frames == 4


def test_ratio_none_preserved():
    ev = generate_evidence(make_agg([], above=0, ratio=None))
    assert ev.high_score_ratio is None


def test_top_frames_by_identity():
    frames = [make_scored(0, 0.0, 0.9), make_scored(1, 0.1, 0.6)]
    ev = generate_evidence(make_agg(frames))
    assert ev.top_frames is frames
    assert ev.top_frames[0] is frames[0]