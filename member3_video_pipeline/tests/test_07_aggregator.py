"""
Tests for src/analysis/aggregator.py

Contract:
  - Only "scored" frames contribute to mean/max/ratio.
  - Zero scored frames → all numeric fields None/0, top_frames=[].
  - Threshold is strict (>), not >=.
  - top_frames sorted desc, respects TOP_K_EVIDENCE.
  - mean_score rounded to 6 decimals.
"""

import pytest

from src.analysis.aggregator import aggregate_frame_results
from src.schemas.result_schema import (
    AggregationResult,
    FrameResult,
    FRAME_STATUS_FAILED,
    FRAME_STATUS_SCORED,
    FRAME_STATUS_UNSCORABLE,
)


class SimpleConfig:
    def __init__(self, threshold=0.5, top_k=3):
        self.DETECTION_THRESHOLD = threshold
        self.TOP_K_EVIDENCE = top_k


def cfg(threshold=0.5, top_k=3):
    return SimpleConfig(threshold, top_k)


def scored(idx, ts, score):
    return FrameResult(
        frame_index=idx, timestamp_seconds=ts,
        status=FRAME_STATUS_SCORED, model_score=score,
        score_type="probability",
        model_name="FakeDetector", model_version="t", is_mock=True,
    )


def unscorable(idx, ts):
    return FrameResult(
        frame_index=idx, timestamp_seconds=ts,
        status=FRAME_STATUS_UNSCORABLE,
        unscorable_reason="no_face_detected",
        model_name="FakeDetector", model_version="t", is_mock=True,
    )


def failed(idx, ts):
    return FrameResult(
        frame_index=idx, timestamp_seconds=ts,
        status=FRAME_STATUS_FAILED, error="boom",
        model_name="FakeDetector", model_version="t", is_mock=True,
    )


def test_empty_list_returns_empty_aggregate():
    result = aggregate_frame_results([], config=cfg())
    assert isinstance(result, AggregationResult)
    assert result.mean_score is None
    assert result.max_score is None
    assert result.scored_frames == 0
    assert result.top_frames == []
    assert result.high_score_ratio is None


def test_all_unscorable():
    frames = [unscorable(i, i * 0.1) for i in range(4)]
    result = aggregate_frame_results(frames, config=cfg())
    assert result.scored_frames == 0
    assert result.unscorable_frames == 4
    assert result.failed_frames == 0


def test_all_failed():
    frames = [failed(i, i * 0.1) for i in range(3)]
    result = aggregate_frame_results(frames, config=cfg())
    assert result.scored_frames == 0
    assert result.failed_frames == 3


def test_single_scored_frame_mean_equals_max():
    result = aggregate_frame_results([scored(0, 0.0, 0.9)], config=cfg())
    assert result.mean_score == pytest.approx(0.9)
    assert result.max_score == pytest.approx(0.9)


def test_mean_and_max():
    frames = [scored(0, 0.0, 0.2), scored(1, 0.1, 0.4), scored(2, 0.2, 0.9)]
    result = aggregate_frame_results(frames, config=cfg())
    assert result.mean_score == pytest.approx(0.5, rel=1e-6)
    assert result.max_score == pytest.approx(0.9)


def test_score_at_threshold_not_counted():
    result = aggregate_frame_results([scored(0, 0.0, 0.5)], config=cfg(threshold=0.5))
    assert result.frames_above_threshold == 0
    assert result.high_score_ratio == pytest.approx(0.0)


def test_score_just_above_threshold_counted():
    result = aggregate_frame_results([scored(0, 0.0, 0.5001)], config=cfg(threshold=0.5))
    assert result.frames_above_threshold == 1


def test_high_score_ratio_mixed():
    frames = [
        scored(0, 0.0, 0.9), scored(1, 0.1, 0.7),
        scored(2, 0.2, 0.3), scored(3, 0.3, 0.1),
    ]
    result = aggregate_frame_results(frames, config=cfg(threshold=0.5))
    assert result.frames_above_threshold == 2
    assert result.high_score_ratio == pytest.approx(0.5)


def test_top_frames_sorted_desc():
    frames = [scored(0, 0.0, 0.1), scored(1, 0.1, 0.95), scored(2, 0.2, 0.5)]
    result = aggregate_frame_results(frames, config=cfg(top_k=3))
    scores = [f.model_score for f in result.top_frames]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits():
    frames = [scored(i, i * 0.1, 0.1 * (i + 1)) for i in range(10)]
    result = aggregate_frame_results(frames, config=cfg(top_k=3))
    assert len(result.top_frames) == 3


def test_top_k_larger_than_count():
    frames = [scored(0, 0.0, 0.2), scored(1, 0.1, 0.6)]
    result = aggregate_frame_results(frames, config=cfg(top_k=10))
    assert len(result.top_frames) == 2


def test_highest_timestamp_matches_top_frame():
    frames = [scored(0, 0.0, 0.2), scored(1, 0.1, 0.99), scored(2, 0.2, 0.3)]
    result = aggregate_frame_results(frames, config=cfg(top_k=1))
    assert result.top_frames[0].model_score == pytest.approx(0.99)
    assert result.highest_score_timestamp == result.top_frames[0].timestamp_seconds


def test_mean_rounded_to_six_decimals():
    frames = [scored(0, 0.0, 1.0), scored(1, 0.1, 0.0), scored(2, 0.2, 0.0)]
    result = aggregate_frame_results(frames, config=cfg())
    assert result.mean_score == 0.333333