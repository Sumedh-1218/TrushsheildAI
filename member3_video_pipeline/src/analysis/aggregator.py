"""
TrustShield AI — Member 3 — Aggregator
src/analysis/aggregator.py

Aggregates a list of FrameResult objects into a single video-level
analytical summary.

DESIGN NOTE:
  Aggregation is independent from video reading. It only cares about
  the list of FrameResult objects — this makes it independently testable.

RESPONSIBLE AI NOTE:
  The aggregate score is named 'aggregate_model_score', NOT 'probability'.
  An uncalibrated model output is not a probability.
"""

from __future__ import annotations

from typing import List, Optional

from src.config import PipelineConfig, DEFAULT_CONFIG
from src.schemas.result_schema import (
    AggregationResult,
    FrameResult,
    FRAME_STATUS_SCORED,
    FRAME_STATUS_UNSCORABLE,
    FRAME_STATUS_FAILED,
)


def aggregate_frame_results(
    frame_results: List[FrameResult],
    config: PipelineConfig = DEFAULT_CONFIG,
) -> AggregationResult:
    """
    Compute a video-level analytical summary from all frame results.

    Args:
        frame_results: List of FrameResult from FrameAnalyzer.
        config: Pipeline configuration (threshold, top-K).

    Returns:
        AggregationResult with mean/max scores, counts, top-K frames, etc.
    """
    scored = [f for f in frame_results if f.status == FRAME_STATUS_SCORED]
    unscorable = [f for f in frame_results if f.status == FRAME_STATUS_UNSCORABLE]
    failed = [f for f in frame_results if f.status == FRAME_STATUS_FAILED]

    if not scored:
        # No frames were scored — return empty aggregate
        return AggregationResult(
            mean_score=None,
            max_score=None,
            scored_frames=0,
            unscorable_frames=len(unscorable),
            failed_frames=len(failed),
            frames_above_threshold=0,
            high_score_ratio=None,
            highest_score_timestamp=None,
            top_frames=[],
        )

    scores = [f.model_score for f in scored]  # all are non-None here

    mean_score = sum(scores) / len(scores)
    max_score = max(scores)

    above_threshold = [
        f for f in scored
        if f.model_score is not None and f.model_score > config.DETECTION_THRESHOLD
    ]
    frames_above_threshold = len(above_threshold)
    high_score_ratio = frames_above_threshold / len(scored) if scored else None

    # Frame with highest score
    best_frame = max(scored, key=lambda f: f.model_score or 0.0)
    highest_score_timestamp = best_frame.timestamp_seconds

    # Top-K frames sorted by score descending
    sorted_scored = sorted(scored, key=lambda f: f.model_score or 0.0, reverse=True)
    top_frames = sorted_scored[:config.TOP_K_EVIDENCE]

    return AggregationResult(
        mean_score=round(mean_score, 6),
        max_score=round(max_score, 6),
        scored_frames=len(scored),
        unscorable_frames=len(unscorable),
        failed_frames=len(failed),
        frames_above_threshold=frames_above_threshold,
        high_score_ratio=round(high_score_ratio, 6) if high_score_ratio is not None else None,
        highest_score_timestamp=highest_score_timestamp,
        top_frames=top_frames,
    )
