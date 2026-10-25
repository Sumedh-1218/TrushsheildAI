"""
TrustShield AI — Member 3 — Evidence Generator
src/analysis/evidence.py

Converts aggregation results into structured, human-readable evidence.
This is the data that Member 1 (Anushka) uses to explain why a video
was flagged — it will power the UI evidence panel.

Evidence is NOT the same as provenance (SHA-256, C2PA) — those are
Member 5's responsibility.
"""

from __future__ import annotations

from src.config import PipelineConfig, DEFAULT_CONFIG
from src.schemas.result_schema import (
    AggregationResult,
    EvidenceResult,
)


def generate_evidence(
    aggregation: AggregationResult,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> EvidenceResult:
    """
    Generate timestamped evidence from an aggregation result.

    Args:
        aggregation: The aggregated frame analysis result.
        config:      Pipeline configuration (not currently used here,
                     but passed for consistency and future extension).

    Returns:
        EvidenceResult with primary_evidence, top_frames, and counts.
    """
    # Primary evidence = highest-scoring frame (first of top_frames, sorted desc)
    primary = aggregation.top_frames[0] if aggregation.top_frames else None

    return EvidenceResult(
        primary_evidence=primary,
        top_frames=aggregation.top_frames,
        frames_above_threshold=aggregation.frames_above_threshold,
        high_score_ratio=aggregation.high_score_ratio,
        unscorable_frames=aggregation.unscorable_frames,
    )
