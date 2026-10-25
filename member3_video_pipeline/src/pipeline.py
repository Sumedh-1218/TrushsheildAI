"""
TrustShield AI — Member 3 — Main Pipeline
src/pipeline.py

High-level entry point for the deepfake video analysis pipeline.

PUBLIC API (for Member 1 / Anushka):
    from src.pipeline import analyze_video
    result = analyze_video("path/to/video.mp4")
    # result is a dict — fully JSON serializable

Also provides a CLI:
    python -m src.pipeline path/to/video.mp4
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

from src.analysis.aggregator import aggregate_frame_results
from src.analysis.evidence import generate_evidence
from src.analysis.frame_analyzer import FrameAnalyzer
from src.config import PipelineConfig, DEFAULT_CONFIG
from src.detector.detector_adapter import BaseDetector
from src.detector.mock_detector import MockDetector
from src.face.face_processor import BaseFaceProcessor, get_default_face_processor
from src.schemas.result_schema import (
    LABEL_FLAGGED,
    LABEL_SUSPICIOUS,
    LABEL_INCONCLUSIVE,
    LABEL_ERROR,
    LIMITATIONS,
    VideoAnalysisResult,
)
from src.validation.video_validator import validate_video
from src.video.frame_sampler import FrameSampler
from src.video.video_reader import extract_metadata


# ─── Main API function ────────────────────────────────────────────────────────

def analyze_video(
    video_path: str,
    config: PipelineConfig = DEFAULT_CONFIG,
    detector: Optional[BaseDetector] = None,
    face_processor: Optional[BaseFaceProcessor] = None,
) -> dict:
    """
    Analyze a video for deepfake indicators.
    """
    start_time = time.time()

    # ── Default dependencies ───────────────────────────────────────────────────
    if detector is None:
        detector = MockDetector()
    if face_processor is None:
        face_processor = get_default_face_processor(config)

    # ── Step 1: Validation ────────────────────────────────────────────────────
    validation = validate_video(video_path, config)
    if not validation.is_valid:
        return VideoAnalysisResult(
            status="failed",
            label=LABEL_ERROR,
            error=validation.error,
            limitations=LIMITATIONS,
            runtime_seconds=time.time() - start_time,
        ).to_dict()

    # ── Step 2: Metadata ──────────────────────────────────────────────────────
    try:
        metadata = extract_metadata(video_path, config)
    except Exception as exc:
        return VideoAnalysisResult(
            status="failed",
            label=LABEL_ERROR,
            error=f"Metadata extraction failed: {exc}",
            limitations=LIMITATIONS,
            runtime_seconds=time.time() - start_time,
        ).to_dict()

    # ── Step 3: Frame sampling + analysis ─────────────────────────────────────
    sampler = FrameSampler(config)
    analyzer = FrameAnalyzer(detector, face_processor, config)
    frame_results = []

    try:
        for sampled_frame in sampler.sample(video_path, metadata.fps):
            result = analyzer.analyze(sampled_frame)
            frame_results.append(result)
    except Exception as exc:
        return VideoAnalysisResult(
            status="failed",
            label=LABEL_ERROR,
            error=f"Frame processing error: {exc}",
            video=metadata,
            limitations=LIMITATIONS,
            runtime_seconds=time.time() - start_time,
        ).to_dict()

    if not frame_results:
        return VideoAnalysisResult(
            status="failed",
            label=LABEL_ERROR,
            error="No frames were sampled from the video.",
            video=metadata,
            limitations=LIMITATIONS,
            runtime_seconds=time.time() - start_time,
        ).to_dict()

    # ── Step 4: Aggregation ───────────────────────────────────────────────────
    aggregation = aggregate_frame_results(frame_results, config)

    # ── Step 5: Evidence ──────────────────────────────────────────────────────
    evidence = generate_evidence(aggregation, config)

    # ── Step 6: Label ─────────────────────────────────────────────────────────
    label = _determine_label(aggregation.mean_score, aggregation.high_score_ratio, config)

    # ── Step 7: Build final result ────────────────────────────────────────────
    limitations = list(LIMITATIONS)
    if detector.is_mock:
        limitations.insert(0, "⚠️  MOCK DETECTOR IN USE — results have no real detection validity.")

    runtime = time.time() - start_time

    result = VideoAnalysisResult(
        status="completed",
        label=label,
        score=aggregation.mean_score,
        score_type=config.AGGREGATE_SCORE_TYPE,
        video=metadata,
        frames_analyzed=aggregation.scored_frames,
        frames_unscorable=aggregation.unscorable_frames,
        frames_failed=aggregation.failed_frames,
        aggregation=aggregation,
        evidence=evidence,
        model_name=detector.model_name,
        model_version=detector.model_version,
        is_mock=detector.is_mock,
        limitations=limitations,
        runtime_seconds=runtime,
    )

    return result.to_dict()


# ─── Label logic ──────────────────────────────────────────────────────────────

def _determine_label(
    mean_score: Optional[float],
    high_score_ratio: Optional[float],
    config: PipelineConfig,
) -> str:
    """
    Determine the video label from the aggregate score.
    """
    if mean_score is None:
        return LABEL_INCONCLUSIVE

    ratio = high_score_ratio or 0.0

    if mean_score >= 0.65 and ratio >= 0.3:
        return LABEL_FLAGGED
    elif mean_score >= 0.45 or ratio > 0.0:
        return LABEL_SUSPICIOUS
    else:
        return LABEL_INCONCLUSIVE


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _format_timestamp(seconds: Optional[float]) -> str:
    """Format seconds as MM:SS.s"""
    if seconds is None:
        return "N/A"
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:04.1f}"


def _print_result(result: dict) -> None:
    """Print a human-readable CLI report."""
    status = result.get("status", "unknown")
    video = result.get("video") or {}
    agg = result.get("aggregation") or {}
    model = result.get("model") or {}

    separator = "─" * 42
    print(f"\n{'═' * 42}")
    print("  TrustShield AI — Deepfake Video Analysis")
    print(f"{'═' * 42}")

    if status == "failed":
        print(f"  STATUS  : ❌ FAILED")
        print(f"  ERROR   : {result.get('error', 'Unknown error')}")
        print(f"{'═' * 42}\n")
        return

    print(f"  STATUS  : ✅ Completed")
    print(separator)
    print(f"  Duration        : {video.get('duration_seconds', 'N/A')} sec")
    print(f"  FPS             : {video.get('fps', 'N/A')}")
    print(f"  Resolution      : {video.get('width', '?')}×{video.get('height', '?')}")
    print(f"  Codec           : {video.get('codec', 'N/A')}")
    print(separator)
    print(f"  Frames sampled  : {agg.get('scored_frames', 0) + agg.get('unscorable_frames', 0) + agg.get('failed_frames', 0)}")
    print(f"  Frames scored   : {agg.get('scored_frames', 0)}")
    print(f"  Frames unscorable: {agg.get('unscorable_frames', 0)}")
    print(f"  Frames failed   : {agg.get('failed_frames', 0)}")
    print(separator)

    mean = result.get("score")
    mx = agg.get("max_score")
    ts = agg.get("highest_score_timestamp")

    print(f"  Mean score      : {f'{mean:.4f}' if mean is not None else 'N/A'}")
    print(f"  Maximum score   : {f'{mx:.4f}' if mx is not None else 'N/A'}")
    print(f"  Highest-risk at : {_format_timestamp(ts)}")
    print(f"  Frames > thresh : {agg.get('frames_above_threshold', 0)}")
    
    # 🛠️ FIXED THIS LINE
    ratio_val = agg.get('high_score_ratio')
    ratio_str = f"{ratio_val:.1%}" if ratio_val is not None else 'N/A'
    print(f"  High-score ratio: {ratio_str}")
    
    print(separator)

    label = result.get("label", "unknown").upper().replace("_", " ")
    print(f"  ⚠️  LABEL        : {label}")
    print(f"  Runtime         : {result.get('runtime_seconds', 0):.2f} sec")
    print(separator)
    is_mock = model.get("is_mock", True)
    mock_label = "⚠️  MOCK / DEVELOPMENT ONLY" if is_mock else "✅ Real model"
    print(f"  Detector        : {model.get('name', 'N/A')} v{model.get('version', 'N/A')}")
    print(f"  Detector status : {mock_label}")
    print(f"{'═' * 42}")

    # Limitations
    limitations = result.get("limitations", [])
    if limitations:
        print("\n  ⚠️  Limitations:")
        for lim in limitations:
            print(f"    • {lim}")
    print()


def main() -> None:
    """CLI entry point: python -m src.pipeline <video_path>"""
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline <video_path>")
        print("Example: python -m src.pipeline sample.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    print(f"Analyzing: {video_path}")

    result = analyze_video(video_path)
    _print_result(result)

    # JSON output flag
    if "--json" in sys.argv:
        print(json.dumps(result, indent=2))

    # Exit code: 0 = completed, 1 = failed
    sys.exit(0 if result.get("status") == "completed" else 1)


if __name__ == "__main__":
    main()