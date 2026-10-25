"""
TrustShield AI — Member 3 — Evaluator
src/evaluation/evaluator.py

Runs the pipeline against labelled test videos and computes metrics.

Directory structure for labelled data:
    tests/data/real/          ← videos known to be unmanipulated
    tests/data/manipulated/   ← videos known to be deepfakes

CRITICAL RULE: Never fabricate metrics.
If the data directories are empty, report exactly that.
"""

from __future__ import annotations

import glob
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import PipelineConfig, DEFAULT_CONFIG
from src.evaluation.metrics import EvaluationMetrics, compute_metrics
from src.pipeline import analyze_video


SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
PENDING_NOTE = (
    "Metric execution pending labelled test data. "
    "Add real videos to tests/data/real/ and deepfake videos to "
    "tests/data/manipulated/ then re-run the evaluator."
)


class VideoEvaluator:
    """
    Evaluates the pipeline against labelled video samples.

    Usage:
        evaluator = VideoEvaluator(real_dir, manipulated_dir, config)
        metrics = evaluator.evaluate()
        print(metrics.to_dict())
    """

    def __init__(
        self,
        real_dir: str,
        manipulated_dir: str,
        config: PipelineConfig = DEFAULT_CONFIG,
    ):
        self.real_dir = real_dir
        self.manipulated_dir = manipulated_dir
        self.config = config

    def _collect_videos(self, directory: str, label: str) -> List[Tuple[str, str]]:
        """Return list of (video_path, label) tuples from a directory."""
        videos = []
        for ext in SUPPORTED_EXTS:
            pattern = os.path.join(directory, f"*{ext}")
            for path in glob.glob(pattern):
                videos.append((path, label))
        return videos

    def evaluate(self) -> EvaluationMetrics:
        """
        Run the pipeline on all labelled videos and compute metrics.

        Returns:
            EvaluationMetrics (with evaluation_note if data is missing).
        """
        real_videos = self._collect_videos(self.real_dir, "real")
        manipulated_videos = self._collect_videos(self.manipulated_dir, "manipulated")
        all_videos = real_videos + manipulated_videos

        if not all_videos:
            return EvaluationMetrics(
                true_positives=0,
                true_negatives=0,
                false_positives=0,
                false_negatives=0,
                total_samples=0,
                precision=None,
                recall=None,
                f1_score=None,
                false_positive_rate=None,
                mean_runtime_seconds=None,
                evaluation_note=PENDING_NOTE,
            )

        predictions: List[str] = []
        ground_truth: List[str] = []
        runtimes: List[float] = []

        for video_path, label in all_videos:
            t0 = time.time()
            result = analyze_video(video_path, self.config)
            elapsed = time.time() - t0

            pred_label = result.get("label", "error")
            predictions.append(pred_label)
            ground_truth.append(label)
            runtimes.append(elapsed)

        metrics = compute_metrics(predictions, ground_truth, runtimes)
        metrics.evaluation_note = (
            f"Evaluated on {len(all_videos)} videos "
            f"({len(real_videos)} real, {len(manipulated_videos)} manipulated)."
        )
        return metrics


# ─── Convenience function ─────────────────────────────────────────────────────

def run_evaluation(
    real_dir: str,
    manipulated_dir: str,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> EvaluationMetrics:
    """Run a full evaluation against labelled video directories."""
    return VideoEvaluator(real_dir, manipulated_dir, config).evaluate()
