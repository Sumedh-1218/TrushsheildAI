"""
TrustShield AI — Member 3 — Evaluation Metrics
src/evaluation/metrics.py

Computes standard classification metrics from labelled test results.

CRITICAL RULE: NEVER fabricate metrics.
If labelled data is insufficient, this module reports that honestly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EvaluationMetrics:
    """
    Standard binary classification metrics for deepfake detection evaluation.

    Terminology:
      Positive  = the video IS a deepfake (label='manipulated')
      Negative  = the video is NOT a deepfake (label='real')
      TP = correctly flagged deepfake
      TN = correctly passed real video
      FP = real video incorrectly flagged (false alarm)
      FN = deepfake not flagged (missed detection)
    """
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    total_samples: int
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    false_positive_rate: Optional[float]
    mean_runtime_seconds: Optional[float]
    evaluation_note: str = ""

    def to_dict(self) -> dict:
        return {
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "total_samples": self.total_samples,
            "precision": round(self.precision, 4) if self.precision is not None else None,
            "recall": round(self.recall, 4) if self.recall is not None else None,
            "f1_score": round(self.f1_score, 4) if self.f1_score is not None else None,
            "false_positive_rate": round(self.false_positive_rate, 4) if self.false_positive_rate is not None else None,
            "mean_runtime_seconds": round(self.mean_runtime_seconds, 4) if self.mean_runtime_seconds is not None else None,
            "evaluation_note": self.evaluation_note,
        }


def compute_metrics(
    predictions: List[str],
    ground_truth: List[str],
    runtimes: Optional[List[float]] = None,
    positive_label: str = "manipulated",
) -> EvaluationMetrics:
    """
    Compute TP/TN/FP/FN and derived metrics from lists of predictions and labels.

    Args:
        predictions:   List of pipeline labels for each video
                       (e.g. 'flagged_for_review', 'suspicious', 'inconclusive').
        ground_truth:  List of true labels (e.g. 'manipulated', 'real').
        runtimes:      Optional list of per-video runtime in seconds.
        positive_label: The ground-truth label meaning "is a deepfake".

    Returns:
        EvaluationMetrics dataclass.

    NOTE: 'flagged_for_review' and 'suspicious' are treated as positive predictions.
          'inconclusive' and 'error' are treated as negative predictions.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions length {len(predictions)} != ground_truth length {len(ground_truth)}"
        )

    # Map pipeline labels to binary: flagged/suspicious = "positive"
    POSITIVE_PREDICTIONS = {"flagged_for_review", "suspicious"}

    tp = tn = fp = fn = 0
    for pred, truth in zip(predictions, ground_truth):
        pred_positive = pred in POSITIVE_PREDICTIONS
        truth_positive = truth == positive_label

        if pred_positive and truth_positive:
            tp += 1
        elif not pred_positive and not truth_positive:
            tn += 1
        elif pred_positive and not truth_positive:
            fp += 1
        else:
            fn += 1

    total = len(predictions)

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else None

    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else None

    # F1 = 2 * (precision * recall) / (precision + recall)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    # False Positive Rate = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None

    mean_runtime = None
    if runtimes:
        mean_runtime = sum(runtimes) / len(runtimes)

    return EvaluationMetrics(
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        total_samples=total,
        precision=precision,
        recall=recall,
        f1_score=f1,
        false_positive_rate=fpr,
        mean_runtime_seconds=mean_runtime,
    )
