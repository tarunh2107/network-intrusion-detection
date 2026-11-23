"""Evaluation metrics, threshold selection, and reporting utilities for binary network intrusion detection.

Implements Phase 6A and Phase 6B requirements:
- ATTACK class probability column lookup via model.classes_ (not hardcoded index).
- Precision, Recall, F1, Accuracy, FPR, FNR, Average Precision (Scikit-learn average_precision_score).
- Constrained threshold selection under FPR <= 0.2% operating budget with deterministic tie-breaking.
- Handling of absent attack categories ("not estimable") and small-sample variance warnings.
- Streaming confusion matrix accumulation.
- Vector SVG confusion matrix visualization without third-party plotting dependencies.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# All 15 known dataset classes (1 baseline + 14 attacks)
ALL_KNOWN_DISPLAY_LABELS: List[str] = [
    "BENIGN",
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "SSH-Patator",
    "Web Attack - Brute Force",
    "Web Attack - Sql Injection",
    "Web Attack - XSS",
]

DEFAULT_OPERATING_RULE: str = (
    "Choose the threshold with highest validation ATTACK recall among thresholds "
    "with false-positive rate <= 0.2%. Break ties by higher precision, then higher threshold. "
    "(This is a project evaluation choice, not an industry standard or a guarantee about live networks.)"
)


def get_attack_class_index(classes: Union[Sequence[Any], np.ndarray]) -> int:
    """Dynamically determine the column index for ATTACK (class 1) from classifier classes.

    Args:
        classes: Sequence or ndarray of class labels (e.g., model.classes_).

    Returns:
        Zero-based column index corresponding to class label 1.

    Raises:
        ValueError: If class label 1 is not present in classes.
    """
    classes_list = list(classes)
    try:
        return classes_list.index(1)
    except ValueError:
        raise ValueError(
            f"Class label 1 (ATTACK) was not found in classifier classes: {classes_list}"
        )


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    display_labels: Optional[Union[Sequence[str], pd.Series]] = None,
    fit_time_seconds: float = 0.0,
    predict_time_seconds: float = 0.0,
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics for binary classification.

    Args:
        y_true: Ground-truth binary targets (0 for BENIGN, 1 for ATTACK).
        y_pred: Predicted binary labels (0 or 1).
        y_scores: Optional predicted probabilities/decision scores for class 1.
        display_labels: Optional sequence of multiclass display labels for granular breakdown.
        fit_time_seconds: Model fitting duration in seconds.
        predict_time_seconds: Model inference duration on this evaluation set.

    Returns:
        Dictionary containing overall metrics, confusion matrix, timing, and per-attack breakdown.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}")

    n_samples = len(y_true)
    if n_samples == 0:
        raise ValueError("Cannot evaluate empty predictions array.")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    average_precision: Optional[float] = None
    if y_scores is not None:
        try:
            scores_arr = np.asarray(y_scores, dtype=np.float64)
            if scores_arr.ndim == 2 and scores_arr.shape[1] >= 2:
                scores_arr = scores_arr[:, 1]
            if len(np.unique(y_true)) > 1:
                average_precision = float(average_precision_score(y_true, scores_arr))
        except Exception:
            average_precision = None

    latency_ms_per_1000 = float((predict_time_seconds / n_samples) * 1_000_000.0) if n_samples > 0 else 0.0

    per_class_summary: Dict[str, Dict[str, Any]] = {}
    if display_labels is not None:
        labels_seq = list(display_labels)
        if len(labels_seq) == n_samples:
            df_breakdown = pd.DataFrame({
                "label": labels_seq,
                "y_true": y_true,
                "y_pred": y_pred,
            })
            present_labels = set(df_breakdown["label"].unique())
            all_labels_to_eval = sorted(list(present_labels.union(set(ALL_KNOWN_DISPLAY_LABELS))))

            for label in all_labels_to_eval:
                is_attack = bool(label != "BENIGN")
                if label not in present_labels:
                    per_class_summary[str(label)] = {
                        "total": 0,
                        "detected_correct": 0,
                        "recall": None,
                        "recall_display": "not estimable",
                        "missed": 0,
                        "is_attack": is_attack,
                        "uncertainty_warning": "Absent class in evaluation partition",
                    }
                else:
                    group = df_breakdown[df_breakdown["label"] == label]
                    lbl_total = len(group)
                    if is_attack:
                        detected = int((group["y_pred"] == 1).sum())
                        missed = int((group["y_pred"] == 0).sum())
                    else:
                        detected = int((group["y_pred"] == 0).sum())
                        missed = int((group["y_pred"] == 1).sum())

                    recall_val = float(detected / lbl_total) if lbl_total > 0 else None
                    warning = None
                    if lbl_total in (1, 2):
                        warning = f"Warning: Based on {lbl_total} example(s); high estimation variance"

                    per_class_summary[str(label)] = {
                        "total": lbl_total,
                        "detected_correct": detected,
                        "recall": recall_val,
                        "recall_display": f"{recall_val:.2%}" if recall_val is not None else "not estimable",
                        "missed": missed,
                        "is_attack": is_attack,
                        "uncertainty_warning": warning,
                    }

    return {
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "accuracy": acc,
        "fpr": fpr,
        "fnr": fnr,
        "average_precision": average_precision,
        "pr_auc": average_precision,  # retained for backward compatibility with Phase 6A tests
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "fit_time_seconds": float(fit_time_seconds),
        "predict_time_seconds": float(predict_time_seconds),
        "latency_ms_per_1000": latency_ms_per_1000,
        "sample_counts": {
            "total": n_samples,
            "benign": int(np.sum(y_true == 0)),
            "attack": int(np.sum(y_true == 1)),
        },
        "per_class_recall": per_class_summary,
    }


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float,
    display_labels: Optional[Union[Sequence[str], pd.Series]] = None,
    predict_time_seconds: float = 0.0,
) -> Dict[str, Any]:
    """Score binary targets given predicted ATTACK probabilities and decision threshold.

    Decision Rule: Classify as ATTACK (1) if y_probs >= threshold, else BENIGN (0).
    """
    y_probs = np.asarray(y_probs, dtype=np.float64)
    y_pred = (y_probs >= threshold).astype(np.int64)

    return evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_probs,
        display_labels=display_labels,
        fit_time_seconds=0.0,
        predict_time_seconds=predict_time_seconds,
    )


def select_decision_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    max_fpr: float = 0.002,
    candidate_thresholds: Optional[Sequence[float]] = None,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """Select decision threshold according to the project operating rule:

    'Choose the threshold with highest validation ATTACK recall among thresholds
    with false-positive rate <= 0.2%. Break ties by higher precision, then higher threshold.'

    Args:
        y_true: Ground truth binary targets.
        y_probs: Predicted ATTACK probabilities.
        max_fpr: Maximum false-positive rate budget (default: 0.002 = 0.2%).
        candidate_thresholds: Sequence of thresholds to evaluate (default 0.05..0.95 in 0.01 steps).

    Returns:
        Tuple of (selected_threshold, selected_metrics_dict, all_evaluated_candidates_list)
    """
    if candidate_thresholds is None:
        candidate_thresholds = [round(0.05 + 0.01 * i, 2) for i in range(91)]

    # Ensure 0.50 is present
    if 0.50 not in candidate_thresholds:
        candidate_thresholds = sorted(list(set(candidate_thresholds + [0.50])))

    y_true = np.asarray(y_true, dtype=np.int64)
    y_probs = np.asarray(y_probs, dtype=np.float64)

    n_benign = int(np.sum(y_true == 0))
    n_attack = int(np.sum(y_true == 1))

    candidates_summary: List[Dict[str, Any]] = []

    for t in candidate_thresholds:
        y_pred = (y_probs >= t).astype(np.int64)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fpr = float(fp / n_benign) if n_benign > 0 else 0.0
        recall = float(tp / n_attack) if n_attack > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        candidates_summary.append({
            "threshold": float(t),
            "fpr": fpr,
            "recall": recall,
            "precision": precision,
            "f1_score": f1,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "satisfies_budget": bool(fpr <= max_fpr),
        })

    # Filter to candidates satisfying false-positive budget
    eligible = [c for c in candidates_summary if c["satisfies_budget"]]

    if not eligible:
        # Budget not met: retain 0.50 and record rationale
        selected_t = 0.50
        best_candidate = next((c for c in candidates_summary if abs(c["threshold"] - 0.50) < 1e-6), candidates_summary[0])
        best_candidate["selection_rationale"] = (
            f"No candidate threshold satisfied the false-positive budget of <= {max_fpr:.2%}. "
            "Retaining default threshold 0.50 per project operating rule."
        )
        best_candidate["budget_satisfied"] = False
    else:
        # Sort by: highest recall, then higher precision, then higher threshold
        eligible.sort(
            key=lambda c: (c["recall"], c["precision"], c["threshold"]),
            reverse=True,
        )
        best_candidate = eligible[0]
        selected_t = best_candidate["threshold"]
        best_candidate["selection_rationale"] = (
            f"Highest validation ATTACK recall ({best_candidate['recall']:.4f}) "
            f"among thresholds with FPR <= {max_fpr:.2%} (actual FPR: {best_candidate['fpr']:.4%}). "
            f"Tie-breakers: precision={best_candidate['precision']:.4f}, threshold={best_candidate['threshold']}."
        )
        best_candidate["budget_satisfied"] = True

    return selected_t, best_candidate, candidates_summary


def accumulate_streaming_confusion(
    existing_counts: Dict[str, int],
    chunk_y_true: np.ndarray,
    chunk_y_pred: np.ndarray,
) -> Dict[str, int]:
    """Incrementally accumulate confusion matrix counts without holding all vectors in memory."""
    cm = confusion_matrix(chunk_y_true, chunk_y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "tn": existing_counts.get("tn", 0) + int(tn),
        "fp": existing_counts.get("fp", 0) + int(fp),
        "fn": existing_counts.get("fn", 0) + int(fn),
        "tp": existing_counts.get("tp", 0) + int(tp),
    }


def generate_confusion_matrix_svg(
    cm: Dict[str, int],
    title: str,
    subtitle: str,
    output_path: pathlib.Path,
) -> None:
    """Generate a readable vector SVG confusion matrix card without third-party plotting packages."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tn = cm["tn"]
    fp = cm["fp"]
    fn = cm["fn"]
    tp = cm["tp"]
    total = tn + fp + fn + tp

    benign_total = tn + fp
    attack_total = fn + tp

    fpr = (fp / benign_total) if benign_total > 0 else 0.0
    rec = (tp / attack_total) if attack_total > 0 else 0.0
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    acc = ((tp + tn) / total) if total > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 520" width="760" height="520" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif">
  <defs>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-opacity="0.08"/>
    </filter>
  </defs>

  <!-- Background Card -->
  <rect x="10" y="10" width="740" height="500" rx="12" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" filter="url(#shadow)"/>

  <!-- Header -->
  <text x="35" y="48" font-size="20" font-weight="700" fill="#0f172a">{title}</text>
  <text x="35" y="74" font-size="13" font-weight="400" fill="#64748b">{subtitle}</text>

  <!-- Axis Headers -->
  <text x="380" y="115" font-size="13" font-weight="600" fill="#475569" text-anchor="middle">PREDICTED CLASS</text>
  <text x="250" y="138" font-size="12" font-weight="600" fill="#64748b" text-anchor="middle">Predicted BENIGN (0)</text>
  <text x="510" y="138" font-size="12" font-weight="600" fill="#64748b" text-anchor="middle">Predicted ATTACK (1)</text>

  <!-- Actual Class Vertical Label -->
  <g transform="translate(45, 290) rotate(-90)">
    <text x="0" y="0" font-size="13" font-weight="600" fill="#475569" text-anchor="middle">ACTUAL CLASS</text>
  </g>
  <text x="95" y="215" font-size="12" font-weight="600" fill="#64748b" text-anchor="end">Actual BENIGN (0)</text>
  <text x="95" y="335" font-size="12" font-weight="600" fill="#64748b" text-anchor="end">Actual ATTACK (1)</text>

  <!-- Matrix Grid -->
  <!-- Top-Left: True Negative -->
  <rect x="130" y="150" width="240" height="115" rx="8" fill="#f0fdf4" stroke="#86efac" stroke-width="1.5"/>
  <text x="145" y="178" font-size="13" font-weight="700" fill="#166534">TRUE NEGATIVE (TN)</text>
  <text x="145" y="212" font-size="26" font-weight="800" fill="#14532d">{tn:,}</text>
  <text x="145" y="242" font-size="12" font-weight="500" fill="#15803d">Specificity: {(tn/benign_total):.2%} ({tn}/{benign_total})</text>

  <!-- Top-Right: False Positive -->
  <rect x="390" y="150" width="240" height="115" rx="8" fill="#fef2f2" stroke="#fca5a5" stroke-width="1.5"/>
  <text x="405" y="178" font-size="13" font-weight="700" fill="#991b1b">FALSE POSITIVE (FP)</text>
  <text x="405" y="212" font-size="26" font-weight="800" fill="#7f1d1d">{fp:,}</text>
  <text x="405" y="242" font-size="12" font-weight="500" fill="#b91c1c">False Positive Rate: {fpr:.4%} ({fp}/{benign_total})</text>

  <!-- Bottom-Left: False Negative -->
  <rect x="130" y="280" width="240" height="115" rx="8" fill="#fff7ed" stroke="#fdba74" stroke-width="1.5"/>
  <text x="145" y="308" font-size="13" font-weight="700" fill="#9a3412">FALSE NEGATIVE (FN)</text>
  <text x="145" y="342" font-size="26" font-weight="800" fill="#7c2d12">{fn:,}</text>
  <text x="145" y="372" font-size="12" font-weight="500" fill="#c2410c">Missed Attacks: {(fn/attack_total):.2%} ({fn}/{attack_total})</text>

  <!-- Bottom-Right: True Positive -->
  <rect x="390" y="280" width="240" height="115" rx="8" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5"/>
  <text x="405" y="308" font-size="13" font-weight="700" fill="#1e40af">TRUE POSITIVE (TP)</text>
  <text x="405" y="342" font-size="26" font-weight="800" fill="#1e3a8a">{tp:,}</text>
  <text x="405" y="372" font-size="12" font-weight="500" fill="#2563eb">Attack Recall: {rec:.2%} ({tp}/{attack_total})</text>

  <!-- Footer Metrics Ribbon -->
  <rect x="35" y="415" width="690" height="70" rx="8" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>
  <text x="60" y="442" font-size="11" font-weight="600" fill="#64748b">ACCURACY</text>
  <text x="60" y="468" font-size="16" font-weight="700" fill="#0f172a">{acc:.4%}</text>

  <text x="200" y="442" font-size="11" font-weight="600" fill="#64748b">ATTACK PRECISION</text>
  <text x="200" y="468" font-size="16" font-weight="700" fill="#0f172a">{prec:.4%}</text>

  <text x="370" y="442" font-size="11" font-weight="600" fill="#64748b">ATTACK RECALL</text>
  <text x="370" y="468" font-size="16" font-weight="700" fill="#0f172a">{rec:.4%}</text>

  <text x="520" y="442" font-size="11" font-weight="600" fill="#64748b">ATTACK F1-SCORE</text>
  <text x="520" y="468" font-size="16" font-weight="700" fill="#0f172a">{f1:.4f}</text>

  <text x="660" y="442" font-size="11" font-weight="600" fill="#64748b">FPR BUDGET</text>
  <text x="660" y="468" font-size="16" font-weight="700" fill="#0f172a">&lt;= 0.20%</text>
</svg>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
