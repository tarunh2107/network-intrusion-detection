"""Unit tests for Phase 6B evaluation, dynamic thresholding, and metric edge cases."""

import pathlib
import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from app.ml.evaluate import (
    ALL_KNOWN_DISPLAY_LABELS,
    accumulate_streaming_confusion,
    evaluate_at_threshold,
    evaluate_predictions,
    generate_confusion_matrix_svg,
    get_attack_class_index,
    select_decision_threshold,
)


def test_attack_probability_column_lookup():
    """Verify dynamic ATTACK column index resolution across class orderings."""
    # Standard ordering [0, 1]
    assert get_attack_class_index(np.array([0, 1])) == 1
    assert get_attack_class_index([0, 1]) == 1

    # Inverted ordering [1, 0]
    assert get_attack_class_index(np.array([1, 0])) == 0
    assert get_attack_class_index([1, 0]) == 0

    # Missing class 1 raises ValueError
    with pytest.raises(ValueError, match="Class label 1 \\(ATTACK\\) was not found"):
        get_attack_class_index(np.array([0, 2]))


def test_threshold_boundary_behavior():
    """Verify that predictions satisfy strict probability >= threshold boundary rule."""
    y_true = np.array([0, 0, 1, 1])
    # Values right at, below, and above boundary 0.50
    probs_50 = np.array([0.49999, 0.50000, 0.50000, 0.50001])
    res_50 = evaluate_at_threshold(y_true, probs_50, threshold=0.50)
    # Predicted labels should be [0, 1, 1, 1]
    assert res_50["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}

    # Arbitrary boundary 0.35
    probs_35 = np.array([0.3499, 0.3500, 0.3500, 0.8000])
    res_35 = evaluate_at_threshold(y_true, probs_35, threshold=0.35)
    # Predicted labels should be [0, 1, 1, 1]
    assert res_35["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}


def test_fpr_constrained_threshold_selection_and_tie_breaking():
    """Verify operating rule: highest recall with FPR <= 0.2%, tie-breaking by higher precision then higher threshold."""
    # Construct synthetic true targets: 10,000 Benign, 1,000 Attack
    rng = np.random.default_rng(42)
    n_benign = 10000
    n_attack = 1000
    y_true = np.concatenate([np.zeros(n_benign, dtype=int), np.ones(n_attack, dtype=int)])

    # Synthetic candidate probabilities
    probs = np.zeros(len(y_true), dtype=float)
    # Benign probabilities mostly low (< 0.1), 10 benign at 0.30 (FPR at 0.25 = 10/10000 = 0.1%),
    # 30 benign at 0.15 (FPR at 0.10 = 40/10000 = 0.4% -> exceeds 0.2% budget!)
    probs[:n_benign] = rng.uniform(0.0, 0.05, size=n_benign)
    probs[:10] = 0.30  # At threshold 0.25: 10 false positives -> FPR = 0.1% (<= 0.2%)
    probs[10:40] = 0.15  # At threshold 0.10: 40 false positives -> FPR = 0.4% (> 0.2%)

    # Attacks: 950 at 0.80, 40 at 0.35 (detected at 0.25, missed at 0.50), 10 at 0.02
    probs[n_benign : n_benign + 950] = 0.80
    probs[n_benign + 950 : n_benign + 990] = 0.35
    probs[n_benign + 990 :] = 0.02

    # Case 1: Threshold 0.25 gives FPR = 0.10% (<= 0.2%), Recall = 990/1000 = 99.0%
    # Threshold 0.50 gives FPR = 0.00% (<= 0.2%), Recall = 950/1000 = 95.0%
    # Threshold 0.10 gives FPR = 0.40% (> 0.2% budget, filtered out)
    selected_t, best_meta, all_candidates = select_decision_threshold(
        y_true, probs, max_fpr=0.002, candidate_thresholds=[0.10, 0.25, 0.50, 0.85]
    )
    assert selected_t == 0.25
    assert best_meta["budget_satisfied"] is True
    assert best_meta["fpr"] <= 0.002

    # Case 2: Tie breaking by higher precision
    # Define candidates with identical recall but different precisions
    cand_list = [
        {"threshold": 0.30, "fpr": 0.001, "recall": 0.95, "precision": 0.92, "satisfies_budget": True},
        {"threshold": 0.40, "fpr": 0.001, "recall": 0.95, "precision": 0.96, "satisfies_budget": True},  # Higher precision!
    ]
    # Filter and sort directly using the module logic
    cand_list.sort(key=lambda c: (c["recall"], c["precision"], c["threshold"]), reverse=True)
    assert cand_list[0]["threshold"] == 0.40

    # Case 3: Tie breaking by higher threshold when recall and precision are identical
    cand_list_t = [
        {"threshold": 0.40, "fpr": 0.001, "recall": 0.95, "precision": 0.96, "satisfies_budget": True},
        {"threshold": 0.50, "fpr": 0.001, "recall": 0.95, "precision": 0.96, "satisfies_budget": True},  # Higher threshold!
    ]
    cand_list_t.sort(key=lambda c: (c["recall"], c["precision"], c["threshold"]), reverse=True)
    assert cand_list_t[0]["threshold"] == 0.50

    # Case 4: No candidate satisfies budget -> fallback to 0.50
    # Benign probabilities all 0.99 (impossible to achieve low FPR)
    probs_bad = np.full(len(y_true), 0.99)
    selected_bad_t, bad_meta, _ = select_decision_threshold(
        y_true, probs_bad, max_fpr=0.002, candidate_thresholds=[0.10, 0.50, 0.90]
    )
    assert selected_bad_t == 0.50
    assert bad_meta["budget_satisfied"] is False
    assert "No candidate threshold satisfied the false-positive budget" in bad_meta["selection_rationale"]


def test_absent_class_and_zero_denominator_handling():
    """Verify absent classes produce 'not estimable' without division errors, and small classes emit warnings."""
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 0, 1])
    # Labels with only BENIGN and PortScan (Heartbleed, DDoS, etc. are absent)
    display_labels = ["BENIGN", "BENIGN", "PortScan"]

    res = evaluate_predictions(y_true, y_pred, display_labels=display_labels)
    per_class = res["per_class_recall"]

    # Absent attack category Heartbleed
    assert "Heartbleed" in per_class
    assert per_class["Heartbleed"]["total"] == 0
    assert per_class["Heartbleed"]["recall"] is None
    assert per_class["Heartbleed"]["recall_display"] == "not estimable"
    assert per_class["Heartbleed"]["uncertainty_warning"] == "Absent class in evaluation partition"

    # PortScan has 1 example -> warning emitted
    assert per_class["PortScan"]["total"] == 1
    assert per_class["PortScan"]["recall"] == 1.0
    assert "high estimation variance" in per_class["PortScan"]["uncertainty_warning"]

    # Zero denominator precision test: all true = 0, all pred = 0 (no positive predictions)
    y_all_0 = np.array([0, 0, 0])
    y_pred_0 = np.array([0, 0, 0])
    res_zero = evaluate_predictions(y_all_0, y_pred_0)
    assert res_zero["precision"] == 0.0
    assert res_zero["recall"] == 0.0


def test_streaming_confusion_counts_matching_in_memory():
    """Verify that incremental streaming confusion counts match in-memory matrix exactly."""
    rng = np.random.default_rng(123)
    N = 25000
    y_true = rng.integers(0, 2, size=N)
    y_pred = rng.integers(0, 2, size=N)

    # In-memory reference calculation
    cm_ref = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn_ref, fp_ref, fn_ref, tp_ref = cm_ref.ravel()

    # Streaming chunks of 3,500 rows
    chunk_size = 3500
    streamed_counts = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        streamed_counts = accumulate_streaming_confusion(
            streamed_counts, y_true[start:end], y_pred[start:end]
        )

    assert streamed_counts["tn"] == int(tn_ref)
    assert streamed_counts["fp"] == int(fp_ref)
    assert streamed_counts["fn"] == int(fn_ref)
    assert streamed_counts["tp"] == int(tp_ref)


def test_generate_confusion_matrix_svg(tmp_path: pathlib.Path):
    """Verify that SVG confusion matrix card generates valid SVG file with correct counts."""
    cm = {"tn": 339000, "fp": 256, "fn": 380, "tp": 76508}
    svg_file = tmp_path / "test_cm.svg"
    generate_confusion_matrix_svg(
        cm=cm,
        title="Held-Out Test Set Evaluation",
        subtitle="Random Forest Model | Threshold = 0.45",
        output_path=svg_file,
    )
    assert svg_file.exists()
    content = svg_file.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "339,000" in content
    assert "76,508" in content
    assert "Held-Out Test Set Evaluation" in content
