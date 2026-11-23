"""Full validation scoring, threshold selection, and frozen held-out test evaluation.

Phase 6B execution script:
1. Verifies model artifact models/baseline_v1/model.joblib and 77-feature contract.
2. Dynamically locates ATTACK column index via model.classes_.
3. Scores entire validation partition (417,395 rows across 8 shards) in chunks of 25,000 rows.
4. Saves resumable per-shard predictions to reports/generated/eval_cache/validation/.
5. Selects decision threshold under the project operating rule:
   'Choose the threshold with highest validation ATTACK recall among thresholds with
   false-positive rate <= 0.2%. Break ties by higher precision, then higher threshold.'
6. Freezes configuration and metadata to models/baseline_v1/frozen_evaluation_config.json
   BEFORE opening any test shards.
7. Evaluates frozen configuration ONCE on entire held-out test partition (416,144 rows across 8 shards).
8. Saves resumable per-shard predictions to reports/generated/eval_cache/test/.
9. Generates:
   - reports/generated/threshold_selection.json and .md
   - reports/generated/test_evaluation.json and .md
   - reports/generated/test_confusion_matrix.svg
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from app.ml.evaluate import (
    ALL_KNOWN_DISPLAY_LABELS,
    DEFAULT_OPERATING_RULE,
    evaluate_at_threshold,
    evaluate_predictions,
    generate_confusion_matrix_svg,
    get_attack_class_index,
    select_decision_threshold,
)
from app.ml.preprocess import MODEL_FEATURE_COUNT, MODEL_FEATURE_NAMES

# Ensure unbuffered standard output for real-time monitoring
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def calculate_file_sha256(filepath: pathlib.Path) -> str:
    """Calculate streaming SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def score_partition_shards(
    partition_dir: pathlib.Path,
    pipeline: Any,
    attack_col_idx: int,
    feature_names: List[str],
    cache_dir: pathlib.Path,
    chunk_size: int = 25000,
    expected_row_counts: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Score all CSV.gz shards in a partition in chunks of 25,000 rows with resumable caching.

    Args:
        partition_dir: Directory containing shard files.
        pipeline: Fitted Scikit-learn Pipeline.
        attack_col_idx: Column index corresponding to ATTACK class.
        feature_names: Ordered list of 77 feature names.
        cache_dir: Directory to save/load per-shard parquet predictions.
        chunk_size: Processing chunk size.
        expected_row_counts: Optional mapping of shard filename to expected rows.

    Returns:
        DataFrame containing concatenated predictions and metadata across all shards.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_files = sorted(list(partition_dir.glob("*.csv.gz")))
    if not shard_files:
        raise FileNotFoundError(f"No .csv.gz shard files found in {partition_dir}")

    scored_shards: List[pd.DataFrame] = []
    total_shards = len(shard_files)

    for idx, sf in enumerate(shard_files, start=1):
        stem = sf.name.replace(".csv.gz", "")
        shard_cache_path = cache_dir / f"{stem}.parquet"
        expected_rows = expected_row_counts.get(sf.name) if expected_row_counts else None

        # Check for valid cache hit
        if shard_cache_path.is_file():
            try:
                cached_df = pd.read_parquet(shard_cache_path)
                if expected_rows is None or len(cached_df) == expected_rows:
                    print(f"  [{idx}/{total_shards}] [CACHE HIT] {sf.name} ({len(cached_df):,} rows)")
                    scored_shards.append(cached_df)
                    continue
                else:
                    print(f"  [{idx}/{total_shards}] Cache count mismatch for {sf.name} (got {len(cached_df)}, expected {expected_rows}). Re-scoring...")
            except Exception as e:
                print(f"  [{idx}/{total_shards}] Error reading cache for {sf.name}: {e}. Re-scoring...")

        # Score shard in chunks
        t0_shard = time.perf_counter()
        shard_chunk_list: List[pd.DataFrame] = []
        rows_processed = 0

        for chunk_df in pd.read_csv(sf, chunksize=chunk_size):
            # Model input: strictly ordered 77 features
            X_chunk = chunk_df[feature_names].values.astype(np.float64)

            # Predict probabilities
            probs = pipeline.predict_proba(X_chunk)
            prob_attack = probs[:, attack_col_idx].astype(np.float32)

            # Extract compact metadata
            compact_df = pd.DataFrame({
                "prob_attack": prob_attack,
                "binary_target": chunk_df["binary_target"].values.astype(np.int8),
                "display_label": chunk_df["display_label"].values,
                "source_file": chunk_df["source_file"].values,
                "source_row_number": chunk_df["source_row_number"].values.astype(np.int64),
                "fingerprint": chunk_df["fingerprint"].values,
            })
            shard_chunk_list.append(compact_df)
            rows_processed += len(compact_df)

        full_shard_df = pd.concat(shard_chunk_list, ignore_index=True)
        # Save to parquet cache
        full_shard_df.to_parquet(shard_cache_path, index=False)
        elapsed_shard = time.perf_counter() - t0_shard
        print(f"  [{idx}/{total_shards}] Scored & Cached {sf.name} ({rows_processed:,} rows) in {elapsed_shard:.2f}s")
        scored_shards.append(full_shard_df)

    combined_df = pd.concat(scored_shards, ignore_index=True)
    return combined_df


def main() -> None:
    t_global_start = time.perf_counter()
    timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    print("=" * 80)
    print("PHASE 6B: FULL VALIDATION EVALUATION, THRESHOLD SELECTION & FROZEN TEST")
    print("=" * 80)
    print(f"Timestamp (UTC): {timestamp_utc}")
    print(f"Features: {MODEL_FEATURE_COUNT} canonical network flow features")
    print("-" * 80)

    # 1. Verification of baseline artifact
    base_dir = pathlib.Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "baseline_v1" / "model.joblib"
    baseline_meta_path = base_dir / "models" / "baseline_v1" / "metadata.json"
    frozen_config_path = base_dir / "models" / "baseline_v1" / "frozen_evaluation_config.json"
    features_path = base_dir / "data" / "processed" / "v1" / "features.json"
    manifest_path = base_dir / "data" / "processed" / "v1" / "manifest.json"
    val_dir = base_dir / "data" / "processed" / "v1" / "validation"
    test_dir = base_dir / "data" / "processed" / "v1" / "test"
    reports_dir = base_dir / "reports" / "generated"

    if not model_path.is_file():
        raise FileNotFoundError(f"Baseline model missing: {model_path}")
    if not features_path.is_file():
        raise FileNotFoundError(f"Features JSON missing: {features_path}")

    # Compute and verify model checksum
    model_sha256 = calculate_file_sha256(model_path)
    print(f"[1/6] Verified model artifact: {model_path.name}")
    print(f"      SHA-256: {model_sha256}")
    print(f"      Size: {model_path.stat().st_size:,} bytes")

    # Load fitted pipeline
    pipeline = joblib.load(model_path)
    clf = pipeline.named_steps["clf"]
    attack_col_idx = get_attack_class_index(clf.classes_)
    print(f"      Classifier classes: {clf.classes_}")
    print(f"      Dynamically resolved ATTACK column index: {attack_col_idx}")

    # Load features schema
    with open(features_path, "r", encoding="utf-8") as f:
        feat_meta = json.load(f)
    feature_names = feat_meta["feature_names"]
    if len(feature_names) != MODEL_FEATURE_COUNT:
        raise ValueError(f"Feature count mismatch: expected {MODEL_FEATURE_COUNT}, got {len(feature_names)}")

    # Load manifest row counts for verification
    manifest_row_lookup: Dict[str, int] = {}
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as f:
            m_data = json.load(f)
        for s_info in m_data.get("shards", []):
            manifest_row_lookup[s_info["filename"]] = int(s_info["rows"])

    # 2. Score entire validation partition
    print("\n[2/6] Scoring entire validation partition (chunks of 25,000 rows)...")
    val_cache_dir = reports_dir / "eval_cache" / "validation"
    val_df = score_partition_shards(
        partition_dir=val_dir,
        pipeline=pipeline,
        attack_col_idx=attack_col_idx,
        feature_names=feature_names,
        cache_dir=val_cache_dir,
        chunk_size=25000,
        expected_row_counts=manifest_row_lookup,
    )

    n_val_total = len(val_df)
    n_val_benign = int(np.sum(val_df["binary_target"] == 0))
    n_val_attack = int(np.sum(val_df["binary_target"] == 1))
    print(f"      Total validation flows: {n_val_total:,}")
    print(f"      Validation distribution: {n_val_benign:,} BENIGN ({n_val_benign/n_val_total:.2%}), {n_val_attack:,} ATTACK ({n_val_attack/n_val_total:.2%})")

    # 3. Threshold Selection
    print("\n[3/6] Threshold selection on full validation set...")
    print(f"      Operating Rule: \"{DEFAULT_OPERATING_RULE}\"")

    val_probs = val_df["prob_attack"].values.astype(np.float64)
    val_targets = val_df["binary_target"].values.astype(np.int64)
    val_labels = val_df["display_label"].values

    selected_t, best_cand_meta, all_candidates = select_decision_threshold(
        y_true=val_targets,
        y_probs=val_probs,
        max_fpr=0.002,  # 0.2% max false-positive rate
        candidate_thresholds=[round(0.05 + 0.01 * i, 2) for i in range(91)],
    )

    # Full evaluation at default 0.50
    eval_val_050 = evaluate_at_threshold(
        y_true=val_targets,
        y_probs=val_probs,
        threshold=0.50,
        display_labels=val_labels,
    )

    # Full evaluation at selected threshold
    eval_val_selected = evaluate_at_threshold(
        y_true=val_targets,
        y_probs=val_probs,
        threshold=selected_t,
        display_labels=val_labels,
    )

    print("\n  Validation Comparison: Default (0.50) vs Selected Threshold")
    print("  " + "-" * 75)
    print(f"  Metric                     | Threshold = 0.50       | Selected ({selected_t:.2f})")
    print("  " + "-" * 75)
    print(f"  ATTACK Recall              | {eval_val_050['recall']:.4%}              | {eval_val_selected['recall']:.4%}")
    print(f"  ATTACK Precision           | {eval_val_050['precision']:.4%}              | {eval_val_selected['precision']:.4%}")
    print(f"  ATTACK F1-Score            | {eval_val_050['f1_score']:.4f}                | {eval_val_selected['f1_score']:.4f}")
    print(f"  False Positive Rate (FPR)  | {eval_val_050['fpr']:.4%}               | {eval_val_selected['fpr']:.4%}")
    print(f"  False Negative Rate (FNR)  | {eval_val_050['fnr']:.4%}               | {eval_val_selected['fnr']:.4%}")
    print(f"  Average Precision (AP)     | {eval_val_050['average_precision']:.4f}                | {eval_val_selected['average_precision']:.4f}")
    print(f"  True Negatives (TN)        | {eval_val_050['confusion_matrix']['tn']:,}               | {eval_val_selected['confusion_matrix']['tn']:,}")
    print(f"  False Positives (FP)       | {eval_val_050['confusion_matrix']['fp']:,}                   | {eval_val_selected['confusion_matrix']['fp']:,}")
    print(f"  False Negatives (FN)       | {eval_val_050['confusion_matrix']['fn']:,}                   | {eval_val_selected['confusion_matrix']['fn']:,}")
    print(f"  True Positives (TP)        | {eval_val_050['confusion_matrix']['tp']:,}                | {eval_val_selected['confusion_matrix']['tp']:,}")
    print("  " + "-" * 75)
    print(f"  Budget Met (FPR <= 0.20%): {best_cand_meta['budget_satisfied']}")
    print(f"  Selection Rationale: {best_cand_meta['selection_rationale']}")

    # Save threshold selection artifacts
    thresh_report_data = {
        "timestamp_utc": timestamp_utc,
        "operating_rule": DEFAULT_OPERATING_RULE,
        "budget_max_fpr": 0.002,
        "budget_satisfied": best_cand_meta["budget_satisfied"],
        "selected_threshold": selected_t,
        "selection_rationale": best_cand_meta["selection_rationale"],
        "validation_rows_evaluated": n_val_total,
        "validation_metrics_at_selected_threshold": eval_val_selected,
        "validation_metrics_at_default_0_50": eval_val_050,
        "all_candidates_summary": all_candidates,
    }
    with open(reports_dir / "threshold_selection.json", "w", encoding="utf-8") as f:
        json.dump(thresh_report_data, f, indent=2)

    generate_threshold_markdown_report(
        reports_dir / "threshold_selection.md",
        thresh_report_data,
        eval_val_050,
        eval_val_selected,
        selected_t,
    )
    print(f"      Saved threshold selection reports to {reports_dir / 'threshold_selection.md'}")

    # 4. Freeze Configuration BEFORE Opening Test Shards
    print("\n[4/6] Freezing model configuration and metadata BEFORE opening test shards...")
    frozen_config = {
        "frozen_timestamp_utc": timestamp_utc,
        "model_path": "models/baseline_v1/model.joblib",
        "model_sha256": model_sha256,
        "baseline_metadata_path": "models/baseline_v1/metadata.json",
        "feature_count": MODEL_FEATURE_COUNT,
        "feature_names": feature_names,
        "operating_rule": DEFAULT_OPERATING_RULE,
        "budget_max_fpr": 0.002,
        "budget_satisfied": best_cand_meta["budget_satisfied"],
        "selected_threshold": selected_t,
        "selection_rationale": best_cand_meta["selection_rationale"],
        "validation_performance_selected_threshold": {
            "threshold": selected_t,
            "precision": eval_val_selected["precision"],
            "recall": eval_val_selected["recall"],
            "f1_score": eval_val_selected["f1_score"],
            "fpr": eval_val_selected["fpr"],
            "fnr": eval_val_selected["fnr"],
            "average_precision": eval_val_selected["average_precision"],
            "confusion_matrix": eval_val_selected["confusion_matrix"],
        },
        "validation_performance_default_0_50": {
            "threshold": 0.50,
            "precision": eval_val_050["precision"],
            "recall": eval_val_050["recall"],
            "f1_score": eval_val_050["f1_score"],
            "fpr": eval_val_050["fpr"],
            "fnr": eval_val_050["fnr"],
            "average_precision": eval_val_050["average_precision"],
            "confusion_matrix": eval_val_050["confusion_matrix"],
        },
        "environment": {
            "python_version": sys.version,
            "platform": sys.platform,
        },
    }
    with open(frozen_config_path, "w", encoding="utf-8") as f:
        json.dump(frozen_config, f, indent=2)
    print(f"      [OK] Frozen configuration written to {frozen_config_path}")

    # 5. Single Frozen Held-Out Test Evaluation
    print("\n[5/6] Opening and evaluating test partition ONCE under frozen threshold...")
    test_cache_dir = reports_dir / "eval_cache" / "test"
    test_df = score_partition_shards(
        partition_dir=test_dir,
        pipeline=pipeline,
        attack_col_idx=attack_col_idx,
        feature_names=feature_names,
        cache_dir=test_cache_dir,
        chunk_size=25000,
        expected_row_counts=manifest_row_lookup,
    )

    n_test_total = len(test_df)
    n_test_benign = int(np.sum(test_df["binary_target"] == 0))
    n_test_attack = int(np.sum(test_df["binary_target"] == 1))
    print(f"      Total test flows: {n_test_total:,}")
    print(f"      Test distribution: {n_test_benign:,} BENIGN ({n_test_benign/n_test_total:.2%}), {n_test_attack:,} ATTACK ({n_test_attack/n_test_total:.2%})")

    test_probs = test_df["prob_attack"].values.astype(np.float64)
    test_targets = test_df["binary_target"].values.astype(np.int64)
    test_labels = test_df["display_label"].values

    # Evaluation on test set using frozen threshold
    eval_test_frozen = evaluate_at_threshold(
        y_true=test_targets,
        y_probs=test_probs,
        threshold=selected_t,
        display_labels=test_labels,
    )

    # Comparison evaluation on test set at 0.50
    eval_test_050 = evaluate_at_threshold(
        y_true=test_targets,
        y_probs=test_probs,
        threshold=0.50,
        display_labels=test_labels,
    )

    print("\n  HELD-OUT TEST EVALUATION RESULTS (Frozen Threshold = " f"{selected_t:.2f})")
    print("  " + "-" * 75)
    print(f"  ATTACK Precision:           {eval_test_frozen['precision']:.4%} ({eval_test_frozen['confusion_matrix']['tp']:,}/{eval_test_frozen['confusion_matrix']['tp']+eval_test_frozen['confusion_matrix']['fp']:,})")
    print(f"  ATTACK Recall:              {eval_test_frozen['recall']:.4%} ({eval_test_frozen['confusion_matrix']['tp']:,}/{eval_test_frozen['confusion_matrix']['tp']+eval_test_frozen['confusion_matrix']['fn']:,})")
    print(f"  ATTACK F1-Score:            {eval_test_frozen['f1_score']:.4f}")
    print(f"  False Positive Rate (FPR):  {eval_test_frozen['fpr']:.4%} ({eval_test_frozen['confusion_matrix']['fp']:,}/{n_test_benign:,})")
    print(f"  False Negative Rate (FNR):  {eval_test_frozen['fnr']:.4%} ({eval_test_frozen['confusion_matrix']['fn']:,}/{n_test_attack:,})")
    print(f"  Average Precision (AP):     {eval_test_frozen['average_precision']:.4f}")
    print(f"  Confusion Matrix:           TN={eval_test_frozen['confusion_matrix']['tn']:,}, FP={eval_test_frozen['confusion_matrix']['fp']:,}, FN={eval_test_frozen['confusion_matrix']['fn']:,}, TP={eval_test_frozen['confusion_matrix']['tp']:,}")
    print("  " + "-" * 75)

    # 6. Artifacts Generation
    print("\n[6/6] Generating reports and SVG confusion matrix figure...")

    # Generate vector SVG confusion matrix
    cm_svg_path = reports_dir / "test_confusion_matrix.svg"
    generate_confusion_matrix_svg(
        cm=eval_test_frozen["confusion_matrix"],
        title="CICIDS2017 Held-Out Test Evaluation",
        subtitle=f"Frozen Random Forest | Threshold = {selected_t:.2f} | 416,144 Evaluated Flows",
        output_path=cm_svg_path,
    )
    print(f"      Saved confusion matrix figure to {cm_svg_path}")

    # Save JSON report
    test_report_data = {
        "timestamp_utc": timestamp_utc,
        "elapsed_seconds": round(time.perf_counter() - t_global_start, 2),
        "frozen_configuration": {
            "model_path": "models/baseline_v1/model.joblib",
            "model_sha256": model_sha256,
            "selected_threshold": selected_t,
            "operating_rule": DEFAULT_OPERATING_RULE,
        },
        "test_partition_summary": {
            "total_rows": n_test_total,
            "benign_rows": n_test_benign,
            "attack_rows": n_test_attack,
        },
        "metrics_at_frozen_threshold": eval_test_frozen,
        "metrics_at_default_0_50": eval_test_050,
        "methodological_notes": {
            "row_weighting": "Row-weighted metrics give higher influence to repeated flow feature vectors present in the benchmark.",
            "unresolved_conflicts": "Conflicting label assignments across identical feature vectors remain unresolved in this offline benchmark.",
            "live_network_disclaimer": "Benchmark held-out test performance evaluates offline flow classification on CICIDS2017 and does not constitute a guarantee of live-network operational effectiveness.",
        },
    }
    with open(reports_dir / "test_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(test_report_data, f, indent=2)

    # Save Markdown report
    generate_test_markdown_report(
        reports_dir / "test_evaluation.md",
        test_report_data,
        eval_test_frozen,
        eval_test_050,
        selected_t,
    )
    print(f"      Saved held-out test report to {reports_dir / 'test_evaluation.md'}")

    total_time = time.perf_counter() - t_global_start
    print("\n" + "=" * 80)
    print(f"PHASE 6B COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS")
    print("=" * 80)


def generate_threshold_markdown_report(
    md_path: pathlib.Path,
    data: Dict[str, Any],
    eval_050: Dict[str, Any],
    eval_sel: Dict[str, Any],
    selected_t: float,
) -> None:
    """Generate Markdown report for threshold selection."""
    lines: List[str] = [
        "# Phase 6B: Validation Threshold Selection Report",
        "",
        f"**Timestamp (UTC):** `{data['timestamp_utc']}`  ",
        f"**Validation Flows Evaluated:** `{data['validation_rows_evaluated']:,}`  ",
        f"**Operating Selection Rule:**  ",
        f"> *\"{data['operating_rule']}\"*",
        "",
        "---",
        "",
        "## 1. Selected Decision Threshold",
        "",
        f"- **Chosen Threshold:** `{selected_t:.2f}`",
        f"- **FPR Budget Satisfied (FPR <= 0.20%):** `{data['budget_satisfied']}`",
        f"- **Selection Rationale:** {data['selection_rationale']}",
        "",
        "---",
        "",
        "## 2. Full Validation Performance: Default (0.50) vs Selected Threshold",
        "",
        "| Metric | Default Threshold (0.50) | Selected Threshold (" f"{selected_t:.2f}) | Operational Impact |",
        "| :--- | :--- | :--- | :--- |",
        f"| **ATTACK Precision** | {eval_050['precision']:.4%} | {eval_sel['precision']:.4%} | Change in positive predictive value |",
        f"| **ATTACK Recall** | {eval_050['recall']:.4%} | {eval_sel['recall']:.4%} | Change in proportion of attacks caught |",
        f"| **ATTACK F1-Score** | {eval_050['f1_score']:.4f} | {eval_sel['f1_score']:.4f} | Harmonic mean |",
        f"| **False Positive Rate (FPR)** | {eval_050['fpr']:.4%} | {eval_sel['fpr']:.4%} | Must remain <= 0.20% budget |",
        f"| **False Negative Rate (FNR)** | {eval_050['fnr']:.4%} | {eval_sel['fnr']:.4%} | Missed attacks proportion |",
        f"| **Average Precision (AP)** | {eval_050['average_precision']:.4f} | {eval_sel['average_precision']:.4f} | Overall PR curve metric |",
        "",
        "### Confusion Matrix Comparison",
        "",
        "| Threshold | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Total |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| **0.50** | {eval_050['confusion_matrix']['tn']:,} | {eval_050['confusion_matrix']['fp']:,} | {eval_050['confusion_matrix']['fn']:,} | {eval_050['confusion_matrix']['tp']:,} | {data['validation_rows_evaluated']:,} |",
        f"| **{selected_t:.2f}** | {eval_sel['confusion_matrix']['tn']:,} | {eval_sel['confusion_matrix']['fp']:,} | {eval_sel['confusion_matrix']['fn']:,} | {eval_sel['confusion_matrix']['tp']:,} | {data['validation_rows_evaluated']:,} |",
        "",
        "---",
        "",
        f"## 3. Per-Category Detection Breakdown on Full Validation Set (Threshold = {selected_t:.2f})",
        "",
        "| Category | Class Type | Support in Validation | Detected Correctly | Detection Rate / Recall | Missed | Notes / Uncertainty |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    p_sel = eval_sel["per_class_recall"]
    for cat in sorted(p_sel.keys(), key=lambda c: (c != "BENIGN", c)):
        info = p_sel[cat]
        class_type = "Attack" if info["is_attack"] else "Benign"
        warn_str = info.get("uncertainty_warning") or "Standard sample"
        lines.append(
            f"| `{cat}` | {class_type} | {info['total']:,} | {info['detected_correct']:,} | "
            f"**{info['recall_display']}** | {info['missed']:,} | {warn_str} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def generate_test_markdown_report(
    md_path: pathlib.Path,
    data: Dict[str, Any],
    eval_frozen: Dict[str, Any],
    eval_050: Dict[str, Any],
    selected_t: float,
) -> None:
    """Generate Markdown report for held-out test evaluation."""
    n_total = data["test_partition_summary"]["total_rows"]
    n_benign = data["test_partition_summary"]["benign_rows"]
    n_attack = data["test_partition_summary"]["attack_rows"]

    lines: List[str] = [
        "# Phase 6B: Held-Out Test Partition Evaluation Report",
        "",
        f"**Timestamp (UTC):** `{data['timestamp_utc']}`  ",
        f"**Model Artifact:** `models/baseline_v1/model.joblib` (SHA-256: `{data['frozen_configuration']['model_sha256']}`)  ",
        f"**Frozen Decision Threshold:** `{selected_t:.2f}`  ",
        f"**Operating Selection Rule:** *\"{data['frozen_configuration']['operating_rule']}\"*  ",
        f"**Total Held-Out Test Flows:** `{n_total:,}` (`{n_benign:,}` BENIGN, `{n_attack:,}` ATTACK)  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "The held-out test partition (`data/processed/v1/test/`) was evaluated **exactly once** using the frozen model and threshold. The model parameters, preprocessing pipeline, and decision threshold were strictly locked before opening any test shards.",
        "",
        f"- **ATTACK Precision:** `{eval_frozen['precision']:.4%}`",
        f"- **ATTACK Recall:** `{eval_frozen['recall']:.4%}`",
        f"- **ATTACK F1-Score:** `{eval_frozen['f1_score']:.4f}`",
        f"- **False Positive Rate (FPR):** `{eval_frozen['fpr']:.4%}` (Budget requirement: `<= 0.20%`)",
        f"- **False Negative Rate (FNR):** `{eval_frozen['fnr']:.4%}`",
        f"- **Average Precision (AP):** `{eval_frozen['average_precision']:.4f}` (Scikit-learn `average_precision_score`)",
        "",
        "---",
        "",
        "## 2. Test Set Confusion Matrix (Threshold = " f"{selected_t:.2f})",
        "",
        "| Actual \\ Predicted | Predicted BENIGN (0) | Predicted ATTACK (1) | Total Actual | Class Metric |",
        "| :--- | :--- | :--- | :--- | :--- |",
        f"| **Actual BENIGN (0)** | **{eval_frozen['confusion_matrix']['tn']:,}** (TN) | **{eval_frozen['confusion_matrix']['fp']:,}** (FP) | {n_benign:,} | Specificity: {(eval_frozen['confusion_matrix']['tn']/n_benign):.2%} (FPR: {eval_frozen['fpr']:.4%}) |",
        f"| **Actual ATTACK (1)** | **{eval_frozen['confusion_matrix']['fn']:,}** (FN) | **{eval_frozen['confusion_matrix']['tp']:,}** (TP) | {n_attack:,} | Recall: {eval_frozen['recall']:.2%} (FNR: {eval_frozen['fnr']:.4%}) |",
        f"| **Total Predicted** | {eval_frozen['confusion_matrix']['tn']+eval_frozen['confusion_matrix']['fn']:,} | {eval_frozen['confusion_matrix']['fp']+eval_frozen['confusion_matrix']['tp']:,} | {n_total:,} | Accuracy: {eval_frozen['accuracy']:.4%} |",
        "",
        "---",
        "",
        "## 3. Threshold Comparison on Held-Out Test Partition",
        "",
        "| Metric | Frozen Threshold (" f"{selected_t:.2f}) | Default Threshold (0.50) | Difference |",
        "| :--- | :--- | :--- | :--- |",
        f"| **ATTACK Precision** | **{eval_frozen['precision']:.4%}** | {eval_050['precision']:.4%} | {(eval_frozen['precision'] - eval_050['precision']):+.4%} |",
        f"| **ATTACK Recall** | **{eval_frozen['recall']:.4%}** | {eval_050['recall']:.4%} | {(eval_frozen['recall'] - eval_050['recall']):+.4%} |",
        f"| **ATTACK F1-Score** | **{eval_frozen['f1_score']:.4f}** | {eval_050['f1_score']:.4f} | {(eval_frozen['f1_score'] - eval_050['f1_score']):+.4f} |",
        f"| **False Positive Rate (FPR)** | **{eval_frozen['fpr']:.4%}** | {eval_050['fpr']:.4%} | {(eval_frozen['fpr'] - eval_050['fpr']):+.4%} |",
        f"| **False Negative Rate (FNR)** | **{eval_frozen['fnr']:.4%}** | {eval_050['fnr']:.4%} | {(eval_frozen['fnr'] - eval_050['fnr']):+.4%} |",
        "",
        "---",
        "",
        f"## 4. Per-Category Detection Breakdown on Held-Out Test Set (Threshold = {selected_t:.2f})",
        "",
        "> [!IMPORTANT]",
        "> This binary classifier outputs predictions strictly for `BENIGN` vs `ATTACK`. Category breakdown is computed strictly by grouping test flows by their ground-truth metadata display labels.",
        "",
        "| Category | Class Type | Support in Test Split | Detected Correctly | Detection Rate / Recall | Missed | Notes / Uncertainty Warning |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    p_test = eval_frozen["per_class_recall"]
    for cat in sorted(p_test.keys(), key=lambda c: (c != "BENIGN", c)):
        info = p_test[cat]
        class_type = "Attack" if info["is_attack"] else "Benign"
        warn_str = info.get("uncertainty_warning") or "Sufficient sample"
        lines.append(
            f"| `{cat}` | {class_type} | {info['total']:,} | {info['detected_correct']:,} | "
            f"**{info['recall_display']}** | {info['missed']:,} | {warn_str} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Methodological Boundaries & Limitations",
        "",
        "1. **Row-Weighted Evaluation:** Repeated identical network flow vectors within the test partition remain uncollapsed and weighted equally per flow row. Repeated patterns exert proportionally higher influence on overall metrics.",
        "2. **Unresolved Binary Conflicts:** Groups mapping to conflicting labels in the original dataset remain in their assigned folds without ad-hoc filtering.",
        "3. **Small-Sample High Variance:** Categories with very few instances (e.g. `Heartbleed` with 1 flow, `Web Attack - Sql Injection` with 2 flows) exhibit high statistical uncertainty. Detection results for these classes must be reported with explicit variance warnings.",
        "4. **Offline Benchmark Scope:** These results measure offline discrimination on the CICIDS2017 flow collection. They do not constitute a demonstration of effectiveness on unconstrained live-network traffic or under temporal distribution drift.",
    ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
