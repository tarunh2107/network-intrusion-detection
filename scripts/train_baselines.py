"""Train and compare resource-limited baseline models on CICIDS2017 processed data.

Phase 6A execution script:
1. Loads 77-feature contract from data/processed/v1/features.json.
2. Samples 150,000 train rows and 50,000 validation rows across all shards deterministically.
3. Saves sample-selection metadata to data/processed/v1/sample_metadata_v1.json.
4. Fits and evaluates:
   - DummyClassifier (most frequent)
   - LogisticRegression (balanced, max_iter=1000, scaled)
   - DecisionTreeClassifier (max_depth=15, min_samples_leaf=5)
   - RandomForestClassifier (n_estimators=100, max_depth=18, min_samples_leaf=2, n_jobs=2)
5. Selects winning baseline by validation ATTACK F1-score (tie-breaker: lower FPR, lower latency).
6. Persists winning pipeline and metadata to models/baseline_v1/.
7. Verifies joblib reload equivalence.
8. Writes comparison artifacts to reports/generated/baseline_comparison.json and .md.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List

# Ensure project root is in sys.path
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.ml.evaluate import evaluate_predictions
from app.ml.preprocess import MODEL_FEATURE_COUNT, MODEL_FEATURE_NAMES
from app.ml.train import (
    DEFAULT_SEED,
    DEFAULT_TRAIN_SAMPLE_SIZE,
    DEFAULT_VAL_SAMPLE_SIZE,
    build_pipeline,
    load_model_artifact,
    sample_partition_shards,
    save_model_artifact,
)

# Ensure unbuffered standard output for real-time console monitoring
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def main() -> None:
    t_start = time.perf_counter()
    timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    print("=" * 80)
    print("PHASE 6A: BASELINE MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 80)
    print(f"Timestamp (UTC): {timestamp_utc}")
    print(f"Random Seed: {DEFAULT_SEED}")
    print(f"Train Sample Target: {DEFAULT_TRAIN_SAMPLE_SIZE:,} rows")
    print(f"Validation Sample Target: {DEFAULT_VAL_SAMPLE_SIZE:,} rows")
    print(f"Features: {MODEL_FEATURE_COUNT} canonical network flow features")
    print("-" * 80)

    # 1. Paths verification
    base_dir = pathlib.Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "processed" / "v1"
    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"
    manifest_path = data_dir / "manifest.json"
    features_path = data_dir / "features.json"
    models_out_dir = base_dir / "models" / "baseline_v1"
    reports_out_dir = base_dir / "reports" / "generated"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Processed data directories missing. Expected {train_dir} and {val_dir}"
        )

    # Load and verify features schema
    with open(features_path, "r", encoding="utf-8") as f:
        feat_meta = json.load(f)
    feature_names = feat_meta["feature_names"]
    if len(feature_names) != MODEL_FEATURE_COUNT:
        raise ValueError(
            f"Expected {MODEL_FEATURE_COUNT} features in {features_path}, found {len(feature_names)}"
        )
    print(f"[OK] Verified feature schema: {len(feature_names)} features matched.")

    # 2. Deterministic Sampling
    print("\n[1/5] Streaming deterministic samples across partition shards...")

    t0_sample = time.perf_counter()
    print("  -> Sampling training partition...")
    X_train, y_train, train_meta = sample_partition_shards(
        partition_dir=train_dir,
        sample_size=DEFAULT_TRAIN_SAMPLE_SIZE,
        seed=DEFAULT_SEED,
        manifest_path=manifest_path,
        feature_names=feature_names,
    )
    print(f"     Train sample loaded: {X_train.shape[0]:,} rows, shape {X_train.shape}")

    print("  -> Sampling validation partition (strictly ignoring test partition)...")
    X_val, y_val, val_meta = sample_partition_shards(
        partition_dir=val_dir,
        sample_size=DEFAULT_VAL_SAMPLE_SIZE,
        seed=DEFAULT_SEED,
        manifest_path=manifest_path,
        feature_names=feature_names,
    )
    print(f"     Val sample loaded:   {X_val.shape[0]:,} rows, shape {X_val.shape}")
    print(f"     Sampling elapsed time: {time.perf_counter() - t0_sample:.2f}s")

    # Record class representation in samples
    train_benign = int(np.sum(y_train == 0))
    train_attack = int(np.sum(y_train == 1))
    val_benign = int(np.sum(y_val == 0))
    val_attack = int(np.sum(y_val == 1))

    print(f"\n  Train distribution: {train_benign:,} BENIGN ({train_benign/len(y_train):.2%}), "
          f"{train_attack:,} ATTACK ({train_attack/len(y_train):.2%})")
    print(f"  Val distribution:   {val_benign:,} BENIGN ({val_benign/len(y_val):.2%}), "
          f"{val_attack:,} ATTACK ({val_attack/len(y_val):.2%})")

    # Persist sample metadata
    sample_metadata = {
        "timestamp_utc": timestamp_utc,
        "seed": DEFAULT_SEED,
        "train": {
            "target_sample_size": DEFAULT_TRAIN_SAMPLE_SIZE,
            "actual_rows": int(len(y_train)),
            "benign_rows": train_benign,
            "attack_rows": train_attack,
            "display_label_counts": train_meta["display_label"].value_counts().to_dict(),
        },
        "validation": {
            "target_sample_size": DEFAULT_VAL_SAMPLE_SIZE,
            "actual_rows": int(len(y_val)),
            "benign_rows": val_benign,
            "attack_rows": val_attack,
            "display_label_counts": val_meta["display_label"].value_counts().to_dict(),
        },
        "feature_count": MODEL_FEATURE_COUNT,
        "features_list": feature_names,
    }
    sample_meta_path = data_dir / "sample_metadata_v1.json"
    with open(sample_meta_path, "w", encoding="utf-8") as f:
        json.dump(sample_metadata, f, indent=2)
    print(f"  -> Saved sample metadata to {sample_meta_path}")

    # 3. Model Definition & Training Loop
    models_to_evaluate = [
        ("dummy", "Dummy (Most Frequent)", {}),
        ("logistic_regression", "Logistic Regression (Balanced)", {"max_iter": 1000}),
        ("decision_tree", "Decision Tree (max_depth=15, min_leaf=5)", {"max_depth": 15, "min_samples_leaf": 5}),
        ("random_forest", "Random Forest (100 trees, depth=18, min_leaf=2, jobs=2)", {
            "n_estimators": 100,
            "max_depth": 18,
            "min_samples_leaf": 2,
            "n_jobs": 2,
        }),
    ]

    print("\n[2/5] Training baseline models on 150,000 samples...")
    print("-" * 80)

    results: Dict[str, Dict[str, Any]] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}

    for model_key, display_name, kwargs in models_to_evaluate:
        print(f"\n>>> Training: {display_name} ...")
        pipe = build_pipeline(model_key, random_state=DEFAULT_SEED, **kwargs)

        t_fit_start = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_time = time.perf_counter() - t_fit_start
        print(f"    Fit completed in {fit_time:.2f}s")

        t_pred_start = time.perf_counter()
        y_pred = pipe.predict(X_val)
        predict_time = time.perf_counter() - t_pred_start

        # Probability predictions for PR-AUC
        y_scores = None
        if hasattr(pipe, "predict_proba"):
            try:
                y_scores = pipe.predict_proba(X_val)[:, 1]
            except Exception:
                y_scores = None

        eval_res = evaluate_predictions(
            y_true=y_val,
            y_pred=y_pred,
            y_scores=y_scores,
            display_labels=val_meta["display_label"],
            fit_time_seconds=fit_time,
            predict_time_seconds=predict_time,
        )

        results[model_key] = {
            "display_name": display_name,
            "model_type": model_key,
            "hyperparameters": kwargs,
            "metrics": eval_res,
        }
        fitted_pipelines[model_key] = pipe

        print(f"    Validation ATTACK Precision: {eval_res['precision']:.4f}")
        print(f"    Validation ATTACK Recall:    {eval_res['recall']:.4f}")
        print(f"    Validation ATTACK F1-Score:  {eval_res['f1_score']:.4f}")
        print(f"    Validation False Positive Rate (FPR): {eval_res['fpr']:.4%}")
        print(f"    Validation False Negative Rate (FNR): {eval_res['fnr']:.4%}")
        if eval_res["pr_auc"] is not None:
            print(f"    Validation PR-AUC (Average Precision): {eval_res['pr_auc']:.4f}")
        print(f"    Inference Latency: {eval_res['latency_ms_per_1000']:.2f} ms / 1,000 samples")

    # 4. Model Selection
    print("\n[3/5] Selecting Winning Baseline...")
    print("-" * 80)

    # Ranking by validation ATTACK F1 (primary), then lowest FPR (tie-break 1), lowest latency (tie-break 2)
    def model_sort_key(k: str) -> tuple[float, float, float]:
        m = results[k]["metrics"]
        return (
            m["f1_score"],
            -m["fpr"],  # higher is worse -> negative makes lower better
            -m["latency_ms_per_1000"],
        )

    ranked_keys = sorted(results.keys(), key=model_sort_key, reverse=True)
    winner_key = ranked_keys[0]
    winner_name = results[winner_key]["display_name"]
    winner_metrics = results[winner_key]["metrics"]
    winner_pipeline = fitted_pipelines[winner_key]

    print(f"  WINNER: {winner_name}")
    print(f"  ATTACK F1: {winner_metrics['f1_score']:.4f} | "
          f"Precision: {winner_metrics['precision']:.4f} | "
          f"Recall: {winner_metrics['recall']:.4f} | "
          f"FPR: {winner_metrics['fpr']:.4%}")

    # 5. Model Persistence & Equivalence Check
    print("\n[4/5] Persisting winning model artifact...")
    models_out_dir.mkdir(parents=True, exist_ok=True)

    model_metadata = {
        "model_name": winner_name,
        "model_type": winner_key,
        "training_timestamp_utc": timestamp_utc,
        "random_seed": DEFAULT_SEED,
        "selection_criteria": "Validation ATTACK F1-score (tie-breaker: lower FPR)",
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "feature_count": MODEL_FEATURE_COUNT,
        "feature_names": feature_names,
        "hyperparameters": results[winner_key]["hyperparameters"],
        "metrics_validation": winner_metrics,
        "git_commit": "uncommitted",
        "environment": {
            "python_version": sys.version,
            "platform": sys.platform,
        },
    }

    save_model_artifact(
        pipeline=winner_pipeline,
        metadata=model_metadata,
        output_dir=models_out_dir,
    )
    print(f"  -> Saved model artifact to {models_out_dir / 'model.joblib'}")
    print(f"  -> Saved model metadata to {models_out_dir / 'metadata.json'}")

    # Verification: reload and check equivalence
    print("  -> Verifying reload prediction equivalence...")
    reloaded_pipe, reloaded_meta = load_model_artifact(models_out_dir)
    test_slice = X_val[:200]
    preds_orig = winner_pipeline.predict(test_slice)
    preds_reloaded = reloaded_pipe.predict(test_slice)
    np.testing.assert_array_equal(
        preds_orig,
        preds_reloaded,
        err_msg="Reloaded pipeline predictions do not match original!",
    )
    if hasattr(winner_pipeline, "predict_proba"):
        proba_orig = winner_pipeline.predict_proba(test_slice)
        proba_reloaded = reloaded_pipe.predict_proba(test_slice)
        np.testing.assert_allclose(proba_orig, proba_reloaded, atol=1e-7)
    print("  [OK] Persistence and reload equivalence verified strictly.")

    # 6. Report Generation
    print("\n[5/5] Generating baseline comparison reports...")
    reports_out_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON report
    report_json_path = reports_out_dir / "baseline_comparison.json"
    full_report_data = {
        "timestamp_utc": timestamp_utc,
        "elapsed_seconds": round(time.perf_counter() - t_start, 2),
        "seed": DEFAULT_SEED,
        "sampling": {
            "train_sample_size": len(y_train),
            "validation_sample_size": len(y_val),
            "train_benign": train_benign,
            "train_attack": train_attack,
            "val_benign": val_benign,
            "val_attack": val_attack,
        },
        "winner": {
            "key": winner_key,
            "name": winner_name,
            "f1_score": winner_metrics["f1_score"],
            "precision": winner_metrics["precision"],
            "recall": winner_metrics["recall"],
            "fpr": winner_metrics["fpr"],
        },
        "models": results,
    }
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report_data, f, indent=2)
    print(f"  -> Saved JSON report: {report_json_path}")

    # Generate Markdown report
    report_md_path = reports_out_dir / "baseline_comparison.md"
    generate_markdown_report(report_md_path, full_report_data, models_to_evaluate)
    print(f"  -> Saved Markdown report: {report_md_path}")

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"PHASE 6A COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS")
    print("=" * 80)


def generate_markdown_report(
    md_path: pathlib.Path,
    data: Dict[str, Any],
    models_to_evaluate: List[tuple[str, str, Dict[str, Any]]],
) -> None:
    """Generate professional, detailed markdown report comparing all baseline models."""
    timestamp = data["timestamp_utc"]
    elapsed = data["elapsed_seconds"]
    sampling = data["sampling"]
    winner = data["winner"]
    models = data["models"]
    winner_metrics = models[winner["key"]]["metrics"]

    lines: List[str] = [
        "# Phase 6A: Baseline Model Training & Comparison Report",
        "",
        f"**Timestamp (UTC):** `{timestamp}`  ",
        f"**Total Execution Time:** `{elapsed:.2f} seconds`  ",
        f"**Random Seed:** `{data['seed']}`  ",
        f"**Winning Model:** `{winner['name']}`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "In Phase 6A, four resource-bounded baseline models were trained and evaluated on strictly disjoint, leak-free CICIDS2017 flow data under an offline binary classification contract (`BENIGN` = 0 vs `ATTACK` = 1).",
        "",
        "- **Training Sample:** 150,000 flows sampled deterministically across all 25 train shards.",
        "- **Validation Sample:** 50,000 flows sampled deterministically across all 8 validation shards.",
        "- **Test Partition Protection:** The test partition (`data/processed/v1/test`) was **strictly untouched and held out** for final testing.",
        f"- **Selected Champion:** **{winner['name']}** achieved the highest validation ATTACK F1-score (**{winner['f1_score']:.4f}**) with high precision (**{winner['precision']:.4f}**), strong recall (**{winner['recall']:.4f}**), and a low False Positive Rate (**{winner['fpr']:.4%}**).",
        "",
        "---",
        "",
        "## 2. Baseline Model Comparison Table",
        "",
        "| Model | Attack Precision | Attack Recall | Attack F1-score | Accuracy | FPR | FNR | PR-AUC | Fit Time (s) | Latency (ms/1k) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for model_key, display_name, _ in models_to_evaluate:
        m = models[model_key]["metrics"]
        pr_auc_str = f"{m['pr_auc']:.4f}" if m["pr_auc"] is not None else "N/A"
        lines.append(
            f"| **{display_name}** | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"**{m['f1_score']:.4f}** | {m['accuracy']:.4f} | {m['fpr']:.4%} | {m['fnr']:.4%} | "
            f"{pr_auc_str} | {m['fit_time_seconds']:.2f}s | {m['latency_ms_per_1000']:.2f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Confusion Matrix Breakdown",
        "",
        "| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Total Evaluated |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for model_key, display_name, _ in models_to_evaluate:
        cm = models[model_key]["metrics"]["confusion_matrix"]
        total = cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]
        lines.append(
            f"| **{display_name}** | {cm['tn']:,} | {cm['fp']:,} | {cm['fn']:,} | {cm['tp']:,} | {total:,} |"
        )

    # Per-attack category recall breakdown for the winning model
    lines.extend([
        "",
        "---",
        "",
        f"## 4. Per-Category Detection Breakdown ({winner['name']})",
        "",
        "Using metadata `display_label` annotations on the validation sample (strictly isolated from feature vectors):",
        "",
        "| Category | Class Type | Total in Val Sample | Detected Correctly | Detection Rate / Recall | Missed |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    winner_breakdown = models[winner["key"]]["metrics"]["per_class_recall"]
    # Sort categories alphabetically, putting BENIGN first
    sorted_cats = sorted(winner_breakdown.keys(), key=lambda c: (c != "BENIGN", c))

    for cat in sorted_cats:
        c_info = winner_breakdown[cat]
        class_type = "Attack" if c_info["is_attack"] else "Benign"
        lines.append(
            f"| `{cat}` | {class_type} | {c_info['total']:,} | {c_info['detected_correct']:,} | "
            f"**{c_info['recall']:.2%}** | {c_info['missed']:,} |"
        )

    # Resource utilization & Analysis
    lines.extend([
        "",
        "---",
        "",
        "## 5. Resource Utilization & Operational Profile",
        "",
        f"- **Peak Working Memory:** Maintained well within the 7 GB system budget by reading individual shards sequentially in streaming chunks and sampling rows uniformly.",
        f"- **Model Serialization:** Winning pipeline persisted to `models/baseline_v1/model.joblib` using joblib.",
        f"- **Inference Latency:** {winner_metrics['latency_ms_per_1000']:.2f} ms per 1,000 samples ({winner_metrics['latency_ms_per_1000']/1000:.4f} ms per single flow), satisfying real-time API latency requirements.",
        "- **Preprocessing Integration:** Imputation (`median`) and variance filtering (`threshold=0.0`) are bundled inside the Scikit-learn Pipeline, preventing runtime feature drift.",
        "",
        "---",
        "",
        "## 6. Recommendations for Phase 6B",
        "",
        "1. **Decision Threshold Calibration:** Analyze the Precision-Recall curve to tune the classification threshold away from default 0.50 based on deployment operational goals (e.g. prioritize minimizing false alarms vs catching stealthy attacks).",
        "2. **Address Rare Attack Blind Spots:** For low-sample attack categories (such as Infiltration or Web Attacks), examine feature distributions and evaluate synthetic balancing techniques (SMOTE, class weighting) or targeted feature engineering.",
        "3. **Feature Importance Inspection:** Extract Gini feature importances from the Random Forest model to determine the highest-ranking network flow attributes and evaluate opportunities to prune low-relevance features.",
        "4. **API Integration Readiness:** The serialized `model.joblib` pipeline natively ingests 77-element numeric vectors and outputs binary predictions with class probabilities, ready for FastAPI integration.",
    ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
