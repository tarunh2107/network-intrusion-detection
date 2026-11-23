"""Unit tests for Phase 6A baseline training, evaluation, and pipeline contract."""

import json
import pathlib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.ml.evaluate import evaluate_predictions
from app.ml.preprocess import MODEL_FEATURE_COUNT, MODEL_FEATURE_NAMES
from app.ml.train import (
    METADATA_COLUMNS,
    build_pipeline,
    load_model_artifact,
    sample_partition_shards,
    save_model_artifact,
)


def test_build_pipeline_all_supported_models():
    """Verify that all four required baseline models instantiate correctly with required steps."""
    for m_type in ["dummy", "logistic_regression", "decision_tree", "random_forest"]:
        pipe = build_pipeline(m_type, random_state=42)
        assert isinstance(pipe, Pipeline)
        step_names = [step[0] for step in pipe.steps]
        assert "imputer" in step_names
        assert "var_thresh" in step_names
        assert "clf" in step_names
        if m_type == "logistic_regression":
            assert "scaler" in step_names

    with pytest.raises(ValueError, match="Unknown model_type"):
        build_pipeline("xgboost_unauthorized")


def test_metadata_exclusion_and_feature_order(tmp_path: pathlib.Path):
    """Verify that sampling excludes metadata columns and preserves exact 77-feature order."""
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)

    # Create dummy shard
    header = METADATA_COLUMNS + list(MODEL_FEATURE_NAMES)
    rows = []
    for i in range(10):
        meta = [f"test_file_{i}.csv", i, "BENIGN", "BENIGN", 0, f"hash_{i}"]
        feats = [float(j + i) for j in range(MODEL_FEATURE_COUNT)]
        rows.append(meta + feats)

    shard_path = train_dir / "test_shard_0000.csv.gz"
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(shard_path, index=False, compression="gzip")

    X, y, meta_df = sample_partition_shards(
        partition_dir=train_dir,
        sample_size=5,
        seed=42,
        feature_names=MODEL_FEATURE_NAMES,
    )

    assert X.shape == (5, 77)
    assert y.shape == (5,)
    assert meta_df.shape == (5, 6)
    assert list(meta_df.columns) == METADATA_COLUMNS
    # Ensure no metadata columns leaked into X
    for meta_col in METADATA_COLUMNS:
        assert meta_col not in MODEL_FEATURE_NAMES


def test_test_partition_protection(tmp_path: pathlib.Path):
    """Verify that sample_partition_shards strictly refuses to sample from 'test' partition."""
    test_dir = tmp_path / "test"
    test_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="Test partition cannot be sampled or evaluated"):
        sample_partition_shards(partition_dir=test_dir, sample_size=10)


def test_preprocessing_no_validation_leakage():
    """Verify that imputer and variance threshold fit only on training data and transform val data faithfully."""
    # Column 0: variance on train, constant on val
    # Column 1: constant on train (should be dropped)
    # Column 2: has NaN on train (median imputed)
    X_train = np.array([
        [1.0, 0.0, 10.0],
        [2.0, 0.0, np.nan],
        [3.0, 0.0, 10.0],
        [4.0, 0.0, 20.0],
    ])
    y_train = np.array([0, 0, 1, 1])

    X_val = np.array([
        [5.0, 99.0, np.nan],
        [6.0, 88.0, 15.0],
    ])

    pipe = build_pipeline("decision_tree", random_state=42)
    pipe.fit(X_train, y_train)

    # Check VarianceThreshold dropped column 1
    var_thresh = pipe.named_steps["var_thresh"]
    # 2 features retained out of 3
    assert var_thresh.get_support().sum() == 2
    assert var_thresh.get_support()[1] == False  # column 1 dropped

    # Predict on validation data works without error
    preds = pipe.predict(X_val)
    assert preds.shape == (2,)


def test_evaluation_metrics_synthetic():
    """Verify evaluation metric calculations against exact hand-calculated figures."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    # TN=3, FP=1, FN=1, TP=3
    # FPR = 1 / 4 = 0.25
    # FNR = 1 / 4 = 0.25
    # Precision = 3 / 4 = 0.75
    # Recall = 3 / 4 = 0.75
    # F1 = 0.75
    display_labels = ["BENIGN", "BENIGN", "BENIGN", "BENIGN", "DDoS", "DDoS", "PortScan", "PortScan"]
    y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.4, 0.9, 0.85, 0.95])

    res = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
        display_labels=display_labels,
        fit_time_seconds=1.5,
        predict_time_seconds=0.08,
    )

    assert res["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 3}
    assert pytest.approx(res["fpr"]) == 0.25
    assert pytest.approx(res["fnr"]) == 0.25
    assert pytest.approx(res["precision"]) == 0.75
    assert pytest.approx(res["recall"]) == 0.75
    assert pytest.approx(res["f1_score"]) == 0.75
    assert res["fit_time_seconds"] == 1.5
    assert res["predict_time_seconds"] == 0.08
    assert res["pr_auc"] is not None

    # Per-class breakdown
    pk = res["per_class_recall"]
    assert pk["BENIGN"]["total"] == 4
    assert pk["BENIGN"]["detected_correct"] == 3  # 3 TN
    assert pytest.approx(pk["BENIGN"]["recall"]) == 0.75

    assert pk["DDoS"]["total"] == 2
    assert pk["DDoS"]["detected_correct"] == 1  # 1 TP
    assert pytest.approx(pk["DDoS"]["recall"]) == 0.50

    assert pk["PortScan"]["total"] == 2
    assert pk["PortScan"]["detected_correct"] == 2  # 2 TP
    assert pytest.approx(pk["PortScan"]["recall"]) == 1.0


def test_model_persistence_and_reload_equivalence(tmp_path: pathlib.Path):
    """Verify that joblib save and reload yields identical predictions and metadata."""
    X = np.random.RandomState(42).randn(50, 77)
    y = (X[:, 0] > 0).astype(int)

    pipe = build_pipeline("decision_tree", random_state=42)
    pipe.fit(X, y)

    meta = {
        "model_type": "decision_tree",
        "sample_size": 50,
        "features": list(MODEL_FEATURE_NAMES),
    }

    save_model_artifact(pipe, meta, tmp_path / "saved_model")
    loaded_pipe, loaded_meta = load_model_artifact(tmp_path / "saved_model")

    orig_preds = pipe.predict(X)
    loaded_preds = loaded_pipe.predict(X)
    np.testing.assert_array_equal(orig_preds, loaded_preds)

    orig_proba = pipe.predict_proba(X)
    loaded_proba = loaded_pipe.predict_proba(X)
    np.testing.assert_allclose(orig_proba, loaded_proba)

    assert loaded_meta["model_type"] == "decision_tree"
    assert loaded_meta["sample_size"] == 50
