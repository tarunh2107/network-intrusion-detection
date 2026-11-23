"""Preflight artifact and runtime configuration integrity checker.

Phase 9 Requirements:
- Verifies model binary, streaming SHA-256 checksum, and metadata schema.
- Verifies evaluation report, frozen threshold config, and benchmark demo data.
- Verifies database path write permissions.
- Does NOT deserialize joblib files or regenerate missing models automatically.
- Explains that a source-only clone requires documented training/preparation steps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import sys
from typing import List, Tuple

EXPECTED_MODEL_SHA256 = "4339282ba6ce68b5c44c6dd6a72e7796e755a77542d65021984dd1a8b0434793"
EXPECTED_THRESHOLD = 0.29
EXPECTED_FEATURE_COUNT = 77

WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parent.parent


def compute_file_sha256(path: pathlib.Path) -> str:
    """Compute streaming SHA-256 hex digest of a file in 64 KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_artifacts(
    model_path: pathlib.Path,
    frozen_config_path: pathlib.Path,
    metadata_path: pathlib.Path,
    evaluation_report_path: pathlib.Path,
    demo_csv_path: pathlib.Path,
    database_path: pathlib.Path,
) -> Tuple[bool, List[str]]:
    """Run all preflight integrity checks without loading or deserializing weights."""
    errors: List[str] = []
    checks_passed: List[str] = []

    print("=" * 70)
    print("NIDS Runtime Artifact & Integrity Preflight Check")
    print("=" * 70)

    # 1. Check Model Binary
    print("\n[1/6] Checking model artifact binary...")
    if not model_path.is_file():
        errors.append(f"Model artifact not found at: {model_path}")
    else:
        file_size = model_path.stat().st_size
        if file_size < 1000:
            errors.append(f"Model artifact is unexpectedly small ({file_size} bytes).")
        else:
            actual_sha = compute_file_sha256(model_path)
            if actual_sha.lower() != EXPECTED_MODEL_SHA256.lower():
                errors.append(
                    f"Model SHA-256 checksum mismatch!\n"
                    f"  Expected: {EXPECTED_MODEL_SHA256}\n"
                    f"  Actual:   {actual_sha}"
                )
            else:
                checks_passed.append(f"Model binary verified (SHA-256: {actual_sha[:16]}..., Size: {file_size:,} bytes).")
                print(f"      OK: Model binary valid (SHA-256: {actual_sha[:16]}...)")

    # 2. Check Frozen Evaluation Config
    print("\n[2/6] Checking frozen evaluation config...")
    if not frozen_config_path.is_file():
        errors.append(f"Frozen configuration not found at: {frozen_config_path}")
    else:
        try:
            with open(frozen_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg_sha = cfg.get("model_sha256", "")
            if cfg_sha.lower() != EXPECTED_MODEL_SHA256.lower():
                errors.append(f"Frozen config model_sha256 ({cfg_sha}) does not match expected model.")
            cfg_thresh = float(cfg.get("selected_threshold", -1.0))
            if abs(cfg_thresh - EXPECTED_THRESHOLD) > 1e-6:
                errors.append(f"Frozen config threshold ({cfg_thresh}) does not match expected {EXPECTED_THRESHOLD}.")
            checks_passed.append(f"Frozen configuration valid (Threshold: {cfg_thresh}).")
            print(f"      OK: Frozen config valid (Threshold: {cfg_thresh})")
        except Exception as exc:
            errors.append(f"Failed parsing frozen config: {exc}")

    # 3. Check Metadata & Feature Names
    print("\n[3/6] Checking feature metadata schema...")
    feature_names: List[str] = []
    if not metadata_path.is_file():
        errors.append(f"Model metadata not found at: {metadata_path}")
    else:
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            feature_names = meta.get("feature_names", [])
            if len(feature_names) != EXPECTED_FEATURE_COUNT:
                errors.append(
                    f"Metadata feature count mismatch: expected {EXPECTED_FEATURE_COUNT}, found {len(feature_names)}."
                )
            elif len(set(feature_names)) != EXPECTED_FEATURE_COUNT:
                errors.append("Metadata feature names contain duplicates.")
            else:
                checks_passed.append(f"Feature metadata valid ({len(feature_names)} unique features).")
                print(f"      OK: Feature schema valid ({len(feature_names)} features)")
        except Exception as exc:
            errors.append(f"Failed parsing metadata: {exc}")

    # 4. Check Held-Out Evaluation Report
    print("\n[4/6] Checking held-out evaluation report artifact...")
    if not evaluation_report_path.is_file():
        errors.append(f"Evaluation report not found at: {evaluation_report_path}")
    else:
        try:
            with open(evaluation_report_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            eval_metrics = eval_data.get("metrics_at_frozen_threshold", {})
            cm = eval_metrics.get("confusion_matrix", {})
            if "tp" not in cm or "tn" not in cm:
                errors.append("Evaluation report is missing confusion matrix fields.")
            else:
                checks_passed.append(
                    f"Evaluation report valid (TN: {cm.get('tn', 0):,}, TP: {cm.get('tp', 0):,}, Prec: {eval_metrics.get('precision', 0):.4f})."
                )
                print("      OK: Evaluation report valid")
        except Exception as exc:
            errors.append(f"Failed parsing evaluation report: {exc}")

    # 5. Check Benchmark Demo CSV
    print("\n[5/6] Checking benchmark demo CSV flows...")
    if not demo_csv_path.is_file():
        errors.append(f"Benchmark demo CSV not found at: {demo_csv_path}")
    else:
        try:
            with open(demo_csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = [h.strip() for h in next(reader)]
                row_count = sum(1 for _ in reader)
            if feature_names and headers != feature_names:
                errors.append("Demo CSV headers do not match model's expected 77 feature names.")
            elif row_count < 1:
                errors.append("Demo CSV contains no data rows.")
            else:
                checks_passed.append(f"Demo CSV valid ({row_count} rows, 77 features).")
                print(f"      OK: Demo CSV valid ({row_count} flows)")
        except Exception as exc:
            errors.append(f"Failed checking demo CSV: {exc}")

        # Check 1,000-flow demo CSV if present
        demo_1000 = demo_csv_path.parent / "example_flows_1000.csv"
        if demo_1000.is_file():
            try:
                with open(demo_1000, "r", encoding="utf-8") as f:
                    r1000 = csv.reader(f)
                    h1000 = [h.strip() for h in next(r1000)]
                    count1000 = sum(1 for _ in r1000)
                if feature_names and h1000 != feature_names:
                    errors.append("1,000-flow demo CSV headers do not match model's expected 77 feature names.")
                elif count1000 != 1000:
                    errors.append(f"1,000-flow demo CSV contains unexpected row count ({count1000} rows).")
                else:
                    checks_passed.append(f"1,000-flow demo CSV valid ({count1000} rows, 77 features).")
                    print(f"      OK: 1,000-flow demo CSV valid ({count1000} flows)")
            except Exception as exc:
                errors.append(f"Failed checking 1,000-flow demo CSV: {exc}")

    # 6. Check Database Directory Permissions
    print("\n[6/6] Checking SQLite database target directory...")
    db_dir = database_path.parent
    if not db_dir.exists():
        try:
            db_dir.mkdir(parents=True, exist_ok=True)
            print(f"      OK: Created database directory: {db_dir}")
        except Exception as exc:
            errors.append(f"Cannot create database directory {db_dir}: {exc}")
    else:
        # Test write permission by touching a temporary file
        test_file = db_dir / ".preflight_write_test"
        try:
            test_file.touch()
            test_file.unlink()
            checks_passed.append(f"Database directory writable ({db_dir}).")
            print(f"      OK: Database directory writable ({db_dir})")
        except Exception as exc:
            errors.append(f"Database directory {db_dir} is not writable: {exc}")

    print("\n" + "=" * 70)
    if errors:
        print("[ERROR] PREFLIGHT VERIFICATION FAILED:")
        for err in errors:
            print(f"  * {err}")
        print("\nNOTE FOR SOURCE-ONLY REPOSITORIES:")
        print("If this is a fresh source-only repository clone without pre-trained model weights,")
        print("you must execute the documented preparation and training pipeline:")
        print("  1. .\\.venv\\Scripts\\python.exe scripts/prepare_dataset.py --raw-dir data/raw/cicids2017")
        print("  2. .\\.venv\\Scripts\\python.exe scripts/train_baseline.py")
        print("  3. .\\.venv\\Scripts\\python.exe scripts/evaluate_phase6b.py")
        print("=" * 70)
        return False, errors

    print("[SUCCESS] ALL RUNTIME ARTIFACTS AND CONFIGURATIONS VERIFIED SUCCESSFULLY.")
    print("=" * 70)
    return True, []


def main():
    parser = argparse.ArgumentParser(description="Preflight check for NIDS runtime artifacts and paths.")
    parser.add_argument(
        "--model-path",
        type=pathlib.Path,
        default=WORKSPACE_ROOT / "models" / "baseline_v1" / "model.joblib",
        help="Path to trained model pipeline artifact.",
    )
    parser.add_argument(
        "--frozen-config",
        type=pathlib.Path,
        default=WORKSPACE_ROOT / "models" / "baseline_v1" / "frozen_evaluation_config.json",
        help="Path to frozen evaluation configuration.",
    )
    parser.add_argument(
        "--metadata-path",
        type=pathlib.Path,
        default=WORKSPACE_ROOT / "models" / "baseline_v1" / "metadata.json",
        help="Path to model metadata JSON.",
    )
    parser.add_argument(
        "--evaluation-report",
        type=pathlib.Path,
        default=WORKSPACE_ROOT / "reports" / "generated" / "test_evaluation.json",
        help="Path to held-out test evaluation report.",
    )
    parser.add_argument(
        "--demo-csv",
        type=pathlib.Path,
        default=WORKSPACE_ROOT / "data" / "demo" / "example_flows.csv",
        help="Path to demo flows CSV.",
    )
    parser.add_argument(
        "--database-path",
        type=pathlib.Path,
        default=WORKSPACE_ROOT / "data" / "predictions.db",
        help="Path to target SQLite database.",
    )

    args = parser.parse_args()

    success, _ = check_artifacts(
        model_path=args.model_path,
        frozen_config_path=args.frozen_config,
        metadata_path=args.metadata_path,
        evaluation_report_path=args.evaluation_report,
        demo_csv_path=args.demo_csv,
        database_path=args.database_path,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
