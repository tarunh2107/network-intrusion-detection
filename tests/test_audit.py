"""Unit tests for CICIDS2017 dataset audit logic."""

import csv
import json
import pathlib
import pytest
from scripts.audit_dataset import DatasetAuditor


def test_header_whitespace_and_duplicates(tmp_path: pathlib.Path):
    csv_file = tmp_path / "test_headers.csv"
    headers = [" Destination Port", " Flow Duration", " Fwd Header Length", " Fwd Header Length", " Label"]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(["80", "1000", "32", "32", "BENIGN"])

    auditor = DatasetAuditor(input_dir=tmp_path, output_dir=tmp_path / "out")
    audit = auditor.inspect_file_headers(csv_file)

    assert audit.column_count == 5
    assert audit.stripped_columns == ["Destination Port", "Flow Duration", "Fwd Header Length", "Fwd Header Length", "Label"]
    assert audit.raw_duplicates == [" Fwd Header Length"]
    assert audit.stripped_duplicates == ["Fwd Header Length"]
    assert audit.duplicate_indices["Fwd Header Length"] == [2, 3]


def test_numeric_anomalies_and_empty_labels(tmp_path: pathlib.Path):
    input_dir = tmp_path / "data"
    output_dir = tmp_path / "reports"
    input_dir.mkdir()

    csv_file = input_dir / "anomalies.csv"
    headers = ["f1", "f2", "Label"]
    rows = [
        ["10.5", "100", "BENIGN"],
        ["", "200", "PortScan"],               # f1 missing
        ["NaN", "300", "DDoS"],                # f1 nan string
        ["corrupted", "400", "DoS Hulk"],      # f1 invalid non-numeric
        ["Infinity", "500", "Botnet"],         # f1 positive infinity
        ["-Infinity", "600", "FTP-Patator"],   # f1 negative infinity
        ["15.0", "700", ""],                   # empty label
        ["20.0", "800", "   "],                # blank whitespace label
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    auditor = DatasetAuditor(input_dir=input_dir, output_dir=output_dir, chunk_size=4)
    results = auditor.run_audit()

    summary = results["overall_summary"]
    assert summary["total_rows"] == 8
    assert summary["benign_rows"] == 1
    assert summary["attack_rows"] == 5
    assert summary["empty_or_missing_labels"] == 2

    # f1 feature stats (pos 0)
    f1_stats = results["feature_statistics"][0]
    assert f1_stats["name"] == "f1"
    assert f1_stats["missing_or_blank_count"] == 1
    assert f1_stats["nan_string_count"] == 1
    assert f1_stats["invalid_numeric_count"] == 1
    assert f1_stats["positive_infinity_count"] == 1
    assert f1_stats["negative_infinity_count"] == 1
    assert f1_stats["finite_numeric_count"] == 3
    assert f1_stats["min_value"] == 10.5
    assert f1_stats["max_value"] == 20.0


def test_duplicate_and_conflicting_labels(tmp_path: pathlib.Path):
    input_dir = tmp_path / "data"
    output_dir = tmp_path / "reports"
    input_dir.mkdir()

    headers = ["feat1", "feat2", "Label"]

    # File 1: has 1 within-file duplicate
    file1 = input_dir / "file1.csv"
    with open(file1, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(["1", "2", "BENIGN"])
        writer.writerow(["1", "2", "BENIGN"])  # duplicate within file1
        writer.writerow(["3", "4", "ATTACK_A"])

    # File 2: has a cross-file duplicate of ('1', '2') and a conflicting label on ('3', '4')
    file2 = input_dir / "file2.csv"
    with open(file2, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(["1", "2", "BENIGN"])    # cross-file duplicate of ('1', '2')
        writer.writerow(["3", "4", "ATTACK_B"])  # conflicting label with file1's ('3', '4', 'ATTACK_A')
        writer.writerow(["5", "6", "BENIGN"])

    auditor = DatasetAuditor(input_dir=input_dir, output_dir=output_dir, chunk_size=2)
    results = auditor.run_audit()

    summary = results["overall_summary"]
    # Total rows = 3 + 3 = 6
    assert summary["total_rows"] == 6

    # Unique feature vectors: ('1', '2'), ('3', '4'), ('5', '6') -> 3 unique vectors
    assert summary["unique_feature_vectors"] == 3
    assert summary["excess_duplicate_features"] == 6 - 3  # 3 excess rows

    # Conflicting label feature vector: ('3', '4') has labels 'ATTACK_A' and 'ATTACK_B'
    assert summary["conflicting_label_feature_vectors"] == 1
    assert summary["conflicting_label_rows"] == 2

    # Cross-file feature vectors: ('1', '2') and ('3', '4') appear in both file1 and file2
    assert summary["cross_file_feature_vectors"] == 2
    assert summary["cross_file_rows"] == 5

    # Within file duplicates: file1 has 1 within-file duplicate, file2 has 0
    assert results["within_file_duplicates"]["file1.csv"] == 1
    assert results["within_file_duplicates"]["file2.csv"] == 0


def test_generated_artifacts_written(tmp_path: pathlib.Path):
    input_dir = tmp_path / "data"
    output_dir = tmp_path / "reports"
    input_dir.mkdir()

    file1 = input_dir / "sample.csv"
    with open(file1, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["f1", "f2", "Label"])
        writer.writerow(["1.0", "2.0", "BENIGN"])

    auditor = DatasetAuditor(input_dir=input_dir, output_dir=output_dir)
    auditor.run_audit()

    assert (output_dir / "dataset_audit.json").exists()
    assert (output_dir / "dataset_audit.md").exists()
    assert (output_dir / "label_counts.csv").exists()

    with open(output_dir / "dataset_audit.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["overall_summary"]["total_rows"] == 1


def test_checkpoint_preservation_and_markdown_table_formatting(tmp_path: pathlib.Path):
    input_dir = tmp_path / "data"
    output_dir = tmp_path / "reports"
    input_dir.mkdir()

    file1 = input_dir / "sample.csv"
    with open(file1, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["f1", "f2", "Label"])
        writer.writerow(["1.0", "2.0", "BENIGN"])
        writer.writerow(["1.0", "2.0", "BENIGN"])

    # Test default keep_db=True
    auditor_keep = DatasetAuditor(input_dir=input_dir, output_dir=output_dir, keep_db=True)
    auditor_keep.run_audit()

    db_path = output_dir / "temp_audit_duplicates.db"
    assert db_path.exists(), "SQLite DB checkpoint should be preserved when keep_db=True"

    # Verify markdown table formatting for Within-File Duplicate Excess Rows
    md_content = (output_dir / "dataset_audit.md").read_text(encoding="utf-8")
    assert "| `sample.csv` | 2 | 1 |" in md_content, "Within-file table should have 3 columns: Filename, Total Rows, Excess Rows"

    # Test keep_db=False deletes the DB
    auditor_clean = DatasetAuditor(input_dir=input_dir, output_dir=output_dir, keep_db=False)
    auditor_clean.run_audit()
    assert not db_path.exists(), "SQLite DB checkpoint should be removed when keep_db=False"

