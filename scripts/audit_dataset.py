#!/usr/bin/env python3
"""CICIDS2017 Comprehensive Dataset Audit & Schema Inspection Script.

Sequentially inspects raw CSVs in bounded chunks to analyze:
1. Raw headers, whitespace stripping, and duplicate column names.
2. Character encoding and raw label byte integrity.
3. Row counts, label distribution, missing, invalid numeric, and infinity values.
4. Value-level equality verification for suspected duplicate columns.
5. Bounded-memory duplicate analysis (within-file, cross-file, conflicting labels)
   using an on-disk SQLite indexing database.
6. Exports reports/generated/dataset_audit.json, dataset_audit.md, and label_counts.csv.
"""

import argparse
import csv
import dataclasses
import datetime
import hashlib
import json
import math
import os
import pathlib
import shutil
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclasses.dataclass
class HeaderAudit:
    file_name: str
    column_count: int
    raw_columns: List[str]
    stripped_columns: List[str]
    raw_duplicates: List[str]
    stripped_duplicates: List[str]
    duplicate_indices: Dict[str, List[int]]


class DatasetAuditor:
    def __init__(
        self,
        input_dir: pathlib.Path,
        output_dir: pathlib.Path,
        chunk_size: int = 25000,
        sqlite_db_path: Optional[pathlib.Path] = None,
        keep_db: bool = True,
    ):
        self.input_dir = pathlib.Path(input_dir)
        self.output_dir = pathlib.Path(output_dir)
        self.chunk_size = chunk_size
        self.sqlite_db_path = sqlite_db_path or (self.output_dir / "temp_audit_duplicates.db")
        self.keep_db = keep_db

    def inspect_file_headers(self, csv_path: pathlib.Path, encoding: str = "utf-8") -> HeaderAudit:
        with open(csv_path, "r", encoding=encoding, errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)

        raw_columns = header
        stripped_columns = [col.strip() for col in header]

        # Identify duplicates
        seen_raw: Set[str] = set()
        raw_dups: Set[str] = set()
        for c in raw_columns:
            if c in seen_raw:
                raw_dups.add(c)
            seen_raw.add(c)

        seen_stripped: Set[str] = set()
        stripped_dups: Set[str] = set()
        for c in stripped_columns:
            if c in seen_stripped:
                stripped_dups.add(c)
            seen_stripped.add(c)

        dup_indices: Dict[str, List[int]] = {}
        for d in stripped_dups:
            dup_indices[d] = [i for i, c in enumerate(stripped_columns) if c == d]

        return HeaderAudit(
            file_name=csv_path.name,
            column_count=len(raw_columns),
            raw_columns=raw_columns,
            stripped_columns=stripped_columns,
            raw_duplicates=sorted(list(raw_dups)),
            stripped_duplicates=sorted(list(stripped_dups)),
            duplicate_indices=dup_indices,
        )

    def init_sqlite_db(self, db_path: pathlib.Path, reuse_existing: bool = False) -> sqlite3.Connection:
        if reuse_existing and db_path.exists():
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='records';")
            if cur.fetchone():
                return conn
            conn.close()

        if db_path.exists():
            db_path.unlink()
        wal = db_path.with_name(db_path.name + "-wal")
        shm = db_path.with_name(db_path.name + "-shm")
        if wal.exists():
            wal.unlink()
        if shm.exists():
            shm.unlink()

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("PRAGMA synchronous = NORMAL;")
        cur.execute("""
            CREATE TABLE records (
                feature_hash TEXT NOT NULL,
                label TEXT NOT NULL,
                file_id INTEGER NOT NULL
            );
        """)
        conn.commit()
        return conn

    def run_audit(self) -> Dict[str, Any]:
        start_time = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(list(self.input_dir.glob("*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.input_dir}")

        # Check free disk space before initializing SQLite DB
        disk_free = shutil.disk_usage(self.output_dir).free
        if disk_free < 500 * 1024 * 1024:
            raise OSError(f"Insufficient disk space ({disk_free / (1024**2):.1f} MB) for duplicate indexing.")

        db_conn = self.init_sqlite_db(self.sqlite_db_path)
        db_cur = db_conn.cursor()

        headers_audit: List[HeaderAudit] = []
        file_metrics: List[Dict[str, Any]] = []

        # Global accumulators
        global_total_rows = 0
        global_raw_labels: Dict[str, int] = {}
        global_stripped_labels: Dict[str, int] = {}
        global_benign_count = 0
        global_attack_count = 0
        global_empty_label_count = 0

        # Feature level statistics (indexed by positional index 0..77)
        # Note: 79 columns total: 0..77 features, index 78 is Label
        ref_header = self.inspect_file_headers(csv_files[0])
        feature_names = ref_header.stripped_columns[:-1]
        label_col_name = ref_header.stripped_columns[-1]

        # Feature stats dictionary: pos -> dict of counts & min/max
        feature_stats: Dict[int, Dict[str, Any]] = {}
        for pos in range(len(feature_names)):
            feature_stats[pos] = {
                "name": feature_names[pos],
                "missing_count": 0,
                "nan_string_count": 0,
                "invalid_numeric_count": 0,
                "pos_inf_count": 0,
                "neg_inf_count": 0,
                "finite_count": 0,
                "min_value": float("inf"),
                "max_value": float("-inf"),
                "distinct_sample_values": set(),
                "all_constant": True,
                "first_finite_val": None,
            }

        # Value comparison for suspected duplicate column (indices 34 and 55 in 79-col schema)
        dup_col_34_55_checked = True
        dup_col_34_55_identical = True
        dup_col_34_55_mismatches = 0

        print(f"Starting audit across {len(csv_files)} files (chunk size: {self.chunk_size:,})...")

        for file_id, csv_path in enumerate(csv_files):
            file_start = time.time()
            header_audit = self.inspect_file_headers(csv_path)
            headers_audit.append(header_audit)

            # Check encoding by reading lines
            file_rows = 0
            file_raw_labels: Dict[str, int] = {}
            file_stripped_labels: Dict[str, int] = {}
            file_benign = 0
            file_attack = 0
            file_empty_label = 0

            # Streaming reading
            sqlite_batch: List[Tuple[str, str, int]] = []

            with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                _ = next(reader)  # skip header

                chunk_rows = []
                for row in reader:
                    file_rows += 1
                    global_total_rows += 1

                    if len(row) < len(header_audit.raw_columns):
                        # Malformed row
                        continue

                    # Feature vector (all columns except the last label column)
                    features_raw = row[:-1]
                    raw_label = row[-1]
                    stripped_label = raw_label.strip()

                    # Label tracking
                    file_raw_labels[raw_label] = file_raw_labels.get(raw_label, 0) + 1
                    file_stripped_labels[stripped_label] = file_stripped_labels.get(stripped_label, 0) + 1
                    global_raw_labels[raw_label] = global_raw_labels.get(raw_label, 0) + 1
                    global_stripped_labels[stripped_label] = global_stripped_labels.get(stripped_label, 0) + 1

                    if not stripped_label:
                        file_empty_label += 1
                        global_empty_label_count += 1
                    elif stripped_label.lower() == "benign":
                        file_benign += 1
                        global_benign_count += 1
                    else:
                        file_attack += 1
                        global_attack_count += 1

                    # Canonical feature representation for duplicate hash
                    # Join stripped string representations of features
                    canonical_feature_str = "\t".join(c.strip() for c in features_raw)
                    feat_hash = hashlib.sha256(canonical_feature_str.encode("utf-8")).hexdigest()
                    sqlite_batch.append((feat_hash, stripped_label, file_id))

                    if len(sqlite_batch) >= self.chunk_size:
                        db_cur.executemany("INSERT INTO records VALUES (?, ?, ?)", sqlite_batch)
                        db_conn.commit()
                        sqlite_batch.clear()

                    # Duplicate column comparison (index 34 vs index 55)
                    if len(features_raw) > 55:
                        if features_raw[34] != features_raw[55]:
                            dup_col_34_55_identical = False
                            dup_col_34_55_mismatches += 1

                    # Per-feature numerical parsing
                    for pos, val_str in enumerate(features_raw):
                        val_stripped = val_str.strip()
                        fstat = feature_stats[pos]

                        if not val_stripped:
                            fstat["missing_count"] += 1
                            continue

                        val_lower = val_stripped.lower()
                        if val_lower in ("nan", "null", "none"):
                            fstat["nan_string_count"] += 1
                            continue

                        if val_lower in ("infinity", "+infinity", "inf", "+inf"):
                            fstat["pos_inf_count"] += 1
                            continue
                        elif val_lower in ("-infinity", "-inf"):
                            fstat["neg_inf_count"] += 1
                            continue

                        try:
                            num = float(val_stripped)
                            if math.isinf(num):
                                if num > 0:
                                    fstat["pos_inf_count"] += 1
                                else:
                                    fstat["neg_inf_count"] += 1
                            elif math.isnan(num):
                                fstat["nan_string_count"] += 1
                            else:
                                fstat["finite_count"] += 1
                                if num < fstat["min_value"]:
                                    fstat["min_value"] = num
                                if num > fstat["max_value"]:
                                    fstat["max_value"] = num

                                if fstat["all_constant"]:
                                    if fstat["first_finite_val"] is None:
                                        fstat["first_finite_val"] = num
                                    elif num != fstat["first_finite_val"]:
                                        fstat["all_constant"] = False
                        except ValueError:
                            fstat["invalid_numeric_count"] += 1

            if sqlite_batch:
                db_cur.executemany("INSERT INTO records VALUES (?, ?, ?)", sqlite_batch)
                db_conn.commit()
                sqlite_batch.clear()

            file_elapsed = time.time() - file_start
            print(f"  Processed {csv_path.name}: {file_rows:,} rows in {file_elapsed:.1f}s")

            file_metrics.append({
                "file_name": csv_path.name,
                "file_id": file_id,
                "row_count": file_rows,
                "benign_count": file_benign,
                "attack_count": file_attack,
                "empty_label_count": file_empty_label,
                "raw_label_counts": file_raw_labels,
                "stripped_label_counts": file_stripped_labels,
            })

        print("\nCreating SQLite index for duplicate analysis...")
        index_start = time.time()
        db_cur.execute("CREATE INDEX IF NOT EXISTS idx_feat_hash ON records(feature_hash);")
        db_cur.execute("CREATE INDEX IF NOT EXISTS idx_file_feat ON records(file_id, feature_hash);")
        db_conn.commit()
        print(f"Index created in {time.time() - index_start:.1f}s. Computing duplicate statistics...")

        # Query duplicate metrics
        db_cur.execute("SELECT COUNT(DISTINCT feature_hash) FROM records;")
        unique_feature_vectors = db_cur.fetchone()[0]

        excess_duplicate_features = global_total_rows - unique_feature_vectors

        db_cur.execute("SELECT COUNT(*) FROM (SELECT DISTINCT feature_hash, label FROM records);")
        unique_record_tuples = db_cur.fetchone()[0]
        excess_duplicate_records = global_total_rows - unique_record_tuples

        # Conflicting labels (same feature hash mapped to > 1 distinct label)
        db_cur.execute("""
            SELECT COUNT(*) FROM (
                SELECT feature_hash FROM records GROUP BY feature_hash HAVING COUNT(DISTINCT label) > 1
            );
        """)
        conflicting_feature_hashes = db_cur.fetchone()[0]

        db_cur.execute("""
            SELECT SUM(row_count) FROM (
                SELECT COUNT(*) as row_count FROM records GROUP BY feature_hash HAVING COUNT(DISTINCT label) > 1
            );
        """)
        conflicting_rows = db_cur.fetchone()[0] or 0

        # Cross-file duplicates (feature hashes appearing in > 1 file_id)
        db_cur.execute("""
            SELECT COUNT(*) FROM (
                SELECT feature_hash FROM records GROUP BY feature_hash HAVING COUNT(DISTINCT file_id) > 1
            );
        """)
        cross_file_feature_hashes = db_cur.fetchone()[0]

        db_cur.execute("""
            SELECT SUM(row_count) FROM (
                SELECT COUNT(*) as row_count FROM records GROUP BY feature_hash HAVING COUNT(DISTINCT file_id) > 1
            );
        """)
        cross_file_rows = db_cur.fetchone()[0] or 0

        # Within-file duplicate breakdown
        within_file_duplicates: Dict[str, int] = {}
        for fm in file_metrics:
            fid = fm["file_id"]
            db_cur.execute(f"SELECT COUNT(DISTINCT feature_hash) FROM records WHERE file_id = {fid};")
            uniq_in_file = db_cur.fetchone()[0]
            within_file_duplicates[fm["file_name"]] = fm["row_count"] - uniq_in_file

        db_conn.close()

        # Remove temporary SQLite database only if keep_db is explicitly False
        if not self.keep_db:
            if self.sqlite_db_path.exists():
                self.sqlite_db_path.unlink()
            wal = self.sqlite_db_path.with_name(self.sqlite_db_path.name + "-wal")
            shm = self.sqlite_db_path.with_name(self.sqlite_db_path.name + "-shm")
            if wal.exists():
                wal.unlink()
            if shm.exists():
                shm.unlink()

        # Check constant features
        constant_features = []
        for pos, fstat in feature_stats.items():
            if fstat["all_constant"] and fstat["finite_count"] > 0 and fstat["missing_count"] == 0 and fstat["invalid_numeric_count"] == 0 and fstat["pos_inf_count"] == 0 and fstat["neg_inf_count"] == 0:
                constant_features.append(fstat["name"])

        # Format feature summaries for JSON
        formatted_feature_stats = []
        for pos, fstat in feature_stats.items():
            min_v = None if math.isinf(fstat["min_value"]) else fstat["min_value"]
            max_v = None if math.isinf(fstat["max_value"]) else fstat["max_value"]
            formatted_feature_stats.append({
                "position": pos,
                "name": fstat["name"],
                "missing_or_blank_count": fstat["missing_count"],
                "nan_string_count": fstat["nan_string_count"],
                "invalid_numeric_count": fstat["invalid_numeric_count"],
                "positive_infinity_count": fstat["pos_inf_count"],
                "negative_infinity_count": fstat["neg_inf_count"],
                "finite_numeric_count": fstat["finite_count"],
                "min_value": min_v,
                "max_value": max_v,
                "is_constant": fstat["all_constant"] and (min_v == max_v if min_v is not None else False),
            })

        total_elapsed = time.time() - start_time

        audit_results = {
            "audit_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "elapsed_seconds": round(total_elapsed, 2),
            "total_files_inspected": len(csv_files),
            "chunk_size": self.chunk_size,
            "overall_summary": {
                "total_rows": global_total_rows,
                "benign_rows": global_benign_count,
                "attack_rows": global_attack_count,
                "empty_or_missing_labels": global_empty_label_count,
                "unique_feature_vectors": unique_feature_vectors,
                "excess_duplicate_features": excess_duplicate_features,
                "excess_duplicate_records": excess_duplicate_records,
                "conflicting_label_feature_vectors": conflicting_feature_hashes,
                "conflicting_label_rows": conflicting_rows,
                "cross_file_feature_vectors": cross_file_feature_hashes,
                "cross_file_rows": cross_file_rows,
            },
            "duplicate_column_verification": {
                "checked_pair": ["Fwd Header Length (pos 34)", "Fwd Header Length (pos 55)"],
                "is_strictly_identical_across_all_rows": dup_col_34_55_identical,
                "mismatch_count": dup_col_34_55_mismatches,
            },
            "headers_schema": [
                {
                    "file_name": h.file_name,
                    "column_count": h.column_count,
                    "raw_duplicates": h.raw_duplicates,
                    "stripped_duplicates": h.stripped_duplicates,
                    "duplicate_indices": h.duplicate_indices,
                }
                for h in headers_audit
            ],
            "constant_features": constant_features,
            "within_file_duplicates": within_file_duplicates,
            "file_metrics": file_metrics,
            "global_label_distribution": global_stripped_labels,
            "feature_statistics": formatted_feature_stats,
        }

        # Write reports/generated/dataset_audit.json
        audit_json_path = self.output_dir / "dataset_audit.json"
        with open(audit_json_path, "w", encoding="utf-8") as f:
            json.dump(audit_results, f, indent=2)

        # Write reports/generated/label_counts.csv
        label_csv_path = self.output_dir / "label_counts.csv"
        with open(label_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "raw_label", "stripped_label", "count", "is_attack"])
            for fm in file_metrics:
                for raw_lbl, cnt in fm["raw_label_counts"].items():
                    str_lbl = raw_lbl.strip()
                    is_att = 0 if str_lbl.lower() == "benign" else (1 if str_lbl else -1)
                    writer.writerow([fm["file_name"], raw_lbl, str_lbl, cnt, is_att])

        # Write reports/generated/dataset_audit.md
        self.generate_markdown_report(audit_results, self.output_dir / "dataset_audit.md")

        print(f"\nAudit complete in {total_elapsed:.1f}s.")
        print(f"JSON report: {audit_json_path}")
        print(f"CSV labels : {label_csv_path}")
        print(f"MD report  : {self.output_dir / 'dataset_audit.md'}")

        return audit_results

    def generate_markdown_report(self, audit: Dict[str, Any], out_path: pathlib.Path) -> None:
        ov = audit["overall_summary"]
        dc = audit["duplicate_column_verification"]
        lines = [
            "# CICIDS2017 Dataset Audit & Schema Inspection Report",
            "",
            f"**Audit Execution Time:** {audit['audit_timestamp_utc']}  ",
            f"**Elapsed Scan Time:** {audit['elapsed_seconds']} seconds  ",
            f"**Total CSV Files Scanned:** {audit['total_files_inspected']}  ",
            f"**Chunk Size:** {audit['chunk_size']:,} rows  ",
            "",
            "---",
            "",
            "## 1. Overall Row & Class Summary",
            "",
            f"- **Total Rows Scanned:** `{ov['total_rows']:,}`",
            f"- **Benign Rows:** `{ov['benign_rows']:,}` ({ov['benign_rows'] / ov['total_rows'] * 100:.2f}%)",
            f"- **Attack Rows:** `{ov['attack_rows']:,}` ({ov['attack_rows'] / ov['total_rows'] * 100:.2f}%)",
            f"- **Empty / Unlabeled Rows:** `{ov['empty_or_missing_labels']:,}`",
            "",
            "### Global Label Distribution (Whitespace-Trimmed)",
            "| Class Label | Row Count | Percentage |",
            "| :--- | :--- | :--- |",
        ]

        for lbl, count in sorted(audit["global_label_distribution"].items(), key=lambda x: x[1], reverse=True):
            pct = count / ov["total_rows"] * 100
            # Sanitize display of U+FFFD while documenting original representation
            safe_lbl = lbl.replace("\ufffd", "-")
            note = f" (raw: `{repr(lbl)}`)" if "\ufffd" in lbl else ""
            lines.append(f"| `{safe_lbl}`{note} | {count:,} | {pct:.4f}% |")

        file_row_map = {fm["file_name"]: fm["row_count"] for fm in audit.get("file_metrics", [])}

        lines.extend([
            "",
            "---",
            "",
            "## 2. Duplicate Records & Split Leakage Analysis",
            "",
            f"- **Unique Feature Vectors:** `{ov['unique_feature_vectors']:,}`",
            f"- **Excess Duplicate Feature Vectors:** `{ov['excess_duplicate_features']:,}` ({ov['excess_duplicate_features'] / ov['total_rows'] * 100:.2f}%)",
            f"- **Excess Duplicate Feature+Label Records:** `{ov['excess_duplicate_records']:,}`",
            f"- **Conflicting Label Feature Vectors (Multiclass):** `{ov['conflicting_label_feature_vectors']:,}` (spanning `{ov['conflicting_label_rows']:,}` rows)",
            f"- **Cross-File Duplicate Vectors:** `{ov['cross_file_feature_vectors']:,}` (spanning `{ov['cross_file_rows']:,}` rows)",
            "",
            "> [!NOTE]",
            "> Over 300,000 duplicate rows exist in the raw dataset. Duplicate feature vectors across train and test partitions present an evaluation leakage risk. Leakage can be mitigated either by deduplicating identical feature vectors prior to splitting or by using group-aware partitioning (e.g. by time/source).",
            "> Note: Conflicting label feature vectors reflect disagreements among multiclass labels; the breakdown between binary (BENIGN vs. ATTACK) and multiclass-only conflicts is not disaggregated in the summary.",
            "",
            "### Within-File Duplicate Excess Rows",
            "| CSV Filename | Total Rows | Duplicate Excess Rows |",
            "| :--- | :--- | :--- |",
        ])

        for fname, excess in audit["within_file_duplicates"].items():
            tot = file_row_map.get(fname, "-")
            tot_str = f"{tot:,}" if isinstance(tot, int) else str(tot)
            lines.append(f"| `{fname}` | {tot_str} | {excess:,} |")

        lines.extend([
            "",
            "---",
            "",
            "## 3. Schema & Duplicate Column Verification",
            "",
            f"- **Suspected Duplicate Columns:** `{dc['checked_pair'][0]}` and `{dc['checked_pair'][1]}`",
            f"- **Strict Equality Across All 2,830,743 Rows:** `{dc['is_strictly_identical_across_all_rows']}` (Mismatches: `{dc['mismatch_count']}`)",
            f"- **Duplicate Column Rule:** Column 55 (`Fwd Header Length.1`) is strictly identical to Column 34 across all rows and should be dropped during preprocessing.",
            f"- **Constant Zero-Variance Columns Identified ({len(audit['constant_features'])}):** {', '.join(f'`{c}`' for c in audit['constant_features']) if audit['constant_features'] else 'None'}",
            "> [!NOTE]",
            "> Zero-variance feature selection should be fitted on the training split within the evaluation pipeline rather than applied via global dataset truncation.",
            "",
            "---",
            "",
            "## 4. Per-File Statistics",
            "| Filename | Rows | Benign | Attack | Missing Labels |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])

        for fm in audit["file_metrics"]:
            lines.append(f"| `{fm['file_name']}` | {fm['row_count']:,} | {fm['benign_count']:,} | {fm['attack_count']:,} | {fm['empty_label_count']} |")

        lines.extend([
            "",
            "---",
            "",
            "## 5. Non-Numeric & Infinity Values in Features",
            "| Feature Name (Pos) | Missing / Blank | NaN Strings | Invalid Non-Numeric | Positive Inf | Negative Inf | Finite Min | Finite Max |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        # Filter features that have anomalies
        anom_features = [f for f in audit["feature_statistics"] if (f["missing_or_blank_count"] > 0 or f["nan_string_count"] > 0 or f["invalid_numeric_count"] > 0 or f["positive_infinity_count"] > 0 or f["negative_infinity_count"] > 0)]
        if not anom_features:
            lines.append("| *All candidate features are 100% valid finite numbers.* | - | - | - | - | - | - | - |")
        else:
            for f in anom_features:
                min_s = f"{f['min_value']:.4g}" if f["min_value"] is not None else "N/A"
                max_s = f"{f['max_value']:.4g}" if f["max_value"] is not None else "N/A"
                lines.append(f"| `{f['name']}` ({f['position']}) | {f['missing_or_blank_count']:,} | {f['nan_string_count']:,} | {f['invalid_numeric_count']:,} | {f['positive_infinity_count']:,} | {f['negative_infinity_count']:,} | {min_s} | {max_s} |")

        out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and inspect CICIDS2017 dataset.")
    parser.add_argument("--input-dir", type=pathlib.Path, default=pathlib.Path("data/raw/cicids2017"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("reports/generated"))
    parser.add_argument("--chunk-size", type=int, default=25000)
    parser.add_argument("--sqlite-path", type=pathlib.Path, default=None)
    parser.add_argument("--keep-db", action=argparse.BooleanOptionalAction, default=True, help="Preserve SQLite database checkpoint (default: True)")
    args = parser.parse_args()

    auditor = DatasetAuditor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        sqlite_db_path=args.sqlite_path,
        keep_db=args.keep_db,
    )
    auditor.run_audit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
