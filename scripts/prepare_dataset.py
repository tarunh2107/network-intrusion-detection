#!/usr/bin/env python3
"""CICIDS2017 Deterministic Data Preparation & Leakage-Aware Splitting Pipeline.

Reads raw CSV files sequentially in chunks of 25,000 rows.
Validates headers, verifies duplicate Fwd Header Length identity, drops column 55.
Maps labels, converts NaNs/infinities and negative rates to missing values.
Computes deterministic SHA-256 fingerprints over cleaned numeric feature vectors.
Assigns rows to train (70%), validation (15%), or test (15%) partitions using a
hash-based bucket rule, guaranteeing that identical fingerprints never cross partitions.
Writes bounded-size CSV.gz shards under data/processed/v1/{train,validation,test}/.
Generates comprehensive reports, feature lists, and manifest checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from app.ml.preprocess import (
    CANONICAL_NAN_UINT64,
    CANONICAL_ZERO_UINT64,
    EXPECTED_RAW_COLUMNS,
    METADATA_COLUMNS,
    MODEL_FEATURE_COUNT,
    MODEL_FEATURE_NAMES,
    RAW_COLUMN_COUNT,
    assign_partition,
    clean_chunk_dataframe,
    compute_fingerprint_for_row,
    compute_fingerprints_batch,
    inspect_and_validate_header,
)


class ShardWriter:
    """Manages bounded-size rotating gzipped CSV shard files for a partition."""

    def __init__(
        self,
        partition_dir: pathlib.Path,
        file_stem: str,
        partition_name: str,
        max_shard_rows: int = 100000,
    ):
        self.partition_dir = partition_dir
        self.file_stem = file_stem
        self.partition_name = partition_name
        self.max_shard_rows = max_shard_rows
        self.shard_index = 0
        self.current_rows = 0
        self.current_file = None
        self.current_writer = None
        self.current_part_path: Optional[pathlib.Path] = None
        self.current_final_path: Optional[pathlib.Path] = None
        self.completed_shards: List[Dict[str, Any]] = []

    def _open_new_shard(self, header: List[str]) -> None:
        shard_name = f"{self.file_stem}_{self.partition_name}_shard_{self.shard_index:04d}.csv.gz"
        self.current_final_path = self.partition_dir / shard_name
        self.current_part_path = self.partition_dir / (shard_name + ".part")
        self.current_file = gzip.open(
            self.current_part_path, "wt", encoding="utf-8", newline="", compresslevel=6
        )
        self.current_writer = csv.writer(self.current_file)
        self.current_writer.writerow(header)
        self.current_rows = 0

    def write_row(self, row: List[Any], header: List[str]) -> None:
        if self.current_writer is None:
            self._open_new_shard(header)
        elif self.current_rows >= self.max_shard_rows:
            self.close_current_shard()
            self.shard_index += 1
            self._open_new_shard(header)

        self.current_writer.writerow(row)
        self.current_rows += 1

    def close_current_shard(self) -> None:
        if self.current_file is not None:
            self.current_file.close()
            if self.current_part_path is not None and self.current_part_path.exists():
                if self.current_rows > 0:
                    assert self.current_final_path is not None
                    self.current_part_path.replace(self.current_final_path)
                    size = self.current_final_path.stat().st_size
                    hasher = hashlib.sha256()
                    with open(self.current_final_path, "rb") as f:
                        while chunk := f.read(1024 * 1024):
                            hasher.update(chunk)
                    self.completed_shards.append({
                        "filename": self.current_final_path.name,
                        "partition": self.partition_name,
                        "rows": self.current_rows,
                        "size_bytes": size,
                        "sha256": hasher.hexdigest(),
                    })
                else:
                    self.current_part_path.unlink()
            self.current_file = None
            self.current_writer = None
            self.current_part_path = None
            self.current_final_path = None


class DatasetPreparer:
    """Orchestrates deterministic dataset cleaning, partition routing, and verification."""

    def __init__(
        self,
        raw_dir: pathlib.Path,
        output_dir: pathlib.Path,
        reports_dir: pathlib.Path,
        chunk_size: int = 25000,
        shard_size: int = 100000,
        seed: int = 42,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        force: bool = False,
    ):
        self.raw_dir = pathlib.Path(raw_dir)
        self.output_dir = pathlib.Path(output_dir)
        self.reports_dir = pathlib.Path(reports_dir)
        self.chunk_size = chunk_size
        self.shard_size = shard_size
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.force = force

        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "validation"
        self.test_dir = self.output_dir / "test"
        self.checkpoint_file = self.output_dir / "checkpoint.json"

        self.full_header = METADATA_COLUMNS + MODEL_FEATURE_NAMES

    def compute_code_hash(self) -> str:
        """Compute SHA-256 of preprocess.py and this script to ensure version fidelity."""
        h = hashlib.sha256()
        p1 = pathlib.Path(__file__).resolve().parent.parent / "app" / "ml" / "preprocess.py"
        p2 = pathlib.Path(__file__).resolve()
        if p1.exists():
            h.update(p1.read_bytes())
        if p2.exists():
            h.update(p2.read_bytes())
        return h.hexdigest()

    def load_checkpoint(self) -> Dict[str, Any]:
        if not self.force and self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Verify configuration compatibility
                if (
                    data.get("seed") == self.seed
                    and data.get("train_ratio") == self.train_ratio
                    and data.get("val_ratio") == self.val_ratio
                    and data.get("code_hash") == self.compute_code_hash()
                ):
                    return data
            except Exception as err:
                print(f"[WARN] Failed to load existing checkpoint: {err}. Starting clean.")
        return {
            "code_hash": self.compute_code_hash(),
            "seed": self.seed,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "completed_files": {},
        }

    def save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        temp_file = self.checkpoint_file.with_suffix(".json.tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)
        temp_file.replace(self.checkpoint_file)

    def process_file(
        self,
        csv_path: pathlib.Path,
        file_index: int,
        total_files: int,
        checkpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process a single raw CSV file in chunks and stream into partition shards."""
        file_start = time.time()
        file_name = csv_path.name
        file_stem = csv_path.stem

        print(f"\n[{file_index}/{total_files}] Processing {file_name}...")

        # Initialize partition shard writers
        writers = {
            "train": ShardWriter(self.train_dir, file_stem, "train", self.shard_size),
            "validation": ShardWriter(self.val_dir, file_stem, "validation", self.shard_size),
            "test": ShardWriter(self.test_dir, file_stem, "test", self.shard_size),
        }

        # Validate header before pandas
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            raw_header = next(reader)
        _ = inspect_and_validate_header(raw_header)

        # Stream chunks with pandas
        total_inspected = 0
        total_valid = 0
        total_rejected = 0
        split_counts = {"train": 0, "validation": 0, "test": 0}
        nan_converted = 0
        inf_converted = 0
        neg_bytes_converted = 0
        neg_pkts_converted = 0
        invalid_numerics = 0

        chunk_reader = pd.read_csv(
            csv_path,
            header=None,
            skiprows=1,
            chunksize=self.chunk_size,
            dtype=str,
            encoding="utf-8",
            on_bad_lines="error",
        )

        last_flush_rows = 0

        for chunk_idx, df_chunk in enumerate(chunk_reader):
            start_row_num = total_inspected + 1
            chunk_res = clean_chunk_dataframe(
                df_chunk,
                source_file=file_name,
                start_row_number=start_row_num,
                validate_column_equality=True,
            )

            # Accumulate cleaning stats
            c_stats = chunk_res.stats
            total_inspected += c_stats.total_rows_inspected
            total_valid += c_stats.valid_rows_emitted
            total_rejected += c_stats.rejected_rows_count
            nan_converted += c_stats.nan_strings_converted
            inf_converted += c_stats.infinities_converted
            neg_bytes_converted += c_stats.negative_flow_bytes_converted
            neg_pkts_converted += c_stats.negative_flow_packets_converted
            invalid_numerics += c_stats.invalid_numerics_converted

            # Route rows to partition writers
            feats_mat = chunk_res.features
            meta_list = chunk_res.metadata

            for r_idx in range(len(meta_list)):
                meta = meta_list[r_idx]
                fp = meta["fingerprint"]
                partition = assign_partition(
                    fp,
                    seed=self.seed,
                    train_ratio=self.train_ratio,
                    val_ratio=self.val_ratio,
                )
                split_counts[partition] += 1

                # Format feature vector for CSV (empty string for NaNs, full float precision)
                f_row = feats_mat[r_idx]
                f_formatted = ["" if np.isnan(v) else str(v) for v in f_row]

                full_row = [
                    meta["source_file"],
                    meta["source_row_number"],
                    meta["raw_label"],
                    meta["display_label"],
                    meta["binary_target"],
                    meta["fingerprint"],
                ] + f_formatted

                writers[partition].write_row(full_row, self.full_header)

            # Periodic flushed progress update
            if total_inspected - last_flush_rows >= 100000:
                elapsed = time.time() - file_start
                rate = total_inspected / elapsed if elapsed > 0 else 0
                print(
                    f"  [PROGRESS] {file_name}: {total_inspected:,} rows "
                    f"({rate:.0f} rows/s) | Train: {split_counts['train']:,}, "
                    f"Val: {split_counts['validation']:,}, Test: {split_counts['test']:,}",
                    flush=True,
                )
                last_flush_rows = total_inspected

        # Close all shard writers
        all_shards = []
        for part_name, writer in writers.items():
            writer.close_current_shard()
            all_shards.extend(writer.completed_shards)

        file_elapsed = time.time() - file_start
        rate = total_inspected / file_elapsed if file_elapsed > 0 else 0
        print(
            f"  [DONE] {file_name}: {total_inspected:,} rows in {file_elapsed:.1f}s ({rate:.0f} rows/s)\n"
            f"         Outputs -> Train: {split_counts['train']:,}, "
            f"Val: {split_counts['validation']:,}, Test: {split_counts['test']:,} | "
            f"Rejected: {total_rejected:,}",
            flush=True,
        )

        file_summary = {
            "file_name": file_name,
            "elapsed_seconds": round(file_elapsed, 2),
            "rows_inspected": total_inspected,
            "valid_rows_emitted": total_valid,
            "rejected_rows_count": total_rejected,
            "split_counts": split_counts,
            "shards": all_shards,
            "cleaning_anomalies": {
                "nan_strings_converted": nan_converted,
                "infinities_converted": inf_converted,
                "negative_flow_bytes_converted": neg_bytes_converted,
                "negative_flow_packets_converted": neg_pkts_converted,
                "invalid_numerics_converted": invalid_numerics,
            },
        }

        # Record in checkpoint
        checkpoint["completed_files"][file_name] = file_summary
        self.save_checkpoint(checkpoint)

        return file_summary

    def verify_outputs_and_disjointness(
        self,
        checkpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform streaming verification: no fingerprint crosses partition boundaries."""
        print("\n" + "=" * 70)
        print("VERIFICATION: Validating Partition Fingerprint Disjointness...")
        print("=" * 70, flush=True)

        t0 = time.time()

        # Collect set of 64-bit fingerprint prefixes per partition
        # Storing 64-bit integer hashes for 400k fingerprints requires only ~3 MB of RAM
        val_fps: Set[int] = set()
        test_fps: Set[int] = set()

        val_rows = 0
        test_rows = 0
        train_rows = 0

        # Detailed class distribution trackers
        partition_class_counts: Dict[str, Dict[str, int]] = {
            "train": {},
            "validation": {},
            "test": {},
        }
        partition_binary_counts: Dict[str, Dict[int, int]] = {
            "train": {0: 0, 1: 0},
            "validation": {0: 0, 1: 0},
            "test": {0: 0, 1: 0},
        }

        # 1. Scan validation shards
        for shard_file in sorted(self.val_dir.glob("*.csv.gz")):
            with gzip.open(shard_file, "rt", encoding="utf-8") as f:
                reader = csv.reader(f)
                _ = next(reader)  # skip header
                for row in reader:
                    val_rows += 1
                    fp = row[5]
                    lbl = row[3]  # display_label
                    tgt = int(row[4])
                    val_fps.add(int(fp[:16], 16))
                    partition_class_counts["validation"][lbl] = (
                        partition_class_counts["validation"].get(lbl, 0) + 1
                    )
                    partition_binary_counts["validation"][tgt] = (
                        partition_binary_counts["validation"].get(tgt, 0) + 1
                    )

        # 2. Scan test shards
        for shard_file in sorted(self.test_dir.glob("*.csv.gz")):
            with gzip.open(shard_file, "rt", encoding="utf-8") as f:
                reader = csv.reader(f)
                _ = next(reader)  # skip header
                for row in reader:
                    test_rows += 1
                    fp = row[5]
                    lbl = row[3]  # display_label
                    tgt = int(row[4])
                    test_fps.add(int(fp[:16], 16))
                    partition_class_counts["test"][lbl] = (
                        partition_class_counts["test"].get(lbl, 0) + 1
                    )
                    partition_binary_counts["test"][tgt] = (
                        partition_binary_counts["test"].get(tgt, 0) + 1
                    )

        # Check val vs test disjointness
        val_test_overlap = len(val_fps.intersection(test_fps))

        # 3. Stream train shards and check collision against val and test
        train_val_overlap = 0
        train_test_overlap = 0
        rule_mismatches = 0

        for shard_file in sorted(self.train_dir.glob("*.csv.gz")):
            with gzip.open(shard_file, "rt", encoding="utf-8") as f:
                reader = csv.reader(f)
                _ = next(reader)
                for row in reader:
                    train_rows += 1
                    fp = row[5]
                    lbl = row[3]
                    tgt = int(row[4])
                    h_int = int(fp[:16], 16)
                    if h_int in val_fps:
                        train_val_overlap += 1
                    if h_int in test_fps:
                        train_test_overlap += 1

                    # Check assignment rule
                    assigned = assign_partition(
                        fp,
                        seed=self.seed,
                        train_ratio=self.train_ratio,
                        val_ratio=self.val_ratio,
                    )
                    if assigned != "train":
                        rule_mismatches += 1

                    partition_class_counts["train"][lbl] = (
                        partition_class_counts["train"].get(lbl, 0) + 1
                    )
                    partition_binary_counts["train"][tgt] = (
                        partition_binary_counts["train"].get(tgt, 0) + 1
                    )

        total_emitted = train_rows + val_rows + test_rows
        elapsed_verif = time.time() - t0

        print(f"Verified {total_emitted:,} emitted rows in {elapsed_verif:.1f}s:")
        print(f"  Train Rows     : {train_rows:,} ({train_rows / total_emitted * 100:.2f}%)")
        print(f"  Validation Rows: {val_rows:,} ({val_rows / total_emitted * 100:.2f}%)")
        print(f"  Test Rows      : {test_rows:,} ({test_rows / total_emitted * 100:.2f}%)")
        print(f"  Cross-Partition Collisions: Train-Val: {train_val_overlap}, Train-Test: {train_test_overlap}, Val-Test: {val_test_overlap}")
        print(f"  Partition Rule Mismatches : {rule_mismatches}")

        is_strictly_disjoint = (
            train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0
        )
        assert is_strictly_disjoint, "Disjointness violated! Partitions share identical fingerprints."
        assert rule_mismatches == 0, "Found rows in incorrect partition folder."

        return {
            "elapsed_seconds": round(elapsed_verif, 2),
            "total_emitted_rows": total_emitted,
            "train_rows": train_rows,
            "validation_rows": val_rows,
            "test_rows": test_rows,
            "is_strictly_disjoint": is_strictly_disjoint,
            "train_val_overlap_count": train_val_overlap,
            "train_test_overlap_count": train_test_overlap,
            "val_test_overlap_count": val_test_overlap,
            "rule_mismatches_count": rule_mismatches,
            "class_counts_by_partition": partition_class_counts,
            "binary_counts_by_partition": partition_binary_counts,
        }

    def generate_reports_and_metadata(
        self,
        checkpoint: Dict[str, Any],
        verif_results: Dict[str, Any],
        total_time: float,
    ) -> None:
        """Write features.json, metadata_columns.json, manifest.json, and audit reports."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save data/processed/v1/features.json
        features_json_path = self.output_dir / "features.json"
        with open(features_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "feature_count": MODEL_FEATURE_COUNT,
                "feature_names": MODEL_FEATURE_NAMES,
            }, f, indent=2)

        # 2. Save data/processed/v1/metadata_columns.json
        meta_json_path = self.output_dir / "metadata_columns.json"
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata_count": len(METADATA_COLUMNS),
                "metadata_columns": METADATA_COLUMNS,
            }, f, indent=2)

        # 3. Save data/processed/v1/label_mapping.json
        label_mapping_path = self.output_dir / "label_mapping.json"
        from app.ml.preprocess import LABEL_MAPPING
        with open(label_mapping_path, "w", encoding="utf-8") as f:
            json.dump({
                raw: {"display_label": disp, "binary_target": tgt}
                for raw, (disp, tgt) in LABEL_MAPPING.items()
            }, f, indent=2)

        # 4. Save data/processed/v1/config.json
        config_json_path = self.output_dir / "config.json"
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "pipeline_version": "v1",
                "seed": self.seed,
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": round(1.0 - self.train_ratio - self.val_ratio, 4),
                "chunk_size": self.chunk_size,
                "shard_size": self.shard_size,
                "code_hash": self.compute_code_hash(),
                "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }, f, indent=2)

        # 4. Save data/processed/v1/manifest.json
        all_shards = []
        for file_info in checkpoint["completed_files"].values():
            all_shards.extend(file_info.get("shards", []))

        total_disk_bytes = sum(s["size_bytes"] for s in all_shards)
        manifest_json_path = self.output_dir / "manifest.json"
        with open(manifest_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_shards": len(all_shards),
                "total_size_bytes": total_disk_bytes,
                "total_size_mb": round(total_disk_bytes / (1024**2), 2),
                "shards": all_shards,
            }, f, indent=2)

        # Aggregate cleaning anomaly totals
        cleaning_totals = {
            "nan_strings_converted": sum(
                f["cleaning_anomalies"]["nan_strings_converted"]
                for f in checkpoint["completed_files"].values()
            ),
            "infinities_converted": sum(
                f["cleaning_anomalies"]["infinities_converted"]
                for f in checkpoint["completed_files"].values()
            ),
            "negative_flow_bytes_converted": sum(
                f["cleaning_anomalies"]["negative_flow_bytes_converted"]
                for f in checkpoint["completed_files"].values()
            ),
            "negative_flow_packets_converted": sum(
                f["cleaning_anomalies"]["negative_flow_packets_converted"]
                for f in checkpoint["completed_files"].values()
            ),
            "invalid_numerics_converted": sum(
                f["cleaning_anomalies"]["invalid_numerics_converted"]
                for f in checkpoint["completed_files"].values()
            ),
        }

        total_inspected = sum(
            f["rows_inspected"] for f in checkpoint["completed_files"].values()
        )
        total_valid = sum(
            f["valid_rows_emitted"] for f in checkpoint["completed_files"].values()
        )
        total_rejected = sum(
            f["rejected_rows_count"] for f in checkpoint["completed_files"].values()
        )

        full_report_data = {
            "pipeline_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "elapsed_seconds": round(total_time, 2),
            "configuration": {
                "seed": self.seed,
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": round(1.0 - self.train_ratio - self.val_ratio, 4),
                "chunk_size": self.chunk_size,
                "shard_size": self.shard_size,
                "model_features_count": MODEL_FEATURE_COUNT,
            },
            "overall_counts": {
                "total_input_rows": total_inspected,
                "total_valid_rows_emitted": total_valid,
                "total_rejected_rows": total_rejected,
                "verification_row_balance_exact": (total_inspected == total_valid + total_rejected),
            },
            "partition_distribution": {
                "train_rows": verif_results["train_rows"],
                "train_pct": round(verif_results["train_rows"] / total_valid * 100, 2),
                "validation_rows": verif_results["validation_rows"],
                "validation_pct": round(verif_results["validation_rows"] / total_valid * 100, 2),
                "test_rows": verif_results["test_rows"],
                "test_pct": round(verif_results["test_rows"] / total_valid * 100, 2),
            },
            "disjointness_verification": {
                "is_strictly_disjoint": verif_results["is_strictly_disjoint"],
                "train_val_collisions": verif_results["train_val_overlap_count"],
                "train_test_collisions": verif_results["train_test_overlap_count"],
                "val_test_collisions": verif_results["val_test_overlap_count"],
            },
            "cleaning_anomalies_counts": cleaning_totals,
            "class_coverage_by_partition": verif_results["class_counts_by_partition"],
            "binary_counts_by_partition": verif_results["binary_counts_by_partition"],
            "disk_usage": {
                "total_shards": len(all_shards),
                "total_compressed_bytes": total_disk_bytes,
                "total_compressed_mb": round(total_disk_bytes / (1024**2), 2),
            },
            "per_file_summaries": checkpoint["completed_files"],
        }

        # 5. Save reports/generated/preprocessing_report.json
        report_json_path = self.reports_dir / "preprocessing_report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(full_report_data, f, indent=2)

        # 6. Save reports/generated/preprocessing_report.md
        report_md_path = self.reports_dir / "preprocessing_report.md"
        self._write_markdown_report(full_report_data, report_md_path)

        print(f"\nSaved Reports & Artifacts:")
        print(f"  JSON Report: {report_json_path}")
        print(f"  MD Report  : {report_md_path}")
        print(f"  Manifest   : {manifest_json_path}")
        print(f"  Features   : {features_json_path}")

    def _write_markdown_report(
        self,
        data: Dict[str, Any],
        out_path: pathlib.Path,
    ) -> None:
        ov = data["overall_counts"]
        pd_dist = data["partition_distribution"]
        disj = data["disjointness_verification"]
        anom = data["cleaning_anomalies_counts"]
        classes = data["class_coverage_by_partition"]
        binaries = data["binary_counts_by_partition"]
        disk = data["disk_usage"]

        # Gather all unique display classes
        all_classes = sorted(list(set(
            list(classes["train"].keys())
            + list(classes["validation"].keys())
            + list(classes["test"].keys())
        )))

        lines = [
            "# Phase 5: Deterministic Data Preparation & Splitting Report",
            "",
            f"**Execution Timestamp:** {data['pipeline_timestamp_utc']}  ",
            f"**Total Elapsed Time:** {data['elapsed_seconds']} seconds  ",
            f"**Model Features Retained:** `{data['configuration']['model_features_count']}`  ",
            f"**Metadata Fields Attached:** `{len(METADATA_COLUMNS)}`  ",
            "",
            "---",
            "",
            "## 1. Overall Row Accounting & Splitting Proportions",
            "",
            f"- **Total Input Raw Rows:** `{ov['total_input_rows']:,}`",
            f"- **Valid Processed Rows Emitted:** `{ov['total_valid_rows_emitted']:,}`",
            f"- **Explicitly Rejected Rows:** `{ov['total_rejected_rows']:,}`",
            f"- **Row Balance Reconciled (`Input == Emitted + Rejected`):** `{ov['verification_row_balance_exact']}`",
            "",
            "| Partition | Row Count | Target Ratio | Actual Proportion | Binary Benign (0) | Binary Attack (1) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
            f"| **Train** | `{pd_dist['train_rows']:,}` | 70.0% | `{pd_dist['train_rows'] / ov['total_valid_rows_emitted'] * 100:.2f}%` | `{binaries['train'].get(0, 0):,}` | `{binaries['train'].get(1, 0):,}` |",
            f"| **Validation** | `{pd_dist['validation_rows']:,}` | 15.0% | `{pd_dist['validation_rows'] / ov['total_valid_rows_emitted'] * 100:.2f}%` | `{binaries['validation'].get(0, 0):,}` | `{binaries['validation'].get(1, 0):,}` |",
            f"| **Test** | `{pd_dist['test_rows']:,}` | 15.0% | `{pd_dist['test_rows'] / ov['total_valid_rows_emitted'] * 100:.2f}%` | `{binaries['test'].get(0, 0):,}` | `{binaries['test'].get(1, 0):,}` |",
            f"| **Total** | `{ov['total_valid_rows_emitted']:,}` | 100.0% | 100.00% | `{binaries['train'].get(0, 0) + binaries['validation'].get(0, 0) + binaries['test'].get(0, 0):,}` | `{binaries['train'].get(1, 0) + binaries['validation'].get(1, 0) + binaries['test'].get(1, 0):,}` |",
            "",
            "---",
            "",
            "## 2. Partition Fingerprint Disjointness Verification",
            "",
            f"- **Cross-Partition Collisions (Train <-> Validation):** `{disj['train_val_collisions']}`",
            f"- **Cross-Partition Collisions (Train <-> Test):** `{disj['train_test_collisions']}`",
            f"- **Cross-Partition Collisions (Validation <-> Test):** `{disj['val_test_collisions']}`",
            f"- **Strict Disjointness Established:** `{disj['is_strictly_disjoint']}`",
            "",
            "> [!NOTE]",
            "> All rows sharing identical 77-element cleaned numeric feature vectors (fingerprints) were deterministically routed into the identical partition. No identical feature vector appears across train, validation, or test partitions.",
            "",
            "---",
            "",
            "## 3. Original Attack-Label Coverage Across Partitions",
            "",
            "| Display Label | Total Flows | Train Count | Validation Count | Test Count | Present in All Folds? |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for lbl in all_classes:
            tr_cnt = classes["train"].get(lbl, 0)
            val_cnt = classes["validation"].get(lbl, 0)
            tst_cnt = classes["test"].get(lbl, 0)
            tot_cnt = tr_cnt + val_cnt + tst_cnt
            in_all = (tr_cnt > 0 and val_cnt > 0 and tst_cnt > 0)
            lines.append(
                f"| `{lbl}` | {tot_cnt:,} | {tr_cnt:,} | {val_cnt:,} | {tst_cnt:,} | {'**Yes**' if in_all else 'No'} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 4. Cleaning Anomalies & Sentinel Handling",
            "",
            f"- **String NaNs (`NaN`, `null`, `none`, empty) converted to missing:** `{anom['nan_strings_converted']:,}`",
            f"- **Infinities (`+inf`, `-inf`, string infinity) converted to missing:** `{anom['infinities_converted']:,}`",
            f"- **Negative `Flow Bytes/s` values converted to missing:** `{anom['negative_flow_bytes_converted']:,}`",
            f"- **Negative `Flow Packets/s` values converted to missing:** `{anom['negative_flow_packets_converted']:,}`",
            f"- **Invalid non-numeric float tokens converted to missing:** `{anom['invalid_numerics_converted']:,}`",
            f"- **Sentinel preservation:** Non-rate negative values (e.g. `Init_Win_bytes_forward = -1`) were preserved intact without conversion.",
            f"- **Constant column status:** The 8 zero-variance features identified in Phase 4 were preserved in the processed shards to allow feature selection (`VarianceThreshold`) to be fitted strictly on training data in Phase 6.",
            "",
            "---",
            "",
            "## 5. Storage Metrics & Compressed Output Shards",
            "",
            f"- **Total Shards Generated:** `{disk['total_shards']}` files",
            f"- **Compressed Disk Usage:** `{disk['total_compressed_mb']:.2f} MB` (`{disk['total_compressed_bytes']:,}` bytes)",
            f"- **Output Location:** `{self.output_dir}`",
            "",
            "---",
            "",
            "## 6. Remaining Methodological Considerations for Phase 6",
            "",
            "1. **Offline Benchmark Split:** This hash-grouped partition constitutes an offline benchmark split designed to prevent duplicate vector leakage. It does not test temporal drift or unconstrained live-network generalization.",
            "2. **Rare Class Coverage:** Classes with very few independent flow groups (e.g. Heartbleed with 11 flows) may fall entirely or predominantly into one partition under grouped hashing.",
            "3. **Binary Conflict Handling:** Groups mapping to conflicting labels were preserved intact in their assigned partition. Explicit conflict-resolution policies (e.g. dropping ambiguous vectors or label smoothing) remain for evaluation in Phase 6.",
            "4. **Duplicate Weighting:** Repeated identical flows remain within their assigned partition. Training can either weight flows equally, deduplicate within-fold, or apply sample weights during model optimization.",
        ])

        out_path.write_text("\n".join(lines), encoding="utf-8")

    def run(self) -> Dict[str, Any]:
        overall_start = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        raw_files = sorted(list(self.raw_dir.glob("*.csv")))
        if not raw_files:
            raise FileNotFoundError(f"No CSV files found in {self.raw_dir}")

        checkpoint = self.load_checkpoint()
        completed = checkpoint.get("completed_files", {})

        print("=" * 70)
        print("CICIDS2017 Phase 5 Data Preparation & Splitting")
        print(f"Target Partitions: Train {self.train_ratio*100:.0f}%, Val {self.val_ratio*100:.0f}%, Test {(1-self.train_ratio-self.val_ratio)*100:.0f}% (Seed: {self.seed})")
        print(f"Chunk size: {self.chunk_size:,} | Shard size: {self.shard_size:,}")
        print("=" * 70)

        for idx, csv_path in enumerate(raw_files, start=1):
            if csv_path.name in completed and not self.force:
                print(f"[{idx}/{len(raw_files)}] Skipping completed file: {csv_path.name} (from checkpoint)")
                continue

            self.process_file(csv_path, idx, len(raw_files), checkpoint)

        # Verification step
        verif = self.verify_outputs_and_disjointness(checkpoint)

        # Reports and metadata
        total_time = time.time() - overall_start
        self.generate_reports_and_metadata(checkpoint, verif, total_time)

        print("\n" + "=" * 70)
        print(f"Phase 5 Complete in {total_time:.1f}s!")
        print("=" * 70)

        return verif


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and split CICIDS2017 dataset.")
    parser.add_argument("--raw-dir", type=pathlib.Path, default=pathlib.Path("data/raw/cicids2017"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("data/processed/v1"))
    parser.add_argument("--reports-dir", type=pathlib.Path, default=pathlib.Path("reports/generated"))
    parser.add_argument("--chunk-size", type=int, default=25000)
    parser.add_argument("--shard-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--force", action="store_true", help="Force rebuild from scratch")
    args = parser.parse_args()

    preparer = DatasetPreparer(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        chunk_size=args.chunk_size,
        shard_size=args.shard_size,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        force=args.force,
    )
    preparer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
