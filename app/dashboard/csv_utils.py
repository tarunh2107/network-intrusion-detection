"""CSV parsing, inspection, and schema-conformance validation for flow analysis.

Phase 8 Requirements:
- Inspect original raw headers before pandas can silently rename duplicates.
- Strip surrounding header whitespace.
- Reject duplicate column names and normalization collisions.
- Require exactly the model's named 77 feature columns (reject missing and extra).
- Validate the entire upload before any prediction request is made.
- Convert blank / NaN cells to Python None (JSON null).
- Reject non-numeric values and infinities with exact row (1-indexed data row) and column errors.
- Enforce 10 MB size cap and 1,000 data rows cap.
"""

from __future__ import annotations

import csv
import io
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


class CsvValidationError(Exception):
    """Raised when uploaded CSV fails validation contract."""

    def __init__(self, errors: List[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def inspect_and_validate_csv(
    file_bytes_or_text: bytes | str,
    expected_feature_names: Sequence[str],
    max_rows: int = 1000,
    max_size_bytes: int = 10 * 1024 * 1024,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, List[str]]:
    """Validate uploaded CSV against the model feature schema.

    Args:
        file_bytes_or_text: Raw uploaded CSV content as bytes or text string.
        expected_feature_names: Ordered list of 77 feature names from /model/info.
        max_rows: Maximum allowed data rows (default: 1000).
        max_size_bytes: Maximum allowed byte size (default: 10 MB).

    Returns:
        Tuple of:
        - payload_flows: List of flow dicts formatted for /predict/batch (`[{"features": {...}}, ...]`).
        - preview_df: Pandas DataFrame suitable for UI preview.
        - validation_errors: List of human-readable error messages (empty if valid).
    """
    errors: List[str] = []

    # 1. Byte size check
    if isinstance(file_bytes_or_text, bytes):
        if len(file_bytes_or_text) > max_size_bytes:
            return [], pd.DataFrame(), [
                f"File size ({len(file_bytes_or_text) / (1024*1024):.2f} MB) exceeds maximum allowed cap of {max_size_bytes / (1024*1024):.0f} MB."
            ]
        try:
            text_content = file_bytes_or_text.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text_content = file_bytes_or_text.decode("latin-1")
            except Exception as exc:
                return [], pd.DataFrame(), [f"Failed decoding CSV file encoding: {exc}"]
    else:
        text_content = file_bytes_or_text

    if not text_content.strip():
        return [], pd.DataFrame(), ["Uploaded file is empty."]

    # 2. Inspect raw headers before Pandas renames duplicates
    stream = io.StringIO(text_content)
    reader = csv.reader(stream)

    try:
        raw_header_row = next(reader)
    except StopIteration:
        return [], pd.DataFrame(), ["CSV file contains no header row."]

    # Strip surrounding whitespace from header names
    stripped_headers = [h.strip() for h in raw_header_row]

    # Check for empty header strings
    if any(h == "" for h in stripped_headers):
        errors.append("Header contains one or more blank column names.")

    # Check for duplicates in raw stripped headers
    seen_headers = set()
    duplicate_headers = set()
    for h in stripped_headers:
        if h in seen_headers:
            duplicate_headers.add(h)
        seen_headers.add(h)

    if duplicate_headers:
        errors.append(f"Duplicate column names detected in header: {sorted(list(duplicate_headers))}")

    # Check for normalization collisions (case-insensitive / collapsed whitespace collisions)
    normalized_map: Dict[str, str] = {}
    norm_collisions = set()
    for h in stripped_headers:
        norm = " ".join(h.lower().split())
        if norm in normalized_map and normalized_map[norm] != h:
            norm_collisions.add(f"'{h}' conflicts with '{normalized_map[norm]}'")
        else:
            normalized_map[norm] = h

    if norm_collisions:
        errors.append(f"Header normalization collisions detected: {sorted(list(norm_collisions))}")

    # Compare with expected 77 features
    expected_set = set(expected_feature_names)
    uploaded_set = set(stripped_headers)

    missing_cols = [f for f in expected_feature_names if f not in uploaded_set]
    extra_cols = [h for h in stripped_headers if h not in expected_set]

    if missing_cols:
        errors.append(
            f"Missing required feature columns ({len(missing_cols)}): {missing_cols[:5]}"
            + (f" ... and {len(missing_cols) - 5} more" if len(missing_cols) > 5 else "")
        )

    if extra_cols:
        errors.append(
            f"Unrecognized extra columns detected ({len(extra_cols)}): {extra_cols[:5]}"
            + (f" ... and {len(extra_cols) - 5} more" if len(extra_cols) > 5 else "")
        )

    # If header validation failed, stop here before reading rows
    if errors:
        return [], pd.DataFrame(), errors

    # 3. Read data rows with strict cell checking
    data_rows: List[List[str]] = []
    for line_idx, row in enumerate(reader, start=1):
        if not row or (len(row) == 1 and row[0].strip() == ""):
            continue  # skip completely empty lines without delimiters
        if len(row) != len(stripped_headers):
            errors.append(
                f"Row {line_idx}: Column count mismatch (expected {len(stripped_headers)}, found {len(row)})."
            )
            if len(errors) >= 20:
                break
        data_rows.append(row)

    if not data_rows and not errors:
        errors.append("CSV contains headers but no data rows.")
    elif len(data_rows) > max_rows:
        errors.append(
            f"CSV exceeds maximum allowed limit of {max_rows} data rows (found {len(data_rows)} rows)."
        )

    if errors:
        return [], pd.DataFrame(), errors

    # 4. Cell value validation & JSON null conversion
    payload_flows: List[Dict[str, Any]] = []
    preview_records: List[Dict[str, Any]] = []

    for row_idx, row in enumerate(data_rows, start=1):
        row_features: Dict[str, Any] = {}
        row_has_at_least_one_numeric = False

        for col_name, cell_str in zip(stripped_headers, row):
            cleaned_str = cell_str.strip()

            # Blank, empty string, or explicit NaN string -> null (Python None)
            if cleaned_str == "" or cleaned_str.lower() in ("nan", "null", "none"):
                row_features[col_name] = None
                continue

            # Reject non-finite string representations like "inf", "-inf", "infinity"
            if cleaned_str.lower() in ("inf", "+inf", "-inf", "infinity", "-infinity"):
                errors.append(
                    f"Row {row_idx}, column '{col_name}': Non-finite value '{cleaned_str}' is not permitted."
                )
                if len(errors) >= 25:
                    break
                continue

            # Parse numeric value
            try:
                num_val = float(cleaned_str)
            except ValueError:
                errors.append(
                    f"Row {row_idx}, column '{col_name}': Non-numeric value '{cleaned_str}' cannot be converted to float."
                )
                if len(errors) >= 25:
                    break
                continue

            if not math.isfinite(num_val):
                errors.append(
                    f"Row {row_idx}, column '{col_name}': Non-finite number '{cleaned_str}' is not permitted."
                )
                if len(errors) >= 25:
                    break
                continue

            row_features[col_name] = num_val
            row_has_at_least_one_numeric = True

        if len(errors) >= 25:
            break

        if not row_has_at_least_one_numeric:
            errors.append(f"Row {row_idx}: Row contains entirely null/empty values.")

        payload_flows.append({"features": row_features})
        if len(preview_records) < 50:
            preview_records.append(row_features)

    if errors:
        return [], pd.DataFrame(), errors

    preview_df = pd.DataFrame(preview_records)
    return payload_flows, preview_df, []


def generate_header_template_csv(feature_names: Sequence[str]) -> str:
    """Generate a clean CSV header template string containing the 77 model features."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(list(feature_names))
    return out.getvalue()
