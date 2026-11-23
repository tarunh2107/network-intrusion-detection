#!/usr/bin/env python3
"""CICIDS2017 Dataset Acquisition, Verification, and Safe Extraction Script.

Downloads MachineLearningCSV.zip from verified academic mirror with streaming,
verifies ZIP magic bytes, tests archive integrity, validates disk space,
and extracts CSV files safely with path traversal protection.
"""

import datetime
import hashlib
import json
import os
import pathlib
import shutil
import ssl
import sys
import time
import urllib.request
import zipfile

SOURCE_URL = "https://huggingface.co/datasets/bencorn/CICIDS2017/resolve/main/csvs/MachineLearningCSV.zip"
RAW_DIR = pathlib.Path("data/raw")
TARGET_ARCHIVE = RAW_DIR / "MachineLearningCSV.zip"
PART_FILE = RAW_DIR / "MachineLearningCSV.zip.part"
EXTRACT_DIR = RAW_DIR / "cicids2017"
MANIFEST_PATH = pathlib.Path("reports/dataset_manifest.json")


def compute_sha256(file_path: pathlib.Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 checksum using chunked streaming reads."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_disk_space(required_bytes: int, target_path: pathlib.Path) -> bool:
    """Check if free disk space exceeds required bytes plus safety margin (1 GB)."""
    free_bytes = shutil.disk_usage(target_path).free
    safety_margin = 1024 * 1024 * 1024  # 1 GB safety margin
    print(f"Disk check: {free_bytes / (1024**3):.2f} GB free, required {required_bytes / (1024**3):.2f} GB (+1GB margin)")
    return free_bytes > (required_bytes + safety_margin)


def download_archive(url: str, target: pathlib.Path, part_file: pathlib.Path, max_retries: int = 3) -> None:
    """Stream download with retries, timeouts, and atomic rename."""
    if target.exists() and target.stat().st_size > 0:
        print(f"Target archive {target} already exists ({target.stat().st_size} bytes). Skipping download.")
        return

    ctx = ssl.create_default_context()
    ctx.load_default_certs()

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NIDS-Acquisition/1.0"}

    for attempt in range(1, max_retries + 1):
        print(f"Download attempt {attempt}/{max_retries} from {url}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Unexpected HTTP status {resp.status}")

                content_len = resp.headers.get("Content-Length")
                expected_size = int(content_len) if content_len else None
                print(f"Server reported Content-Length: {expected_size} bytes")

                downloaded = 0
                chunk_size = 1024 * 1024  # 1 MB chunk
                start_time = time.time()

                with open(part_file, "wb") as f_out:
                    # Check magic bytes on first chunk
                    first_chunk = resp.read(chunk_size)
                    if not first_chunk.startswith(b"PK\x03\x04"):
                        raise ValueError(f"Downloaded stream does not have ZIP magic bytes. Header: {first_chunk[:16]}")
                    f_out.write(first_chunk)
                    downloaded += len(first_chunk)

                    while chunk := resp.read(chunk_size):
                        f_out.write(chunk)
                        downloaded += len(chunk)
                        if expected_size and (downloaded % (20 * 1024 * 1024) < chunk_size):
                            pct = (downloaded / expected_size) * 100
                            mb = downloaded / (1024 * 1024)
                            total_mb = expected_size / (1024 * 1024)
                            elapsed = time.time() - start_time
                            speed = mb / elapsed if elapsed > 0 else 0
                            print(f"  Downloaded: {mb:.1f} MB / {total_mb:.1f} MB ({pct:.1f}%) at {speed:.2f} MB/s")

                if expected_size and downloaded != expected_size:
                    raise IOError(f"Incomplete download: expected {expected_size} bytes, got {downloaded}")

                # Atomic rename
                part_file.replace(target)
                print(f"Download complete: {target} ({downloaded} bytes)")
                return

        except Exception as exc:
            print(f"Attempt {attempt} failed: {exc}")
            if part_file.exists():
                part_file.unlink()
            if attempt < max_retries:
                wait_sec = attempt * 5
                print(f"Waiting {wait_sec} seconds before retry...")
                time.sleep(wait_sec)
            else:
                raise RuntimeError(f"Failed to download archive after {max_retries} attempts: {exc}") from exc


def main() -> int:
    print("=" * 70)
    print("CICIDS2017 Dataset Acquisition & Verification Pipeline")
    print("=" * 70)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download archive
    download_archive(SOURCE_URL, TARGET_ARCHIVE, PART_FILE)

    # 2. Compute archive SHA-256 and size
    archive_size = TARGET_ARCHIVE.stat().st_size
    print(f"Computing SHA-256 for archive {TARGET_ARCHIVE.name}...")
    archive_sha256 = compute_sha256(TARGET_ARCHIVE)
    print(f"Archive SHA-256: {archive_sha256}")
    print(f"Archive Size   : {archive_size:,} bytes ({archive_size / (1024**2):.2f} MB)")

    # 3. Inspect ZIP members & test archive integrity
    print("\nInspecting ZIP members and testing integrity...")
    with zipfile.ZipFile(TARGET_ARCHIVE, "r") as zf:
        corrupted = zf.testzip()
        if corrupted:
            raise ValueError(f"ZIP archive corrupted at member: {corrupted}")
        print("ZIP integrity test: PASSED (CRC-32 valid for all members)")

        members = zf.infolist()
        total_uncompressed = sum(m.file_size for m in members)
        print(f"Total members: {len(members)}, Total uncompressed size: {total_uncompressed:,} bytes ({total_uncompressed / (1024**2):.2f} MB)")

        # Disk space check
        if not check_disk_space(total_uncompressed, EXTRACT_DIR):
            raise OSError("Insufficient disk space to extract dataset safely!")

        # 4. Safe extraction into EXTRACT_DIR
        extracted_csv_files = []
        for info in members:
            # Filter directories
            if info.is_dir() or info.filename.endswith("/"):
                continue

            # Path traversal & security checks
            orig_path = pathlib.PurePosixPath(info.filename)
            if orig_path.is_absolute() or ".." in orig_path.parts:
                raise ValueError(f"Path traversal detected in ZIP member: {info.filename}")

            filename = orig_path.name
            if not filename.lower().endswith(".csv"):
                print(f"Skipping non-CSV member: {info.filename}")
                continue

            target_file = EXTRACT_DIR / filename
            print(f"Extracting: {filename} ({info.file_size:,} bytes)...")

            with zf.open(info) as src, open(target_file, "wb") as dst:
                shutil.copyfileobj(src, dst)

            extracted_csv_files.append((filename, target_file))

    print(f"\nSuccessfully extracted {len(extracted_csv_files)} CSV files into {EXTRACT_DIR}")

    # 5. Compute manifest entries
    manifest_csvs = []
    for fname, fpath in sorted(extracted_csv_files, key=lambda x: x[0]):
        fsize = fpath.stat().st_size
        print(f"Computing SHA-256 for {fname}...")
        fhash = compute_sha256(fpath)
        manifest_csvs.append({
            "filename": fname,
            "relative_path": str(fpath.relative_to(pathlib.Path("."))).replace("\\", "/"),
            "size_bytes": fsize,
            "sha256": fhash
        })
        print(f"  {fname:<45} : {fsize:,} bytes | {fhash[:16]}...")

    manifest = {
        "dataset_name": "CICIDS2017",
        "collection_name": "MachineLearningCSV",
        "publisher": "Canadian Institute for Cybersecurity (CIC), University of New Brunswick (UNB)",
        "source_url": SOURCE_URL,
        "acquisition_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "publisher_checksums_available": False,
        "publisher_checksums_note": "UNB registration portal does not publish official SHA-256 hashes on public pages. Locally computed hashes serve as internal integrity baseline.",
        "archive": {
            "filename": TARGET_ARCHIVE.name,
            "relative_path": str(TARGET_ARCHIVE.relative_to(pathlib.Path("."))).replace("\\", "/"),
            "size_bytes": archive_size,
            "sha256": archive_sha256
        },
        "csv_file_count": len(manifest_csvs),
        "csv_files": manifest_csvs
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest generated successfully at {MANIFEST_PATH}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
