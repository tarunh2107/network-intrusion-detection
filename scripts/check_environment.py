#!/usr/bin/env python3
"""Environment Verification Script.

Inspects Python runtime environment and validates imports and installed package
versions for all required dependencies using Python standard library.
"""

import importlib
import importlib.metadata
import sys


def main() -> int:
    print("=" * 65)
    print("AI-Based Network Intrusion Detection System — Environment Audit")
    print("=" * 65)
    print(f"Python Version    : {sys.version.split()[0]} ({sys.version})")
    print(f"Python Executable : {sys.executable}")
    print("-" * 65)

    # Map: module_name_to_import -> distribution_package_name_in_metadata
    required_packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("joblib", "joblib"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("streamlit", "streamlit"),
        ("requests", "requests"),
        ("plotly", "plotly"),
        ("pytest", "pytest"),
        ("httpx", "httpx"),
    ]

    failed = []
    print(f"{'Module':<15} {'Distribution':<18} {'Version':<15} {'Status'}")
    print("-" * 65)

    for mod_name, dist_name in required_packages:
        try:
            importlib.import_module(mod_name)
            try:
                version = importlib.metadata.version(dist_name)
            except importlib.metadata.PackageNotFoundError:
                version = "unknown"
            print(f"{mod_name:<15} {dist_name:<18} {version:<15} [OK]")
        except Exception as exc:
            print(f"{mod_name:<15} {dist_name:<18} {'FAILED':<15} [ERROR: {exc}]")
            failed.append((mod_name, str(exc)))

    print("=" * 65)
    if failed:
        print(f"FAILURE: {len(failed)} module(s) failed verification:")
        for mod, err in failed:
            print(f"  - {mod}: {err}")
        return 1

    print("SUCCESS: All 12 required dependencies imported and verified cleanly.")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
