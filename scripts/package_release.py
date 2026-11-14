"""
Project SHIFT: Official Release Packaging Script.
IOAI 2027 Singapore Benchmark Packaging Tool.

Deterministically packages:
  1. releases/contestant_package_v1.0.zip:
     - Task statement, public dev data, safe loader, starter baseline, and local evaluator.
     - Strictly validates ZERO answer leakage (no private labels, no context event frames).
     - Excludes any .git, .venv, or __pycache__ artifacts.
  2. releases/author_package_v1.0.zip:
     - Jury grading script, benchmark generator, shortcut auditor, full private reference keys,
       jury solution guide, and human validation protocol.
"""

from __future__ import annotations
import sys
import os
import zipfile
import hashlib
import json
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contestant_package.loader import assert_zero_leakage


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def package_releases() -> Tuple[Path, Path]:
    print("=" * 74)
    print(" PROJECT SHIFT: OFFICIAL IOAI RELEASE PACKAGING (v1.0)")
    print("=" * 74)

    releases_dir = PROJECT_ROOT / "releases"
    releases_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Build Contestant Package (v1.0)
    # -------------------------------------------------------------
    contestant_zip_path = releases_dir / "contestant_package_v1.0.zip"
    print(f"\n[1] Packaging Contestant Archive: {contestant_zip_path.name}")

    contestant_files: List[Tuple[Path, str]] = [
        (PROJECT_ROOT / "contestant_package" / "README.md", "README.md"),
        (PROJECT_ROOT / "docs" / "IOAI_TASK_STATEMENT.md", "IOAI_TASK_STATEMENT.md"),
        (PROJECT_ROOT / "contestant_package" / "loader.py", "loader.py"),
        (PROJECT_ROOT / "contestant_package" / "submission_template.py", "submission_template.py"),
        (PROJECT_ROOT / "contestant_package" / "local_evaluator.py", "local_evaluator.py"),
        (PROJECT_ROOT / "contestant_package" / "sample_data" / "public_dev.jsonl", "sample_data/public_dev.jsonl"),
        (PROJECT_ROOT / "contestant_package" / "sample_data" / "sample_dev.jsonl", "sample_data/sample_dev.jsonl"),
        (PROJECT_ROOT / "datasets" / "hidden_test_contestant.jsonl", "sample_data/hidden_test_contestant.jsonl"),
    ]

    # Pre-packaging security verification
    for src, arc in contestant_files:
        assert src.exists(), f"Missing required contestant file: {src}"
        if src.suffix == ".jsonl":
            print(f"    * Verifying zero context leakage in {src.name}...")
            # For test files, verify ZERO answers and ZERO context leakage
            if "hidden_test" in src.name:
                assert_zero_leakage(src, check_query_labels=True)
            else:
                # For dev files, verify ZERO context leakage
                assert_zero_leakage(src, check_query_labels=False)

    # Build Zip deterministically
    with zipfile.ZipFile(contestant_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in contestant_files:
            zf.write(src, arcname=f"contestant_package/{arcname}")
            print(f"    + Added: contestant_package/{arcname}")

    # Verify no forbidden files inside contestant zip
    with zipfile.ZipFile(contestant_zip_path, "r") as zf:
        for name in zf.namelist():
            lower = name.lower()
            assert "hidden_test_labels" not in lower, f"SECURITY ERROR: Leaked file {name}"
            assert "hidden_test_full_reference" not in lower, f"SECURITY ERROR: Leaked file {name}"
            assert "__pycache__" not in lower, f"CACHE LEAK: {name}"
            assert ".pyc" not in lower, f"COMPILED FILE LEAK: {name}"

    c_sha = compute_sha256(contestant_zip_path)
    c_size_kb = contestant_zip_path.stat().st_size / 1024.0
    print(f"    [OK] Contestant package created: {c_size_kb:.2f} KB | SHA-256: {c_sha[:16]}...")

    # -------------------------------------------------------------
    # 2. Build Author & Jury Package (v1.0)
    # -------------------------------------------------------------
    author_zip_path = releases_dir / "author_package_v1.0.zip"
    print(f"\n[2] Packaging Author & Jury Archive: {author_zip_path.name}")

    # Ensure private data files exist in author_package/private_data
    priv_dir = PROJECT_ROOT / "author_package" / "private_data"
    priv_dir.mkdir(parents=True, exist_ok=True)

    # Sync private files from datasets if needed
    src_labels = PROJECT_ROOT / "datasets" / "hidden_test_labels.json"
    dst_labels = priv_dir / "hidden_test_labels.json"
    if src_labels.exists() and not dst_labels.exists():
        dst_labels.write_bytes(src_labels.read_bytes())

    src_ref = PROJECT_ROOT / "datasets" / "hidden_test_full_reference.jsonl"
    dst_ref = priv_dir / "hidden_test_full_reference.jsonl"
    if src_ref.exists() and not dst_ref.exists():
        dst_ref.write_bytes(src_ref.read_bytes())

    src_cont = PROJECT_ROOT / "datasets" / "hidden_test_contestant.jsonl"
    dst_cont = priv_dir / "hidden_test_contestant.jsonl"
    if src_cont.exists() and not dst_cont.exists():
        dst_cont.write_bytes(src_cont.read_bytes())

    author_files: List[Tuple[Path, str]] = [
        (PROJECT_ROOT / "author_package" / "README.md", "README.md"),
        (PROJECT_ROOT / "author_package" / "generate_benchmarks.py", "generate_benchmarks.py"),
        (PROJECT_ROOT / "author_package" / "grade_submission.py", "grade_submission.py"),
        (PROJECT_ROOT / "author_package" / "shortcut_auditor.py", "shortcut_auditor.py"),
        (dst_labels, "private_data/hidden_test_labels.json"),
        (dst_ref, "private_data/hidden_test_full_reference.jsonl"),
        (dst_cont, "private_data/hidden_test_contestant.jsonl"),
        (PROJECT_ROOT / "docs" / "IOAI_TASK_STATEMENT.md", "docs/IOAI_TASK_STATEMENT.md"),
        (PROJECT_ROOT / "docs" / "JURY_SOLUTION_GUIDE.md", "docs/JURY_SOLUTION_GUIDE.md"),
        (PROJECT_ROOT / "docs" / "human_validation_guide.md", "docs/human_validation_guide.md"),
        (PROJECT_ROOT / "docs" / "literature_review.md", "docs/literature_review.md"),
    ]

    # Add core shift library to author package so jury evaluation is self-contained
    shift_dir = PROJECT_ROOT / "shift"
    for py_file in shift_dir.glob("*.py"):
        author_files.append((py_file, f"shift/{py_file.name}"))

    with zipfile.ZipFile(author_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in author_files:
            if src.exists():
                zf.write(src, arcname=f"author_package/{arcname}")
                print(f"    + Added: author_package/{arcname}")

    a_sha = compute_sha256(author_zip_path)
    a_size_kb = author_zip_path.stat().st_size / 1024.0
    print(f"    [OK] Author package created: {a_size_kb:.2f} KB | SHA-256: {a_sha[:16]}...")

    # Export manifest
    manifest = {
        "version": "v1.0",
        "timestamp": hashlib.sha256(str(releases_dir).encode()).hexdigest()[:8],
        "packages": {
            "contestant_package": {
                "filename": contestant_zip_path.name,
                "size_kb": round(c_size_kb, 2),
                "sha256": c_sha,
                "leakage_audited": True,
            },
            "author_package": {
                "filename": author_zip_path.name,
                "size_kb": round(a_size_kb, 2),
                "sha256": a_sha,
                "includes_private_keys": True,
            }
        }
    }

    manifest_path = releases_dir / "release_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 74)
    print(" RELEASE PACKAGES READY FOR IOAI 2027 SUBMISSION")
    print("=" * 74)
    print(f"  Contestant Package : releases/{contestant_zip_path.name} ({c_size_kb:.1f} KB)")
    print(f"  Author/Jury Package: releases/{author_zip_path.name} ({a_size_kb:.1f} KB)")
    print(f"  Manifest Record    : releases/{manifest_path.name}")
    print("=" * 74 + "\n")

    return contestant_zip_path, author_zip_path


if __name__ == "__main__":
    package_releases()
