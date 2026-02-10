#!/usr/bin/env python3
"""Fetch test data files from GitHub release.

This script downloads test data files from a GitHub release if they're missing
or have incorrect checksums. It's designed to be run before tests.

Usage:
    python scripts/fetch_test_data.py
    # or via make:
    make fetch-test-data
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "data"
MANIFEST_PATH = TEST_DATA_DIR / "manifest.json"


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_manifest() -> dict:
    """Load the manifest file."""
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def file_needs_download(filepath: Path, expected_sha256: str) -> bool:
    """Check if a file needs to be downloaded."""
    if not filepath.exists():
        return True
    actual_sha256 = compute_sha256(filepath)
    return actual_sha256 != expected_sha256


def download_with_gh(repo: str, tag: str, filename: str, dest_dir: Path) -> bool:
    """Download a file using gh CLI."""
    try:
        result = subprocess.run(
            [
                "gh",
                "release",
                "download",
                tag,
                "--repo",
                repo,
                "--pattern",
                filename,
                "--dir",
                str(dest_dir),
                "--clobber",
            ],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_with_urllib(repo: str, tag: str, filename: str, dest_path: Path) -> bool:
    """Download a file using urllib (fallback)."""
    url = f"https://github.com/{repo}/releases/download/{tag}/{filename}"
    try:
        print(f"  Downloading from {url}")
        with urllib.request.urlopen(url, timeout=60) as response:
            with open(dest_path, "wb") as f:
                shutil.copyfileobj(response, f)
        return True
    except Exception as e:
        print(f"  Failed to download: {e}")
        return False


def download_file(repo: str, tag: str, filename: str, dest_dir: Path) -> bool:
    """Download a file, trying gh CLI first, then urllib."""
    dest_path = dest_dir / filename

    # Try gh CLI first
    if shutil.which("gh"):
        print(f"  Using gh CLI to download {filename}...")
        if download_with_gh(repo, tag, filename, dest_dir):
            return True
        print("  gh CLI failed, falling back to direct download...")

    # Fall back to urllib
    return download_with_urllib(repo, tag, filename, dest_path)


def main() -> int:
    """Main entry point."""
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return 1

    manifest = load_manifest()
    repo = manifest["repo"]
    tag = manifest["release_tag"]
    files = manifest["files"]

    print(f"Checking test data files (release: {tag})...")

    all_ok = True
    for file_info in files:
        filename = file_info["name"]
        expected_sha256 = file_info["sha256"]
        filepath = TEST_DATA_DIR / filename

        if not file_needs_download(filepath, expected_sha256):
            print(f"  {filename}: OK (cached)")
            continue

        print(f"  {filename}: downloading...")
        if not download_file(repo, tag, filename, TEST_DATA_DIR):
            print(f"  Error: Failed to download {filename}")
            all_ok = False
            continue

        # Verify download
        if not filepath.exists():
            print(f"  Error: {filename} not found after download")
            all_ok = False
            continue

        actual_sha256 = compute_sha256(filepath)
        if actual_sha256 != expected_sha256:
            print(f"  Error: {filename} checksum mismatch")
            print(f"    Expected: {expected_sha256}")
            print(f"    Got:      {actual_sha256}")
            all_ok = False
        else:
            print(f"  {filename}: OK (downloaded)")

    if all_ok:
        print("All test data files ready.")
        return 0
    else:
        print("Some files failed to download or verify.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
