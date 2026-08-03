#!/usr/bin/env python3
"""Verify bundled and optionally downloaded release assets against SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-external", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "release_manifest.json").read_text())
    groups = ["bundled_files"]
    if args.include_external:
        groups.append("external_files")
    failures = []
    checked = 0
    for group in groups:
        for relative, metadata in manifest[group].items():
            path = root / relative
            if not path.is_file():
                failures.append(f"missing: {relative}")
                continue
            observed = sha256(path)
            if observed != metadata["sha256"]:
                failures.append(f"hash mismatch: {relative}")
            checked += 1
    if failures:
        raise SystemExit("Release verification failed:\n" + "\n".join(failures))
    print(f"Verified {checked} release files")


if __name__ == "__main__":
    main()
