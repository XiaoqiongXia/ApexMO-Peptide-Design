#!/usr/bin/env python3
"""Compatibility entry point for scripts/scoring/run_hemopi2_fixed.py."""
import runpy
from pathlib import Path

_target = Path(__file__).resolve().parent / "scoring/run_hemopi2_fixed.py"
if __name__ == "__main__":
    runpy.run_path(str(_target), run_name="__main__")
else:
    globals().update({key: value for key, value in runpy.run_path(str(_target)).items()
                      if not key.startswith("__")})
