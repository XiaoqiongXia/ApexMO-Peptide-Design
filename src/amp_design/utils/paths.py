"""Source-tree locations shared by configuration and provenance code."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
