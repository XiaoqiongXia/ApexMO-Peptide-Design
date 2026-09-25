"""Public entry points and provenance paths survive the functional reorganization."""

import importlib
import importlib.metadata
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOVES = json.loads((ROOT / "docs/module_moves.json").read_text())


@pytest.mark.parametrize("old,new", MOVES["modules"].items())
def test_old_and_new_module_paths_share_one_implementation(old, new, monkeypatch):
    legacy = importlib.import_module(f"amp_design.{old}")
    canonical = importlib.import_module(f"amp_design.{new}")
    assert legacy is canonical
    monkeypatch.setattr(legacy, "_layout_probe", object(), raising=False)
    assert canonical._layout_probe is legacy._layout_probe


def test_historical_pickle_class_path_is_resolvable():
    from amp_design.generation.model import AMPDDiTConfig

    # Protocol-0 GLOBAL reference, as used by old artifacts storing this class name.
    assert pickle.loads(b"camp_design.model\nAMPDDiTConfig\n.") is AMPDDiTConfig


@pytest.mark.parametrize("old,new", [
    (old, new) for old, new in MOVES["scripts"].items() if old.endswith(".py")
])
def test_python_entrypoints_preserve_argument_parsing_outside_repo(old, new, tmp_path):
    for relative in [old, new]:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / relative), "--help"],
            cwd=tmp_path, capture_output=True, text=True,
        )
        if old == "run_hemopi2_fixed.py":
            # This adapter intentionally forwards --help and requires its own options.
            assert result.returncode == 2
            assert "--official-script" in result.stderr
            assert "--runtime-dir" in result.stderr
        else:
            assert result.returncode == 0
            assert "usage:" in result.stdout


@pytest.mark.parametrize("module", ["amp_design.release_cli", "amp_design.workflows.cli"])
def test_module_cli_outside_repository(module, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert "init-config" in result.stdout


@pytest.mark.parametrize("command", ["run_full_pipeline.sh", "run_smoke_test.sh"])
def test_shell_forwarders_preserve_project_root_override(command, tmp_path):
    env = {**os.environ, "AMP_DESIGN_REPO_ROOT": str(tmp_path)}
    for relative in [command, MOVES["scripts"][command]]:
        result = subprocess.run(
            ["bash", str(ROOT / "scripts" / relative)],
            cwd=tmp_path, env=env, capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert f"Missing {tmp_path}/.amp_design_env" in result.stderr


def test_provenance_hashes_bind_canonical_source_files(tmp_path, monkeypatch):
    from amp_design.optimization import search
    from amp_design.workflows.config import OptimizationConfig

    # Exercise the function's file selection independently from external weights.
    seen = []
    monkeypatch.setattr(search, "discover_models", lambda root: [])
    monkeypatch.setattr(search, "_sha256_file", lambda path: seen.append(Path(path)) or "hash")
    config = OptimizationConfig(
        **{name: tmp_path / name for name in (
            "ranked_candidates", "initial_population", "training_sequences", "promotion_gate",
            "scorer_root", "safety_ad_root", "stability_model", "esm_model", "output_root",
        )},
        activity_scorer="apex_pathogen", apex_root=tmp_path / "external_apex",
    )
    # The reference config selects apex_pathogen, whose weight lookup is also isolated.
    monkeypatch.setattr(search, "discover_pathogen_models", lambda root: [])
    monkeypatch.setattr(search, "_required_esm_files", lambda root: [])
    search.frozen_hashes(config)
    source_files = [p for p in seen if p.suffix == ".py" and "amp_design" in p.parts]
    assert source_files
    assert all(p.exists() for p in source_files)
    assert all(p.parent.name in {"optimization", "predictors", "workflows"} for p in source_files)


def test_installed_console_entrypoints_resolve_to_canonical_modules():
    entries = importlib.metadata.distribution("amp-design").entry_points
    commands = [entry for entry in entries if entry.group == "console_scripts"]
    assert len(commands) == 9
    for entry in commands:
        assert entry.module.removeprefix("amp_design.") in MOVES["modules"].values()
        assert callable(entry.load())
