import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from amp_design.release_cli import check_optimization, main, resolve_device
from amp_design.release_config import (
    ObjectiveSpec,
    OptimizationConfig,
    SamplingConfig,
    SamplingRunsConfig,
)
from amp_design.release_optimization import (
    _cluster_command,
    _early_stopping_status,
    _external_runtime_contract,
    _objective_tolerance,
    _read_complete_cluster_assignments,
    _validated_candidate_identities,
    optimization_contract,
    read_table,
)


def test_sampling_config_loads_and_validates(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config_path = tmp_path / "sample.yaml"
    config_path.write_text(
        "\n".join(
            [
                "sampling:",
                f"  checkpoint: {checkpoint}",
                f"  output_root: {tmp_path / 'out'}",
                "  training_seed: 42",
                "  lengths: [10, 20]",
                "  samples_per_length: 20",
                "  shard_size: 10",
                "  steps: 8",
                "  batch_size: 4",
                "  device: cpu",
            ]
        )
        + "\n"
    )

    config = SamplingConfig.from_file(config_path)

    assert config.checkpoint == checkpoint.resolve()
    assert config.lengths == (10, 20)
    assert config.samples_per_length == 20
    assert resolve_device(config.device).type == "cpu"


def test_sampling_config_rejects_unknown_keys(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config_path = tmp_path / "sample.yaml"
    config_path.write_text(
        f"sampling:\n  checkpoint: {checkpoint}\n  output_root: {tmp_path}\n"
        "  training_seed: 1\n  lengths: [10]\n  typo: true\n"
    )

    with pytest.raises(ValueError, match="Unknown sampling configuration keys"):
        SamplingConfig.from_file(config_path)


def test_config_paths_expand_environment_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv("AMP_TEST_CHECKPOINT", str(checkpoint))
    config_path = tmp_path / "sample.yaml"
    config_path.write_text(
        "sampling:\n"
        '  checkpoint: "${AMP_TEST_CHECKPOINT}"\n'
        f"  output_root: {tmp_path / 'out'}\n"
        "  training_seed: 42\n"
        "  lengths: [10]\n",
        encoding="utf-8",
    )

    assert SamplingConfig.from_file(config_path).checkpoint == checkpoint.resolve()


def test_config_rejects_unresolved_path_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AMP_MISSING_CHECKPOINT", raising=False)
    config_path = tmp_path / "sample.yaml"
    config_path.write_text(
        "sampling:\n"
        '  checkpoint: "${AMP_MISSING_CHECKPOINT}"\n'
        f"  output_root: {tmp_path / 'out'}\n"
        "  training_seed: 42\n"
        "  lengths: [10]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unresolved environment variable"):
        SamplingConfig.from_file(config_path)


def test_sampling_runs_config_loads_multiple_checkpoints(tmp_path: Path) -> None:
    checkpoints = [tmp_path / "seed42.pt", tmp_path / "seed123.pt"]
    for checkpoint in checkpoints:
        checkpoint.write_bytes(b"checkpoint")
    config_path = tmp_path / "sample_many.yaml"
    config_path.write_text(
        "sampling_runs:\n"
        + "\n".join(
            [
                f"  - checkpoint: {checkpoint}\n"
                f"    output_root: {tmp_path / 'sampling'}\n"
                f"    training_seed: {seed}\n"
                "    lengths: [10, 11]\n"
                "    samples_per_length: 20\n"
                "    shard_size: 10\n"
                "    steps: 8\n"
                "    batch_size: 4\n"
                "    device: cpu"
                for checkpoint, seed in zip(checkpoints, (42, 123), strict=True)
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = SamplingRunsConfig.from_file(config_path)

    assert [run.training_seed for run in config.runs] == [42, 123]
    assert SamplingRunsConfig.is_configured(config_path)


def test_sampling_runs_reject_duplicate_manifest_targets(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    values = {
        "checkpoint": checkpoint,
        "output_root": tmp_path / "sampling",
        "training_seed": 42,
        "lengths": [10],
    }
    run = SamplingConfig.from_mapping(values)

    with pytest.raises(ValueError, match="training_seed values must be unique"):
        SamplingRunsConfig((run, run)).validate()


def test_init_config_writes_packaged_templates(tmp_path: Path) -> None:
    main(["init-config", "--output-dir", str(tmp_path)])

    sampling = tmp_path / "release_sampling_v1_3.yaml"
    optimization = tmp_path / "release_optimization_v1_3.yaml"
    pipeline = tmp_path / "release_pipeline_v1_3.yaml"
    assert sampling.is_file()
    assert optimization.is_file()
    assert pipeline.is_file()
    assert "sampling:" in sampling.read_text()
    assert "optimization:" in optimization.read_text()


def test_cluster_command_preserves_frozen_protocol(tmp_path: Path) -> None:
    config = SimpleNamespace(
        mmseqs_command=("conda", "run", "-n", "bg", "mmseqs"),
        mmseqs_threads=8,
    )

    command = _cluster_command(config, tmp_path / "in.fa", tmp_path / "clusters", tmp_path / "tmp")

    assert command[:5] == ["conda", "run", "-n", "bg", "mmseqs"]
    assert command[5] == "easy-cluster"
    assert command[command.index("--min-seq-id") + 1] == "0.8"
    assert command[command.index("--threads") + 1] == "8"


def test_cluster_assignments_require_every_input_member(tmp_path: Path) -> None:
    complete = tmp_path / "complete.tsv"
    complete.write_text("a\ta\na\tb\n", encoding="utf-8")
    assert _read_complete_cluster_assignments(complete, {"a", "b"}) == {
        "a": "a",
        "b": "a",
    }

    incomplete = tmp_path / "incomplete.tsv"
    incomplete.write_text("a\ta\n", encoding="utf-8")
    with pytest.raises(ValueError, match="membership mismatch"):
        _read_complete_cluster_assignments(incomplete, {"a", "b"})


def test_external_runtime_contract_binds_mmseqs_and_activity_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def completed(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[-1] == "version":
            return SimpleNamespace(returncode=0, stdout="MMseqs2 18.8cc5c\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")

    monkeypatch.setattr("amp_design.release_optimization.subprocess.run", completed)
    config = SimpleNamespace(
        mmseqs_command=("mmseqs",),
        activity_scorer="apex",
        apex_root=Path("external/apex"),
    )

    assert _external_runtime_contract(config) == {
        "mmseqs_version": "MMseqs2 18.8cc5c",
        "activity_repository_commit": "abc123",
    }


def test_optimization_contract_excludes_resume_cursor() -> None:
    config = SimpleNamespace(
        protocol="release",
        activity_scorer="apex_pathogen",
        seeds=(42, 123),
        generations=100,
        population_size=500,
        batch_size_apex=3000,
        batch_size_esm=32,
        device="cuda",
        mmseqs_command=("mmseqs",),
        mmseqs_threads=16,
        cluster_max_members=5,
        resume_generation=52,
        apex_uncertainty_max=0.3,
        novelty_max_identity=0.5,
        novelty_min_coverage=0.8,
        enforce_safety_ad=True,
        enforce_apex_uncertainty=True,
        enforce_novelty=True,
        novelty_references=(),
    )

    contract = optimization_contract(config)

    assert contract["seeds"] == [42, 123]
    assert contract["population_size"] == 500
    assert "resume_generation" not in contract


def test_safety_objective_tolerance_covers_gpu_batch_roundoff() -> None:
    config = SimpleNamespace(effective_activity_output_column="activity")

    assert _objective_tolerance("toxicity_score", config) == pytest.approx(1e-5)
    assert _objective_tolerance("hemolysis_score", config) == pytest.approx(1e-5)


def test_read_table_supports_csv_and_parquet(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"sequence": ["ACDE", "KLMN"], "score": [1, 2]})
    csv_path = tmp_path / "table.csv"
    parquet_path = tmp_path / "table.parquet"
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)

    assert read_table(csv_path).equals(frame)
    assert read_table(parquet_path).equals(frame)


def test_candidate_identity_validation_rejects_duplicates_and_stale_hashes() -> None:
    with pytest.raises(ValueError, match="duplicate normalized sequences"):
        _validated_candidate_identities(
            pd.DataFrame({"sequence": ["acde", "ACDE"]}),
            "Candidates",
        )
    with pytest.raises(ValueError, match="SHA-256 mismatches"):
        _validated_candidate_identities(
            pd.DataFrame({"sequence": ["ACDE"], "sequence_sha256": ["stale"]}),
            "Candidates",
        )


def _optimization_assets(tmp_path: Path) -> dict[str, Path]:
    files = {}
    for name in ("train.csv", "gate.json", "stability.pkl"):
        files[name] = tmp_path / name
        files[name].write_text("{}\n")
    for name in ("scorers", "ad", "esm", "apex"):
        files[name] = tmp_path / name
        files[name].mkdir()
    return files


def test_apex_pathogen_config_does_not_require_strain_groups(tmp_path: Path) -> None:
    assets = _optimization_assets(tmp_path)
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(
        '{"activity_scorer":"apex_pathogen","uncertainty_threshold_q95_log10_mic_sd":0.35}\n'
    )
    config = OptimizationConfig(
        ranked_candidates=tmp_path / "ranked.csv",
        initial_population=tmp_path / "initial.csv",
        training_sequences=assets["train.csv"],
        promotion_gate=assets["gate.json"],
        scorer_root=assets["scorers"],
        safety_ad_root=assets["ad"],
        stability_model=assets["stability.pkl"],
        esm_model=assets["esm"],
        apex_root=assets["apex"],
        output_root=tmp_path / "output",
        activity_scorer="apex_pathogen",
        activity_thresholds=thresholds,
        apex_uncertainty_max=0.35,
    )

    config.validate(require_search_inputs=False)

    with pytest.raises(ValueError, match="max_initial_edit_fraction"):
        replace(config, max_initial_edit_fraction=0).validate(require_search_inputs=False)
    with pytest.raises(ValueError, match="max_initial_edits"):
        replace(config, max_initial_edits=0).validate(require_search_inputs=False)
    with pytest.raises(ValueError, match="numeric parameters"):
        replace(config, max_offspring_attempts=0).validate(require_search_inputs=False)
    with pytest.raises(ValueError, match="early_stopping_patience"):
        replace(config, early_stopping_patience=-1).validate(require_search_inputs=False)
    with pytest.raises(ValueError, match="early_stopping_metric"):
        replace(config, early_stopping_metric="loss").validate(require_search_inputs=False)


def test_early_stopping_status_respects_material_improvement() -> None:
    best, stale = _early_stopping_status([1.0, 1.0005, 1.002, 1.0025], 0.001)
    assert best == pytest.approx(1.002)
    assert stale == 1


def test_explicit_finalization_seed_directories_follow_seed_order(tmp_path: Path) -> None:
    assets = _optimization_assets(tmp_path)
    seed_42 = tmp_path / "first" / "seed_42"
    seed_123 = tmp_path / "second" / "seed_123"
    seed_42.mkdir(parents=True)
    seed_123.mkdir(parents=True)
    config = OptimizationConfig(
        ranked_candidates=tmp_path / "ranked.csv",
        initial_population=tmp_path / "initial.csv",
        training_sequences=assets["train.csv"],
        promotion_gate=assets["gate.json"],
        scorer_root=assets["scorers"],
        safety_ad_root=assets["ad"],
        stability_model=assets["stability.pkl"],
        esm_model=assets["esm"],
        output_root=tmp_path / "output",
        activity_scorer="command",
        activity_command=("score", "{input}", "{output}"),
        activity_model_files=(assets["train.csv"],),
        enforce_apex_uncertainty=False,
        seeds=(42, 123),
        finalization_seed_directories=(seed_42, seed_123),
    )

    config.validate(require_search_inputs=False)

    assert config.effective_seed_directories == (seed_42, seed_123)


def test_apex_pathogen_config_rejects_threshold_drift(tmp_path: Path) -> None:
    assets = _optimization_assets(tmp_path)
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(
        '{"activity_scorer":"apex_pathogen","uncertainty_threshold_q95_log10_mic_sd":0.35}\n'
    )
    config = OptimizationConfig(
        ranked_candidates=tmp_path / "ranked.csv",
        initial_population=tmp_path / "initial.csv",
        training_sequences=assets["train.csv"],
        promotion_gate=assets["gate.json"],
        scorer_root=assets["scorers"],
        safety_ad_root=assets["ad"],
        stability_model=assets["stability.pkl"],
        esm_model=assets["esm"],
        apex_root=assets["apex"],
        output_root=tmp_path / "output",
        activity_scorer="apex_pathogen",
        activity_thresholds=thresholds,
        apex_uncertainty_max=0.36,
    )

    with pytest.raises(ValueError, match="does not match"):
        config.validate(require_search_inputs=False)


def test_command_scorer_and_custom_objectives_validate_without_apex(tmp_path: Path) -> None:
    assets = _optimization_assets(tmp_path)
    model_file = tmp_path / "custom-model.bin"
    model_file.write_bytes(b"model")
    config = OptimizationConfig(
        ranked_candidates=tmp_path / "ranked.csv",
        initial_population=tmp_path / "initial.csv",
        training_sequences=assets["train.csv"],
        promotion_gate=assets["gate.json"],
        scorer_root=assets["scorers"],
        safety_ad_root=assets["ad"],
        stability_model=assets["stability.pkl"],
        esm_model=assets["esm"],
        output_root=tmp_path / "output",
        activity_scorer="command",
        activity_command=("custom-score", "{input}", "{output}"),
        activity_model_files=(model_file,),
        objectives=(
            ObjectiveSpec("activity_score", "maximize"),
            ObjectiveSpec("toxicity_score", "minimize"),
        ),
        enforce_apex_uncertainty=False,
    )

    config.validate(require_search_inputs=False)

    assert config.apex_root is None
    assert config.objective_columns == ("activity_score", "toxicity_score")
    assert config.objective_directions == ("maximize", "minimize")


def test_command_scorer_configuration_loads_from_yaml(tmp_path: Path) -> None:
    assets = _optimization_assets(tmp_path)
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text("# scorer wrapper\n", encoding="utf-8")
    config_path = tmp_path / "optimization.yaml"
    config_path.write_text(
        "\n".join(
            [
                "optimization:",
                f"  ranked_candidates: {tmp_path / 'ranked.csv'}",
                f"  initial_population: {tmp_path / 'initial.csv'}",
                f"  training_sequences: {assets['train.csv']}",
                f"  promotion_gate: {assets['gate.json']}",
                f"  scorer_root: {assets['scorers']}",
                f"  safety_ad_root: {assets['ad']}",
                f"  stability_model: {assets['stability.pkl']}",
                f"  esm_model: {assets['esm']}",
                f"  output_root: {tmp_path / 'output'}",
                "  activity_scorer: command",
                f"  activity_command: [python, {wrapper}, '{{input}}', '{{output}}']",
                f"  activity_model_files: [{wrapper}]",
                "  activity_output_column: external_score",
                "  objectives:",
                "    - {column: external_score, direction: maximize}",
                "  enforce_apex_uncertainty: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = OptimizationConfig.from_file(config_path, require_search_inputs=False)

    assert config.activity_command[-2:] == ("{input}", "{output}")
    assert config.objective_columns == ("external_score",)
    assert config.objective_directions == ("maximize",)


def test_check_optimization_accepts_command_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = _optimization_assets(tmp_path)
    assets["gate.json"].write_text(
        '{"schema_version":1,"all_scorers_promoted":true}\n', encoding="utf-8"
    )
    for task in ("toxicity", "hemolysis"):
        (assets["scorers"] / f"{task}_model.pkl").write_bytes(b"model")
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text("# wrapper\n", encoding="utf-8")
    config = OptimizationConfig(
        ranked_candidates=tmp_path / "ranked.csv",
        initial_population=tmp_path / "initial.csv",
        training_sequences=assets["train.csv"],
        promotion_gate=assets["gate.json"],
        scorer_root=assets["scorers"],
        safety_ad_root=assets["ad"],
        stability_model=assets["stability.pkl"],
        esm_model=assets["esm"],
        output_root=tmp_path / "output",
        activity_scorer="command",
        activity_command=(sys.executable, str(wrapper), "{input}", "{output}"),
        activity_model_files=(wrapper,),
        enforce_apex_uncertainty=False,
        device="cpu",
    )
    monkeypatch.setattr("amp_design.release_cli.shutil.which", lambda command: command)
    monkeypatch.setattr(
        "amp_design.release_cli.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="mmseqs 1\n", stderr=""),
    )

    report = check_optimization(config)

    assert report["passed"]
    assert report["activity_scorer"] == "command"
    assert report["n_apex_models"] == 0
