import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from amp_design.release_config import PipelineConfig
from amp_design.release_pipeline import (
    _filter_known_amp_candidates,
    hard_filter_candidates,
    initialize_pareto,
    prepare_candidates,
)


def test_known_amp_novelty_removes_mmseqs_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pyarrow")
    sequences = ["ACDEFGHIKL", "KLMNPQRSTV"]
    hashes = [hashlib.sha256(sequence.encode()).hexdigest() for sequence in sequences]
    frame = pd.DataFrame(
        {
            "sequence": sequences,
            "sequence_sha256": hashes,
            "length": [10, 10],
        }
    )
    reference = tmp_path / "known_amp.fasta"
    reference.write_text(">known\nACDEFGHIKL\n", encoding="utf-8")
    config = PipelineConfig(
        sampling_manifests=(tmp_path / "manifest.json",),
        prepared_candidates=tmp_path / "prepared.parquet",
        scored_candidates=tmp_path / "scored.parquet",
        finalized_dir=tmp_path / "finalized",
        hard_filter_dir=tmp_path / "hard",
        require_known_amp_novelty=True,
        known_amp_reference=reference,
        known_amp_mmseqs_command=("mmseqs",),
    )

    def fake_search(command: list[str], **_: object) -> SimpleNamespace:
        easy_search = command.index("easy-search")
        hits = Path(command[easy_search + 3])
        hits.write_text(f"{hashes[0]}\tknown\t100\t100\t100\t10\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("amp_design.release_pipeline.subprocess.run", fake_search)

    retained, metadata = _filter_known_amp_candidates(
        frame, config, config.prepared_candidates
    )

    assert retained["sequence"].tolist() == [sequences[1]]
    assert metadata["removed"] == 1
    assert metadata["retained"] == 1
    audit = pd.read_parquet(tmp_path / "prepared_known_amp_novelty_audit.parquet")
    assert audit["known_amp_novelty_pass"].tolist() == [False, True]


def test_prepare_candidates_streams_valid_unique_rows(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    shard = tmp_path / "shard.csv"
    pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "sequence": ["ACDEFGHIKL", "ACDEFGHIKL", "A" * 9],
            "length": [10, 10, 9],
            "training_seed": [42] * 3,
            "generation_seed": [1] * 3,
            "shard_id": [0] * 3,
            "checkpoint_sha256": ["checkpoint"] * 3,
            "sampling_steps": [8] * 3,
            "use_ema": [True] * 3,
        }
    ).to_csv(shard, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "checkpoint_sha256": "checkpoint",
                "tasks": [{"output": str(shard), "count": 3}],
            }
        )
    )
    config = PipelineConfig(
        sampling_manifests=(manifest,),
        prepared_candidates=tmp_path / "prepared.parquet",
        scored_candidates=tmp_path / "scored.parquet",
        finalized_dir=tmp_path / "finalized",
        hard_filter_dir=tmp_path / "hard",
    )

    metadata = prepare_candidates(config)

    result = pd.read_parquet(config.prepared_candidates)
    assert result["sequence"].tolist() == ["ACDEFGHIKL"]
    assert metadata["input_rows"] == 3
    assert metadata["duplicate_rows"] == 1
    assert metadata["invalid_rows"] == 1


def test_prepare_candidates_does_not_publish_partial_output_when_novelty_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pyarrow")
    shard = tmp_path / "shard.csv"
    pd.DataFrame(
        {
            "candidate_id": ["a"],
            "sequence": ["ACDEFGHIKL"],
            "length": [10],
            "training_seed": [42],
            "generation_seed": [1],
            "shard_id": [0],
            "checkpoint_sha256": ["checkpoint"],
            "sampling_steps": [8],
            "use_ema": [True],
        }
    ).to_csv(shard, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "checkpoint_sha256": "checkpoint",
                "tasks": [{"output": str(shard), "count": 1}],
            }
        ),
        encoding="utf-8",
    )
    reference = tmp_path / "known_amp.fasta"
    reference.write_text(">known\nACDEFGHIKL\n", encoding="utf-8")
    config = PipelineConfig(
        sampling_manifests=(manifest,),
        prepared_candidates=tmp_path / "prepared.parquet",
        scored_candidates=tmp_path / "scored.parquet",
        finalized_dir=tmp_path / "finalized",
        hard_filter_dir=tmp_path / "hard",
        require_known_amp_novelty=True,
        known_amp_reference=reference,
    )
    monkeypatch.setattr(
        "amp_design.release_pipeline.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="failed"),
    )

    with pytest.raises(RuntimeError, match="Known-AMP MMseqs2 search failed"):
        prepare_candidates(config)

    assert not config.prepared_candidates.exists()


def test_hard_filter_requires_all_declared_gates(tmp_path: Path) -> None:
    finalized = tmp_path / "finalized"
    finalized.mkdir()
    pd.DataFrame(
        {
            "sequence": ["A" * 10, "C" * 10],
            "apex_pathogen_median_log10_mic": [1.0, 1.0],
            "toxicity_score": [0.1, 0.1],
            "hemolysis_score": [0.1, 0.1],
            "formal_safety_ad_pass": [True, True],
            "apex_median_log10_mic_sd": [0.1, 0.1],
            "novelty_pass": [True, True],
            "stability_half_life_hours": [2.0, 0.5],
            "activity_scorer": ["apex_pathogen", "apex_pathogen"],
            "safety_ad_protocol": [
                "portable_esm_length_v1",
                "portable_esm_length_v1",
            ],
        }
    ).to_csv(finalized / "combined_rank0_diverse.csv", index=False)
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "tasks": {
                    "toxicity": {"dehomologized_metrics": {"threshold": 0.2}},
                    "hemolysis": {"dehomologized_metrics": {"threshold": 0.2}},
                }
            }
        )
    )
    pipeline = PipelineConfig(
        sampling_manifests=(tmp_path / "manifest.json",),
        prepared_candidates=tmp_path / "prepared.parquet",
        scored_candidates=tmp_path / "scored.parquet",
        finalized_dir=finalized,
        hard_filter_dir=tmp_path / "hard",
    )
    optimization = SimpleNamespace(
        promotion_gate=gate,
        apex_uncertainty_max=0.3,
        novelty_max_identity=0.5,
        activity_scorer="apex_pathogen",
    )

    metadata = hard_filter_candidates(pipeline, optimization)

    selected = pd.read_csv(pipeline.hard_filter_dir / "final_candidates.csv")
    assert selected["sequence"].tolist() == ["A" * 10]
    assert metadata["selected_rows"] == 1


def test_hard_filter_supports_custom_maximize_activity(tmp_path: Path) -> None:
    finalized = tmp_path / "finalized"
    finalized.mkdir()
    pd.DataFrame(
        {
            "sequence": ["A" * 10, "C" * 10],
            "custom_activity": [0.9, 0.2],
            "toxicity_score": [0.1, 0.1],
            "hemolysis_score": [0.1, 0.1],
            "formal_safety_ad_pass": [True, True],
            "novelty_pass": [True, True],
            "stability_half_life_hours": [2.0, 2.0],
            "activity_scorer": ["command", "command"],
        }
    ).to_csv(finalized / "combined_rank0_diverse.csv", index=False)
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "tasks": {
                    "toxicity": {"dehomologized_metrics": {"threshold": 0.2}},
                    "hemolysis": {"dehomologized_metrics": {"threshold": 0.2}},
                }
            }
        )
    )
    pipeline = PipelineConfig(
        sampling_manifests=(tmp_path / "manifest.json",),
        prepared_candidates=tmp_path / "prepared.parquet",
        scored_candidates=tmp_path / "scored.parquet",
        finalized_dir=finalized,
        hard_filter_dir=tmp_path / "hard",
        activity_hard_threshold=0.5,
        activity_hard_direction="maximize",
        require_apex_uncertainty=False,
    )
    optimization = SimpleNamespace(
        promotion_gate=gate,
        apex_uncertainty_max=0.3,
        novelty_max_identity=0.5,
        activity_scorer="command",
        effective_activity_output_column="custom_activity",
        objective_columns=("custom_activity", "toxicity_score", "hemolysis_score"),
        objective_directions=("maximize", "minimize", "minimize"),
    )

    metadata = hard_filter_candidates(pipeline, optimization)

    selected = pd.read_csv(pipeline.hard_filter_dir / "final_candidates.csv")
    assert selected["sequence"].tolist() == ["A" * 10]
    assert metadata["thresholds"]["activity_direction"] == "maximize"


def test_initialize_pareto_uses_configured_objective_direction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pyarrow")
    scored_path = tmp_path / "scored.parquet"
    pd.DataFrame(
        {
            "sequence": ["A" * 10, "C" * 10],
            "sequence_sha256": ["a", "c"],
            "length": [10, 10],
            "custom_activity": [0.1, 0.9],
        }
    ).to_parquet(scored_path, index=False)
    pipeline = SimpleNamespace(scored_candidates=scored_path)
    optimization = SimpleNamespace(
        ranked_candidates=tmp_path / "ranked.parquet",
        initial_population=tmp_path / "initial.csv",
        population_size=1,
        cluster_max_members=1,
        objective_columns=("custom_activity",),
        objective_directions=("maximize",),
        enforce_safety_ad=False,
        enforce_apex_uncertainty=False,
        enforce_novelty=False,
    )
    monkeypatch.setattr(
        "amp_design.release_pipeline._cluster_assignments",
        lambda frame, optimization, destination: {
            row.sequence_sha256: row.sequence_sha256 for row in frame.itertuples()
        },
    )

    initialize_pareto(pipeline, optimization)

    selected = pd.read_csv(optimization.initial_population)
    assert selected["custom_activity"].tolist() == [0.9]
