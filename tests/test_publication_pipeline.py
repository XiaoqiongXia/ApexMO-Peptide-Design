import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from amp_design.genetic_pareto import sequence_sha256
from amp_design.publication_pipeline import (
    merge_and_filter_internal,
    rank_and_cluster_final,
    run_publication_pipeline,
)


def test_merge_internal_filters_starts_from_every_final_rank(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
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
    seed_dirs = []
    for seed, sequence, rank in ((42, "A" * 10, 0), (123, "C" * 10, 3)):
        directory = tmp_path / f"seed_{seed}"
        directory.mkdir()
        pd.DataFrame(
            {
                "sequence": [sequence],
                "sequence_sha256": [sequence_sha256(sequence)],
                "pareto_rank": [rank],
                "crowding_distance": [1.0],
                "apex_pathogen_median_log10_mic": [1.0],
                "toxicity_score": [0.1],
                "hemolysis_score": [0.1],
                "formal_safety_ad_pass": [True],
                "apex_median_log10_mic_sd": [0.1],
                "novelty_pass": [True],
                "stability_half_life_hours": [2.0],
            }
        ).to_csv(directory / "generation_001.csv", index=False)
        seed_dirs.append(directory)
    pipeline = SimpleNamespace(
        mic_max_um=128.0,
        require_safety_ad=True,
        require_apex_uncertainty=True,
        require_novelty=True,
        require_stability=True,
        stability_min_hours=1.0,
    )
    optimization = SimpleNamespace(
        generations=1,
        effective_seed_directories=tuple(seed_dirs),
        seeds=(42, 123),
        promotion_gate=gate,
        effective_activity_output_column="apex_pathogen_median_log10_mic",
        apex_uncertainty_max=0.3,
        enforce_initial_edit_distance=False,
    )
    publication = SimpleNamespace(output_dir=tmp_path / "publication")

    metadata = merge_and_filter_internal(pipeline, optimization, publication)

    eligible = pd.read_parquet(
        publication.output_dir / "01_internal_filters/internal_eligible.parquet"
    )
    assert metadata["rows"] == {"pooled": 2, "unique": 2, "eligible": 2}
    assert set(eligible["seed_pareto_rank"]) == {0, 3}


def test_final_representative_selection_prefers_global_rank(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("pyarrow")
    output = tmp_path / "publication"
    source = output / "03_external_safety"
    source.mkdir(parents=True)
    sequences = ["A" * 10, "C" * 10, "D" * 10]
    pd.DataFrame(
        {
            "sequence": sequences,
            "sequence_sha256": [sequence_sha256(value) for value in sequences],
            "apex_pathogen_median_log10_mic": [1.0, 2.0, 1.5],
            "toxicity_score": [0.1, 0.2, 0.15],
            "hemolysis_score": [0.1, 0.2, 0.15],
        }
    ).to_parquet(source / "eligible.parquet", index=False)

    def assignments(frame, optimization, destination):
        destination.write_text("cluster\tmembers\n")
        hashes = dict(zip(frame.sequence, frame.sequence_sha256, strict=True))
        return {
            hashes[sequences[0]]: "shared",
            hashes[sequences[1]]: "shared",
            hashes[sequences[2]]: "other",
        }

    monkeypatch.setattr("amp_design.publication_pipeline._cluster_assignments", assignments)
    publication = SimpleNamespace(output_dir=output)

    metadata = rank_and_cluster_final(SimpleNamespace(), publication)

    representatives = pd.read_csv(output / "04_final/final_representatives.csv")
    assert metadata["rows"]["representatives"] == 2
    assert sequences[0] in set(representatives["sequence"])
    assert sequences[1] not in set(representatives["sequence"])


def test_publication_manifest_hashes_front_half_and_seed_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = [tmp_path / name for name in ("sample.json", "prepared.parquet", "scored.parquet")]
    ranked = tmp_path / "ranked.parquet"
    initial = tmp_path / "initial.csv"
    seed_dir = tmp_path / "seed_42"
    seed_dir.mkdir()
    completion = seed_dir / "optimization_complete.json"
    for path in [*upstream, ranked, initial, completion]:
        path.write_text(path.name, encoding="utf-8")
    pipeline = SimpleNamespace(
        sampling_manifests=(upstream[0],),
        prepared_candidates=upstream[1],
        scored_candidates=upstream[2],
    )
    optimization = SimpleNamespace(
        ranked_candidates=ranked,
        initial_population=initial,
        effective_seed_directories=(seed_dir,),
    )
    publication = SimpleNamespace(output_dir=tmp_path / "publication")
    for function in (
        "merge_and_filter_internal",
        "score_and_filter_apex_pathogen",
        "apply_external_safety",
        "rank_and_cluster_final",
    ):
        monkeypatch.setattr(
            f"amp_design.publication_pipeline.{function}",
            lambda *args, **kwargs: {"rows": {"eligible": 1}},
        )

    manifest = run_publication_pipeline(
        pipeline, optimization, publication, device=torch.device("cpu")
    )

    expected = {str(path.resolve()) for path in [*upstream, ranked, initial, completion]}
    assert set(manifest["upstream_artifacts"]) == expected
