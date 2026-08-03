import pandas as pd
import pytest

from amp_design.genetic_pareto import sequence_sha256
from amp_design.pareto_finalization import finalize_combined_archive

OBJECTIVES = ["activity", "toxicity", "hemolysis"]


def _row(sequence: str, values: tuple[float, float, float], **extra):
    return {
        "sequence": sequence,
        "sequence_sha256": sequence_sha256(sequence),
        "length": len(sequence),
        **dict(zip(OBJECTIVES, values, strict=True)),
        **extra,
    }


def test_finalization_recomputes_dominance_across_seeds_and_original_pool() -> None:
    original = pd.DataFrame([_row("A" * 10, (1.0, 1.0, 1.0), formal_pareto_eligible=True)])
    seed42 = pd.DataFrame([_row("C" * 10, (2.0, 2.0, 2.0), genetic_seed=42)])
    seed123 = pd.DataFrame([_row("D" * 10, (0.5, 1.5, 1.5), genetic_seed=123)])
    ranked = finalize_combined_archive(original, [seed42, seed123], OBJECTIVES)
    dominated = ranked.set_index("sequence").loc["C" * 10]
    assert dominated["pareto_rank"] > 0
    assert not dominated["combined_archive_front"]
    assert set(ranked.loc[ranked.combined_archive_front, "sequence"]) == {
        "A" * 10,
        "D" * 10,
    }


def test_finalization_rejects_conflicting_cached_scores() -> None:
    original = pd.DataFrame([_row("A" * 10, (1.0, 1.0, 1.0), formal_pareto_eligible=True)])
    seed42 = pd.DataFrame([_row("C" * 10, (2.0, 2.0, 2.0), genetic_seed=42)])
    seed123 = pd.DataFrame([_row("C" * 10, (2.1, 2.0, 2.0), genetic_seed=123)])
    with pytest.raises(ValueError, match="Conflicting objective"):
        finalize_combined_archive(original, [seed42, seed123], OBJECTIVES)


def test_incremental_front_matches_full_non_dominated_sorting() -> None:
    original = pd.DataFrame(
        [
            _row("A" * 10, (1.0, 1.0, 1.0), formal_pareto_eligible=True),
            _row("C" * 10, (0.5, 2.0, 2.0), formal_pareto_eligible=True),
        ]
    )
    seed42 = pd.DataFrame(
        [
            _row("D" * 10, (2.0, 2.0, 2.0), genetic_seed=42),
            _row("E" * 10, (2.0, 0.5, 2.0), genetic_seed=42),
        ]
    )
    seed123 = pd.DataFrame(
        [
            _row("F" * 10, (2.0, 2.0, 0.5), genetic_seed=123),
            _row("G" * 10, (1.0, 1.0, 1.0), genetic_seed=123),
        ]
    )
    full = finalize_combined_archive(original, [seed42, seed123], OBJECTIVES)
    incremental = finalize_combined_archive(
        original,
        [seed42, seed123],
        OBJECTIVES,
        front_only=True,
        front_chunk_size=2,
    )
    assert set(full.loc[full["combined_archive_front"], "sequence"]) == set(
        incremental.loc[incremental["combined_archive_front"], "sequence"]
    )


def test_finalization_excludes_infeasible_evolved_sequences() -> None:
    original = pd.DataFrame([_row("A" * 10, (1.0, 1.0, 1.0), formal_pareto_eligible=True)])
    seed42 = pd.DataFrame(
        [
            _row(
                "C" * 10,
                (0.1, 0.1, 0.1),
                genetic_seed=42,
                formal_genetic_feasible=False,
            )
        ]
    )

    ranked = finalize_combined_archive(
        original,
        [seed42],
        OBJECTIVES,
        evolved_eligibility_column="formal_genetic_feasible",
    )

    assert ranked["sequence"].tolist() == ["A" * 10]


def test_finalization_respects_maximize_direction() -> None:
    original = pd.DataFrame([_row("A" * 10, (0.1, 1.0, 1.0), formal_pareto_eligible=True)])
    seed42 = pd.DataFrame([_row("C" * 10, (0.9, 1.0, 1.0), genetic_seed=42)])

    ranked = finalize_combined_archive(
        original,
        [seed42],
        OBJECTIVES,
        directions=["maximize", "minimize", "minimize"],
    )

    assert ranked.loc[ranked["combined_archive_front"], "sequence"].tolist() == ["C" * 10]
