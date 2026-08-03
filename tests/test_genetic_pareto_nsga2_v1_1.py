import numpy as np
import pandas as pd

from amp_design.genetic_pareto import (
    AMINO_ACIDS,
    allowed_initial_edits,
    amino_acid_frequencies,
    assign_rank_and_crowding,
    constrained_binary_tournament,
    crossover_sequences,
    generate_offspring,
    hash_sorted_fasta_records,
    levenshtein_distance,
    mutate_sequence,
    select_parent_pair,
)


def _population() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sequence": ["A" * 10, "C" * 10, "D" * 10, "E" * 11],
            "length": [10, 10, 10, 11],
            "sequence_sha256": ["d", "c", "b", "a"],
            "f1": [0.0, 1.0, 2.0, 3.0],
            "f2": [3.0, 2.0, 1.0, 0.0],
        }
    )
    return assign_rank_and_crowding(frame, ["f1", "f2"])


def test_training_frequencies_follow_observed_residue_counts() -> None:
    frequencies = amino_acid_frequencies(["AAAA", "AC"])
    assert frequencies[AMINO_ACIDS.index("A")] == 5 / 6
    assert frequencies[AMINO_ACIDS.index("C")] == 1 / 6


def test_pareto_ranking_supports_maximize_direction() -> None:
    frame = pd.DataFrame(
        {
            "sequence": ["A" * 10, "C" * 10],
            "activity": [0.1, 0.9],
        }
    )

    ranked = assign_rank_and_crowding(
        frame,
        ["activity"],
        directions=["maximize"],
    )

    assert ranked.loc[ranked["activity"].idxmax(), "pareto_rank"] == 0
    assert ranked.loc[ranked["activity"].idxmin(), "pareto_rank"] == 1


def test_mutation_preserves_contract_and_converts_boundaries() -> None:
    frequencies = np.ones(len(AMINO_ACIDS)) / len(AMINO_ACIDS)
    rng = np.random.default_rng(42)
    observed_boundary_conversion = False
    for _ in range(500):
        child, metadata = mutate_sequence("A" * 30, rng, frequencies)
        assert set(child).issubset(set(AMINO_ACIDS))
        assert 10 <= len(child) <= 30
        assert child != "A" * 30
        if metadata["raw_operator"] == "insertion":
            assert metadata["operator"] == "substitution"
            assert metadata["boundary_conversion"]
            observed_boundary_conversion = True
            break
    assert observed_boundary_conversion


def test_crossover_rejects_mixed_length_parents() -> None:
    with np.testing.assert_raises(ValueError):
        crossover_sequences("A" * 10, "C" * 11, np.random.default_rng(1))


def test_constrained_tournament_prefers_feasibility_then_rank_and_crowding() -> None:
    population = _population()
    population.loc[0, "formal_genetic_feasible"] = False
    population.loc[0, "constraint_violation"] = 0.01
    assert constrained_binary_tournament(population, [0, 1]) == 1
    population.loc[0, "formal_genetic_feasible"] = True
    population.loc[0, "pareto_rank"] = 2
    population.loc[1, "pareto_rank"] = 0
    assert constrained_binary_tournament(population, [0, 1]) == 1


def test_parent_pair_is_always_exact_length() -> None:
    population = _population()
    rng = np.random.default_rng(9)
    for _ in range(50):
        first, second = select_parent_pair(population, rng)
        if second is not None:
            assert population.iloc[first].length == population.iloc[second].length


def test_edit_distance_and_length_scaled_budget() -> None:
    assert levenshtein_distance("ACDE", "ACDE") == 0
    assert levenshtein_distance("ACDE", "ACKE") == 1
    assert levenshtein_distance("ACDE", "ACDKE") == 1
    assert levenshtein_distance("ACDE", "ACE") == 1
    assert allowed_initial_edits(
        "A" * 10, max_initial_edit_fraction=0.25, max_initial_edits=5
    ) == 2
    assert allowed_initial_edits(
        "A" * 30, max_initial_edit_fraction=0.25, max_initial_edits=5
    ) == 5


def test_parent_pair_can_be_restricted_to_the_same_initial_anchor() -> None:
    population = _population()
    population["anchor_initial_sequence"] = ["anchor-a", "anchor-a", "anchor-b", "anchor-c"]
    rng = np.random.default_rng(19)
    for _ in range(50):
        first, second = select_parent_pair(
            population,
            rng,
            anchor_column="anchor_initial_sequence",
        )
        if second is not None:
            assert (
                population.iloc[first].anchor_initial_sequence
                == population.iloc[second].anchor_initial_sequence
            )


def test_offspring_lineage_records_both_parents_and_attempt_limit() -> None:
    population = _population()
    frequencies = np.ones(len(AMINO_ACIDS)) / len(AMINO_ACIDS)
    history = set(population.sequence)
    children, lineage = generate_offspring(
        population,
        np.random.default_rng(22),
        frequencies,
        count=20,
        historical_sequences=history,
        generation=3,
        seed=42,
        max_attempts=20,
    )
    assert len(children) == len(set(children)) == 20
    assert all(row["parent_a"] and row["parent_b"] for row in lineage)
    assert all(1 <= row["proposal_attempt"] <= 20 for row in lineage)


def test_offspring_respects_external_forbidden_sequences() -> None:
    population = _population()
    frequencies = np.ones(len(AMINO_ACIDS)) / len(AMINO_ACIDS)
    forbidden = {"A" * 9 + amino_acid for amino_acid in AMINO_ACIDS}
    children, _ = generate_offspring(
        population,
        np.random.default_rng(7),
        frequencies,
        count=10,
        historical_sequences=set(population.sequence),
        forbidden_sequences=forbidden,
        generation=1,
        seed=42,
        max_attempts=100,
    )
    assert set(children).isdisjoint(forbidden)


def test_offspring_respects_initial_anchor_edit_budget() -> None:
    population = _population()
    population["anchor_initial_sequence"] = population["sequence"]
    frequencies = np.ones(len(AMINO_ACIDS)) / len(AMINO_ACIDS)
    children, lineage = generate_offspring(
        population,
        np.random.default_rng(31),
        frequencies,
        count=20,
        historical_sequences=set(population.sequence),
        generation=1,
        seed=42,
        max_attempts=100,
        enforce_initial_edit_distance=True,
        max_initial_edit_fraction=0.25,
        max_initial_edits=5,
        same_anchor_crossover=True,
    )
    assert len(children) == 20
    for row in lineage:
        assert row["initial_edit_distance_pass"]
        assert row["initial_edit_distance"] <= row["max_allowed_initial_edits"]
        assert row["parent_a_anchor_initial_sequence"] == row["anchor_initial_sequence"]
        assert row["parent_b_anchor_initial_sequence"] == row["anchor_initial_sequence"]


def test_mmseqs_fasta_records_are_hash_sorted_independent_of_rank() -> None:
    population = _population()
    records = hash_sorted_fasta_records(population)
    assert [identifier for identifier, _ in records] == ["a", "b", "c", "d"]
