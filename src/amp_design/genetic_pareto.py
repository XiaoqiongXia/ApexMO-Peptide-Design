"""Deterministic sequence operators for constrained genetic Pareto search.

This module intentionally has no model-inference dependencies so the frozen
operators and selection contract can be unit tested in a lightweight runtime.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MIN_LENGTH = 10
MAX_LENGTH = 30


class OffspringExhaustion(RuntimeError):
    """Raised when a requested offspring cannot be proposed within the limit."""


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def amino_acid_frequencies(sequences: Iterable[str]) -> np.ndarray:
    """Return canonical residue frequencies in ``AMINO_ACIDS`` order."""

    counts = np.zeros(len(AMINO_ACIDS), dtype=np.float64)
    lookup = {residue: index for index, residue in enumerate(AMINO_ACIDS)}
    for sequence in sequences:
        value = str(sequence).strip().upper()
        if not value or not set(value).issubset(lookup):
            raise ValueError(f"Noncanonical sequence in frozen AMP training set: {value!r}")
        for residue in value:
            counts[lookup[residue]] += 1
    if counts.sum() == 0:
        raise ValueError("Frozen AMP training set contains no residues")
    return counts / counts.sum()


def validate_residue_frequencies(frequencies: Sequence[float]) -> np.ndarray:
    values = np.asarray(frequencies, dtype=np.float64)
    if values.shape != (len(AMINO_ACIDS),):
        raise ValueError(f"Expected {len(AMINO_ACIDS)} amino-acid frequencies")
    if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("Amino-acid frequencies must be finite, nonnegative, and nonzero")
    return values / values.sum()


def _draw_replacement(
    current: str,
    rng: np.random.Generator,
    frequencies: np.ndarray,
) -> str:
    probabilities = frequencies.copy()
    probabilities[AMINO_ACIDS.index(current)] = 0.0
    if probabilities.sum() <= 0:
        raise ValueError("Replacement distribution has no mass outside current residue")
    probabilities /= probabilities.sum()
    return str(rng.choice(list(AMINO_ACIDS), p=probabilities))


def mutate_sequence(
    sequence: str,
    rng: np.random.Generator,
    frequencies: Sequence[float],
) -> tuple[str, dict[str, Any]]:
    """Apply the frozen one-event mutation contract.

    Insertion at length 30 and deletion at length 10 are converted directly to
    substitution while retaining the original draw in ``raw_operator``.
    """

    sequence = str(sequence).upper()
    if not set(sequence).issubset(set(AMINO_ACIDS)):
        raise ValueError("Mutation parent must contain canonical amino acids only")
    if not MIN_LENGTH <= len(sequence) <= MAX_LENGTH:
        raise ValueError(f"Mutation parent length must be in [{MIN_LENGTH}, {MAX_LENGTH}]")
    residue_frequencies = validate_residue_frequencies(frequencies)
    raw_operator = str(rng.choice(["substitution", "insertion", "deletion"], p=[0.80, 0.10, 0.10]))
    operator = raw_operator
    boundary_conversion = False
    if (raw_operator == "insertion" and len(sequence) == MAX_LENGTH) or (
        raw_operator == "deletion" and len(sequence) == MIN_LENGTH
    ):
        operator = "substitution"
        boundary_conversion = True

    positions: list[int] = []
    removed: list[str] = []
    inserted: list[str] = []
    if operator == "substitution":
        n_edits = int(rng.integers(1, min(3, len(sequence)) + 1))
        positions = sorted(
            int(value) for value in rng.choice(len(sequence), n_edits, replace=False)
        )
        chars = list(sequence)
        for position in positions:
            old = chars[position]
            new = _draw_replacement(old, rng, residue_frequencies)
            chars[position] = new
            removed.append(old)
            inserted.append(new)
        child = "".join(chars)
    elif operator == "insertion":
        position = int(rng.integers(0, len(sequence) + 1))
        residue = str(rng.choice(list(AMINO_ACIDS), p=residue_frequencies))
        child = sequence[:position] + residue + sequence[position:]
        positions = [position]
        inserted = [residue]
    else:
        position = int(rng.integers(0, len(sequence)))
        child = sequence[:position] + sequence[position + 1 :]
        positions = [position]
        removed = [sequence[position]]

    if child == sequence:
        raise RuntimeError("Frozen mutation contract produced an unchanged sequence")
    return child, {
        "raw_operator": raw_operator,
        "operator": operator,
        "boundary_conversion": boundary_conversion,
        "positions": positions,
        "removed": removed,
        "inserted": inserted,
    }


def crossover_sequences(
    parent_a: str,
    parent_b: str,
    rng: np.random.Generator,
    *,
    probability: float = 0.70,
) -> tuple[str, str, dict[str, Any]]:
    """Produce reciprocal exact-length children using one-point crossover."""

    if len(parent_a) != len(parent_b):
        raise ValueError("Crossover parents must belong to the same exact-length stratum")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Crossover probability must be within [0, 1]")
    crossover_draw = bool(rng.random() < probability)
    if len(parent_a) < 2 or not crossover_draw:
        return (
            parent_a,
            parent_b,
            {
                "crossover_operator": "no_crossover",
                "crossover_breakpoint": None,
                "crossover_draw": crossover_draw,
            },
        )
    breakpoint = int(rng.integers(1, len(parent_a)))
    return (
        parent_a[:breakpoint] + parent_b[breakpoint:],
        parent_b[:breakpoint] + parent_a[breakpoint:],
        {
            "crossover_operator": "one_point",
            "crossover_breakpoint": breakpoint,
            "crossover_draw": crossover_draw,
        },
    )


def assign_rank_and_crowding(
    frame: pd.DataFrame,
    objectives: Sequence[str],
    *,
    feasible: Sequence[bool] | None = None,
    directions: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Assign zero-based feasible Pareto rank and per-front crowding distance."""

    result = frame.copy()
    values = result[list(objectives)].to_numpy(dtype=float)
    if directions is not None:
        if len(directions) != len(objectives):
            raise ValueError("Objective directions must match objective columns")
        invalid = sorted(set(directions).difference({"minimize", "maximize"}))
        if invalid:
            raise ValueError(f"Unsupported objective directions: {invalid}")
        signs = np.asarray(
            [1.0 if direction == "minimize" else -1.0 for direction in directions],
            dtype=float,
        )
        values = values * signs
    finite = np.isfinite(values).all(axis=1)
    if feasible is None:
        feasible_values = finite
    else:
        feasible_values = np.asarray(feasible, dtype=bool) & finite
    rank = np.full(len(result), np.iinfo(np.int32).max, dtype=np.int64)
    crowding = np.full(len(result), -np.inf, dtype=np.float64)
    indices = np.flatnonzero(feasible_values)
    if len(indices):
        fronts = NonDominatedSorting().do(values[indices], only_non_dominated_front=False)
        for front_rank, front in enumerate(fronts):
            front_indices = indices[np.asarray(front, dtype=int)]
            rank[front_indices] = front_rank
            crowding[front_indices] = calc_crowding_distance(values[front_indices])
    result["pareto_rank"] = rank
    result["crowding_distance"] = crowding
    result["formal_genetic_feasible"] = feasible_values
    return result


def _comparison_key(row: pd.Series) -> tuple[Any, ...]:
    sequence_hash = str(row.get("sequence_sha256", sequence_sha256(str(row["sequence"]))))
    if bool(row["formal_genetic_feasible"]):
        crowding = float(row["crowding_distance"])
        if np.isnan(crowding):
            crowding = -np.inf
        return (0, int(row["pareto_rank"]), -crowding, sequence_hash)
    violation = float(row.get("constraint_violation", np.inf))
    if not np.isfinite(violation):
        violation = np.inf
    return (1, violation, sequence_hash)


def constrained_binary_tournament(
    population: pd.DataFrame,
    candidate_indices: Sequence[int],
) -> int:
    """Return the positional winner under feasibility, rank/CV, and crowding."""

    if len(candidate_indices) != 2:
        raise ValueError("A binary tournament requires exactly two candidates")
    first, second = (int(value) for value in candidate_indices)
    return min((first, second), key=lambda index: _comparison_key(population.iloc[index]))


def levenshtein_distance(first: str, second: str) -> int:
    """Return unit-cost substitution/insertion/deletion distance."""

    left, right = str(first), str(second)
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for right_index, right_residue in enumerate(right, start=1):
        current = [right_index]
        for left_index, left_residue in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[left_index] + 1,
                    previous[left_index - 1] + (left_residue != right_residue),
                )
            )
        previous = current
    return previous[-1]


def allowed_initial_edits(
    anchor_sequence: str,
    *,
    max_initial_edit_fraction: float,
    max_initial_edits: int,
) -> int:
    """Return the length-scaled absolute edit budget for an initial anchor."""

    if not 0 < max_initial_edit_fraction <= 1:
        raise ValueError("max_initial_edit_fraction must lie within (0, 1]")
    if max_initial_edits < 1:
        raise ValueError("max_initial_edits must be positive")
    scaled = int(np.floor(max_initial_edit_fraction * len(str(anchor_sequence))))
    return min(max_initial_edits, max(1, scaled))


def select_parent_pair(
    population: pd.DataFrame,
    rng: np.random.Generator,
    *,
    anchor_column: str | None = None,
) -> tuple[int, int | None]:
    """Select a constrained-tournament parent and an exact-length mate."""

    if len(population) < 1:
        raise ValueError("Cannot select parents from an empty population")
    if anchor_column is not None and anchor_column not in population:
        raise ValueError(f"Population lacks anchor column {anchor_column!r}")
    first_draw = rng.choice(len(population), size=2, replace=len(population) < 2)
    first = constrained_binary_tournament(population, first_draw)
    length = int(population.iloc[first]["length"])
    anchor = population.iloc[first][anchor_column] if anchor_column is not None else None
    mates = [
        index
        for index in range(len(population))
        if index != first
        and int(population.iloc[index]["length"]) == length
        and (
            anchor_column is None
            or str(population.iloc[index][anchor_column]) == str(anchor)
        )
    ]
    if not mates:
        return first, None
    mate_draw = rng.choice(mates, size=2, replace=len(mates) < 2)
    return first, constrained_binary_tournament(population, mate_draw)


def generate_offspring(
    population: pd.DataFrame,
    rng: np.random.Generator,
    frequencies: Sequence[float],
    *,
    count: int,
    historical_sequences: set[str],
    forbidden_sequences: set[str] | frozenset[str] | None = None,
    generation: int,
    seed: int,
    max_attempts: int = 20,
    enforce_initial_edit_distance: bool = False,
    max_initial_edit_fraction: float = 0.25,
    max_initial_edits: int = 5,
    same_anchor_crossover: bool = True,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Generate unique offspring with a bounded proposal budget per slot."""

    if count < 1 or max_attempts < 1:
        raise ValueError("count and max_attempts must be positive")
    required = {
        "sequence",
        "length",
        "formal_genetic_feasible",
        "pareto_rank",
        "crowding_distance",
    }
    missing = required.difference(population.columns)
    if missing:
        raise ValueError(f"Population lacks selection columns: {sorted(missing)}")
    if enforce_initial_edit_distance and "anchor_initial_sequence" not in population:
        raise ValueError(
            "Population lacks anchor_initial_sequence required by the edit-distance constraint"
        )
    if enforce_initial_edit_distance:
        allowed_initial_edits(
            str(population.iloc[0]["anchor_initial_sequence"]),
            max_initial_edit_fraction=max_initial_edit_fraction,
            max_initial_edits=max_initial_edits,
        )
    children: list[str] = []
    lineage: list[dict[str, Any]] = []
    proposed = set(historical_sequences)
    if forbidden_sequences is not None:
        proposed.update(forbidden_sequences)
    for offspring_index in range(count):
        accepted = False
        for attempt in range(1, max_attempts + 1):
            first_index, second_index = select_parent_pair(
                population,
                rng,
                anchor_column=(
                    "anchor_initial_sequence"
                    if enforce_initial_edit_distance and same_anchor_crossover
                    else None
                ),
            )
            parent_a = str(population.iloc[first_index]["sequence"])
            parent_b = (
                parent_a if second_index is None else str(population.iloc[second_index]["sequence"])
            )
            reciprocal_a, reciprocal_b, crossover_meta = crossover_sequences(
                parent_a, parent_b, rng
            )
            reciprocal_index = offspring_index % 2
            pre_mutation = (reciprocal_a, reciprocal_b)[reciprocal_index]
            child, mutation_meta = mutate_sequence(pre_mutation, rng, frequencies)
            child = child.upper()
            anchor_metadata: dict[str, Any] = {}
            distance_valid = True
            if enforce_initial_edit_distance:
                parent_a_anchor = str(
                    population.iloc[first_index]["anchor_initial_sequence"]
                )
                parent_b_anchor = (
                    parent_a_anchor
                    if second_index is None
                    else str(population.iloc[second_index]["anchor_initial_sequence"])
                )
                if parent_a_anchor == parent_b_anchor:
                    anchor = parent_a_anchor
                    anchor_source = "shared_parent_anchor"
                elif crossover_meta["crossover_operator"] == "no_crossover":
                    anchor = (parent_a_anchor, parent_b_anchor)[reciprocal_index]
                    anchor_source = "selected_reciprocal_parent"
                else:
                    breakpoint = int(crossover_meta["crossover_breakpoint"])
                    contribution_a = (
                        breakpoint if reciprocal_index == 0 else len(parent_a) - breakpoint
                    )
                    contribution_b = len(parent_b) - contribution_a
                    if contribution_a > contribution_b:
                        anchor = parent_a_anchor
                        anchor_source = "larger_parent_a_crossover_contribution"
                    elif contribution_b > contribution_a:
                        anchor = parent_b_anchor
                        anchor_source = "larger_parent_b_crossover_contribution"
                    else:
                        anchor = min(parent_a_anchor, parent_b_anchor)
                        anchor_source = "equal_crossover_contribution_lexical_tie_break"
                edit_distance = levenshtein_distance(child, anchor)
                edit_budget = allowed_initial_edits(
                    anchor,
                    max_initial_edit_fraction=max_initial_edit_fraction,
                    max_initial_edits=max_initial_edits,
                )
                distance_valid = edit_distance <= edit_budget
                anchor_metadata = {
                    "parent_a_anchor_initial_sequence": parent_a_anchor,
                    "parent_b_anchor_initial_sequence": parent_b_anchor,
                    "anchor_initial_sequence": anchor,
                    "anchor_initial_sequence_sha256": sequence_sha256(anchor),
                    "anchor_source": anchor_source,
                    "initial_edit_distance": edit_distance,
                    "normalized_initial_edit_distance": edit_distance
                    / max(len(anchor), len(child)),
                    "max_allowed_initial_edits": edit_budget,
                    "initial_edit_distance_pass": distance_valid,
                }
            valid = (
                MIN_LENGTH <= len(child) <= MAX_LENGTH
                and set(child).issubset(set(AMINO_ACIDS))
                and child != pre_mutation
                and child not in proposed
                and distance_valid
            )
            if not valid:
                continue
            proposed.add(child)
            historical_sequences.add(child)
            children.append(child)
            lineage.append(
                {
                    "generation": generation,
                    "seed": seed,
                    "offspring_index": offspring_index,
                    "proposal_attempt": attempt,
                    "parent_a": parent_a,
                    "parent_b": parent_b,
                    "parent_a_sha256": sequence_sha256(parent_a),
                    "parent_b_sha256": sequence_sha256(parent_b),
                    "reciprocal_child_index": reciprocal_index,
                    "pre_mutation_sequence": pre_mutation,
                    "sequence": child,
                    "sequence_sha256": sequence_sha256(child),
                    **anchor_metadata,
                    **crossover_meta,
                    **mutation_meta,
                }
            )
            accepted = True
            break
        if not accepted:
            raise OffspringExhaustion(
                f"offspring_exhaustion: slot {offspring_index} exceeded {max_attempts} attempts"
            )
    return children, lineage


def survivor_order(frame: pd.DataFrame) -> list[int]:
    """Return the frozen constrained NSGA-II traversal order."""

    return sorted(range(len(frame)), key=lambda index: _comparison_key(frame.iloc[index]))


def hash_sorted_fasta_records(frame: pd.DataFrame) -> list[tuple[str, str]]:
    """Return FASTA identifiers/sequences in protocol-required SHA-256 order."""

    required = {"sequence", "sequence_sha256"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"FASTA frame lacks columns: {sorted(missing)}")
    ordered = frame.sort_values("sequence_sha256", kind="mergesort")
    return [(str(row.sequence_sha256), str(row.sequence)) for row in ordered.itertuples()]
