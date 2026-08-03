"""Validated pooling and ranking for multi-seed genetic Pareto archives."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .genetic_pareto import (
    AMINO_ACIDS,
    MAX_LENGTH,
    MIN_LENGTH,
    assign_rank_and_crowding,
    sequence_sha256,
)


def _validate_frame(frame: pd.DataFrame, objectives: Sequence[str], label: str) -> pd.DataFrame:
    required = {"sequence", *objectives}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{label} lacks required columns: {sorted(missing)}")
    result = frame.copy()
    result["sequence"] = result["sequence"].astype(str).str.strip().str.upper()
    derived_hash = result["sequence"].map(sequence_sha256)
    if (
        "sequence_sha256" in result
        and not result["sequence_sha256"].astype(str).eq(derived_hash).all()
    ):
        raise ValueError(f"{label} contains sequence SHA-256 mismatches")
    result["sequence_sha256"] = derived_hash
    result["length"] = result["sequence"].str.len()
    return result


def finalize_combined_archive(
    original: pd.DataFrame,
    evolved_by_seed: Sequence[pd.DataFrame],
    objectives: Sequence[str],
    *,
    original_eligibility_column: str | None = "formal_pareto_eligible",
    evolved_eligibility_column: str | None = None,
    directions: Sequence[str] | None = None,
    score_tolerance: float | Mapping[str, float] = 1e-6,
    front_only: bool = False,
    front_chunk_size: int = 4096,
) -> pd.DataFrame:
    """Pool original and all evaluated evolved sequences, de-duplicate, and re-rank."""

    original_validated = _validate_frame(original, objectives, "original comparison pool")
    if original_eligibility_column is not None:
        if original_eligibility_column not in original_validated:
            raise ValueError(f"Original comparison pool lacks {original_eligibility_column!r}")
        original_validated = original_validated.loc[
            original_validated[original_eligibility_column].eq(True)
        ].copy()
    original_validated["archive_source"] = "original"
    original_validated["genetic_seed"] = pd.NA

    parts = [original_validated]
    observed_seeds: set[int] = set()
    for index, frame in enumerate(evolved_by_seed):
        evolved = _validate_frame(frame, objectives, f"evolved seed frame {index}")
        if "genetic_seed" not in evolved:
            raise ValueError(f"Evolved seed frame {index} lacks genetic_seed")
        seeds = set(pd.to_numeric(evolved["genetic_seed"], errors="raise").astype(int))
        if len(seeds) != 1:
            raise ValueError(f"Evolved seed frame {index} must contain exactly one seed")
        seed = next(iter(seeds))
        if seed in observed_seeds:
            raise ValueError(f"Duplicate evolved cache for seed {seed}")
        observed_seeds.add(seed)
        if evolved_eligibility_column is not None:
            if evolved_eligibility_column not in evolved:
                raise ValueError(f"Evolved seed frame {index} lacks {evolved_eligibility_column!r}")
            evolved = evolved.loc[evolved[evolved_eligibility_column].eq(True)].copy()
        evolved["genetic_seed"] = seed
        evolved["archive_source"] = "evolved"
        parts.append(evolved)
    if not observed_seeds:
        raise ValueError("At least one evolved seed cache is required")

    combined = pd.concat(parts, ignore_index=True, sort=False)
    objective_array = combined[list(objectives)].to_numpy(dtype=float)
    finite_rows = np.isfinite(objective_array).all(axis=1)
    finite = combined.loc[finite_rows, ["sequence_sha256", *objectives]]
    statistics = finite.groupby("sequence_sha256", sort=False)[list(objectives)].agg(["min", "max"])
    spread = statistics.xs("max", axis=1, level=1) - statistics.xs("min", axis=1, level=1)
    if isinstance(score_tolerance, Mapping):
        missing_tolerances = set(objectives).difference(score_tolerance)
        if missing_tolerances:
            raise ValueError(
                f"Missing score tolerances for objectives: {sorted(missing_tolerances)}"
            )
        tolerance_by_objective = pd.Series(
            {objective: float(score_tolerance[objective]) for objective in objectives}
        )
        conflicts = spread.gt(tolerance_by_objective, axis="columns").any(axis=1)
    else:
        conflicts = spread.gt(float(score_tolerance)).any(axis=1)
    if conflicts.any():
        sequence_hash = str(conflicts.index[np.flatnonzero(conflicts.to_numpy())[0]])
        raise ValueError(f"Conflicting objective values for {sequence_hash}")

    combined["_source_order"] = combined["archive_source"].map({"original": 0, "evolved": 1})
    unique = (
        combined.sort_values(["sequence_sha256", "_source_order"], kind="mergesort")
        .drop_duplicates("sequence_sha256", keep="first")
        .drop(columns="_source_order")
        .reset_index(drop=True)
    )
    hashes = unique["sequence_sha256"].astype(str)
    duplicate_counts = combined.groupby("sequence_sha256", sort=False).size()
    unique["archive_duplicate_rows"] = hashes.map(duplicate_counts).to_numpy()
    original_hashes = set(
        combined.loc[combined["archive_source"].eq("original"), "sequence_sha256"]
    )
    evolved_hashes = set(combined.loc[combined["archive_source"].eq("evolved"), "sequence_sha256"])
    unique["archive_source"] = [
        "original+evolved"
        if sequence_hash in original_hashes and sequence_hash in evolved_hashes
        else "original"
        if sequence_hash in original_hashes
        else "evolved"
        for sequence_hash in hashes
    ]
    seed_hashes = {
        seed: set(
            combined.loc[
                pd.to_numeric(combined["genetic_seed"], errors="coerce").eq(seed),
                "sequence_sha256",
            ]
        )
        for seed in sorted(observed_seeds)
    }
    unique["genetic_seeds_json"] = [
        json.dumps([seed for seed in sorted(observed_seeds) if sequence_hash in seed_hashes[seed]])
        for sequence_hash in hashes
    ]
    objective_values = unique[list(objectives)].to_numpy(dtype=float)
    if directions is not None:
        if len(directions) != len(objectives):
            raise ValueError("Objective directions must match objective columns")
        invalid = sorted(set(directions).difference({"minimize", "maximize"}))
        if invalid:
            raise ValueError(f"Unsupported objective directions: {invalid}")
        signs = np.asarray([1.0 if direction == "minimize" else -1.0 for direction in directions])
        objective_values = objective_values * signs
    feasible = (
        unique["sequence"].str.fullmatch(f"[{AMINO_ACIDS}]+", na=False).to_numpy()
        & unique["length"].between(MIN_LENGTH, MAX_LENGTH).to_numpy()
        & np.isfinite(objective_values).all(axis=1)
    )
    if front_only:
        if front_chunk_size < 1:
            raise ValueError("front_chunk_size must be positive")
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

        feasible_indices = np.flatnonzero(feasible)
        archive = np.asarray([], dtype=int)
        for start in range(0, len(feasible_indices), front_chunk_size):
            candidates = np.concatenate(
                [archive, feasible_indices[start : start + front_chunk_size]]
            )
            local_front = NonDominatedSorting().do(
                objective_values[candidates], only_non_dominated_front=True
            )
            archive = candidates[np.asarray(local_front, dtype=int)]
        ranked = unique.copy()
        ranked["formal_genetic_feasible"] = feasible
        ranked["pareto_rank"] = -1
        ranked["crowding_distance"] = -np.inf
        if len(archive):
            front_ranked = assign_rank_and_crowding(
                unique.iloc[archive].copy(),
                objectives,
                feasible=np.ones(len(archive), dtype=bool),
                directions=directions,
            )
            ranked.loc[archive, "pareto_rank"] = 0
            ranked.loc[archive, "crowding_distance"] = front_ranked["crowding_distance"].to_numpy()
    else:
        ranked = assign_rank_and_crowding(
            unique,
            objectives,
            feasible=feasible,
            directions=directions,
        )
    ranked["combined_archive_front"] = ranked["formal_genetic_feasible"] & ranked["pareto_rank"].eq(
        0
    )
    return ranked.sort_values(
        [
            "combined_archive_front",
            "formal_genetic_feasible",
            "pareto_rank",
            "crowding_distance",
            "sequence_sha256",
        ],
        ascending=[False, False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def apply_cluster_cap(
    ranked_front: pd.DataFrame,
    cluster_by_hash: dict[str, str],
    *,
    max_per_cluster: int = 5,
) -> pd.DataFrame:
    """Apply a deterministic cluster cap after combined-set Pareto ranking."""

    if max_per_cluster < 1:
        raise ValueError("max_per_cluster must be positive")
    order = ranked_front.sort_values(
        ["pareto_rank", "crowding_distance", "sequence_sha256"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    counts: dict[str, int] = {}
    selected: list[int] = []
    for index, row in order.iterrows():
        sequence_hash = str(row["sequence_sha256"])
        cluster = cluster_by_hash.get(sequence_hash, sequence_hash)
        if counts.get(cluster, 0) >= max_per_cluster:
            continue
        counts[cluster] = counts.get(cluster, 0) + 1
        selected.append(index)
    result = ranked_front.loc[selected].copy()
    result["final_cluster_id"] = (
        result["sequence_sha256"].map(cluster_by_hash).fillna(result["sequence_sha256"])
    )
    return result.reset_index(drop=True)
