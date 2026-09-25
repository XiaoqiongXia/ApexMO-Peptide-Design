"""Constraint-aware Pareto ranking for predicted AMP candidate properties."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pymoo
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from amp_design.optimization.objectives import (
    ConstraintSpec,
    ObjectiveSpec,
    parse_constraint,
    parse_objective,
)


def rank_candidates(
    candidates: pd.DataFrame,
    objectives: Sequence[ObjectiveSpec],
    constraints: Sequence[ConstraintSpec] = (),
) -> pd.DataFrame:
    """Rank scored candidates, treating every pymoo objective as minimization.

    Rows with missing/non-finite objectives or failed constraints are retained for
    audit but excluded from non-dominated sorting.
    """

    if len(objectives) < 2:
        raise ValueError("Pareto ranking requires at least two objectives")
    columns = {spec.column for spec in [*objectives, *constraints]}
    missing = columns.difference(candidates.columns)
    if missing:
        raise ValueError(f"Missing objective/constraint columns: {sorted(missing)}")
    objective_names = [objective.column for objective in objectives]
    if len(set(objective_names)) != len(objective_names):
        raise ValueError("Objective columns must be unique")

    frame = candidates.copy().reset_index(drop=True)
    frame["candidate_index"] = np.arange(len(frame), dtype=int)
    objective_matrix = np.column_stack(
        [objective.to_minimization(frame[objective.column]) for objective in objectives]
    )
    complete = np.isfinite(objective_matrix).all(axis=1)
    violations = np.zeros(len(frame), dtype=int)
    for constraint in constraints:
        violations += ~constraint.satisfied(frame[constraint.column])
    feasible = complete & (violations == 0)

    frame["objective_complete"] = complete
    frame["constraint_violation_count"] = violations
    frame["feasible"] = feasible
    frame["pareto_rank"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["crowding_distance"] = np.nan
    frame["pareto_front"] = False

    feasible_indices = np.flatnonzero(feasible)
    if len(feasible_indices):
        feasible_objectives = objective_matrix[feasible_indices]
        fronts = NonDominatedSorting().do(feasible_objectives)
        for rank, front in enumerate(fronts):
            indices = feasible_indices[front]
            frame.loc[indices, "pareto_rank"] = rank
            frame.loc[indices, "crowding_distance"] = calc_crowding_distance(
                feasible_objectives[front]
            )
            if rank == 0:
                frame.loc[indices, "pareto_front"] = True

    return frame.sort_values(
        ["feasible", "pareto_rank", "crowding_distance", "candidate_index"],
        ascending=[False, True, False, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def pareto_summary(
    ranked: pd.DataFrame,
    objectives: Sequence[ObjectiveSpec],
    constraints: Sequence[ConstraintSpec],
) -> dict[str, Any]:
    ranks = ranked.loc[ranked["feasible"], "pareto_rank"].dropna().astype(int)
    return {
        "pymoo_version": pymoo.__version__,
        "n_candidates": int(len(ranked)),
        "n_feasible": int(ranked["feasible"].sum()),
        "n_objective_incomplete": int((~ranked["objective_complete"]).sum()),
        "n_pareto_front": int(ranked["pareto_front"].sum()),
        "front_sizes": {str(rank): count for rank, count in sorted(Counter(ranks).items())},
        "objectives": [
            {"column": objective.column, "direction": objective.direction}
            for objective in objectives
        ],
        "constraints": [
            {
                "column": constraint.column,
                "operator": constraint.operator,
                "threshold": constraint.threshold,
            }
            for constraint in constraints
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--front-output", type=Path)
    parser.add_argument(
        "--objective",
        action="append",
        required=True,
        metavar="COLUMN:min|max",
        help="Repeat for each objective; at least two are required.",
    )
    parser.add_argument(
        "--constraint",
        action="append",
        default=[],
        metavar="EXPRESSION",
        help="Optional per-row constraint, for example toxicity<=0.2.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    objectives = [parse_objective(specification) for specification in args.objective]
    constraints = [parse_constraint(specification) for specification in args.constraint]
    ranked = rank_candidates(pd.read_csv(args.input), objectives, constraints)
    summary = pareto_summary(ranked, objectives, constraints)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output, index=False)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.front_output is not None:
        args.front_output.parent.mkdir(parents=True, exist_ok=True)
        ranked.loc[ranked["pareto_front"]].to_csv(args.front_output, index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
