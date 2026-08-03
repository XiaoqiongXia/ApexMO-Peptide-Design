import numpy as np
import pandas as pd
import pytest

from amp_design.objectives import parse_constraint, parse_objective
from amp_design.pareto import pareto_summary, rank_candidates


def test_parse_objective_and_constraint() -> None:
    objective = parse_objective("activity:max")
    constraint = parse_constraint("toxicity <= 0.2")
    assert (objective.column, objective.direction) == ("activity", "max")
    assert (constraint.column, constraint.operator, constraint.threshold) == (
        "toxicity",
        "<=",
        0.2,
    )
    with pytest.raises(ValueError, match="Invalid objective"):
        parse_objective("activity")
    with pytest.raises(ValueError, match="Invalid constraint"):
        parse_constraint("toxicity around 0.2")


def test_constraint_aware_pareto_ranking() -> None:
    candidates = pd.DataFrame(
        {
            "sequence": ["AAAAA", "CCCCC", "DDDDD", "EEEEE", "FFFFF"],
            "activity": [10.0, 8.0, 6.0, 9.0, np.nan],
            "toxicity": [5.0, 3.0, 4.0, 6.0, 1.0],
            "charge": [3.0, 2.0, 4.0, 3.0, 3.0],
        }
    )
    objectives = [parse_objective("activity:max"), parse_objective("toxicity:min")]
    constraints = [parse_constraint("charge>=2")]
    ranked = rank_candidates(candidates, objectives, constraints)
    by_sequence = ranked.set_index("sequence")

    assert by_sequence.loc["AAAAA", "pareto_rank"] == 0
    assert by_sequence.loc["CCCCC", "pareto_rank"] == 0
    assert by_sequence.loc["DDDDD", "pareto_rank"] == 1
    assert by_sequence.loc["EEEEE", "pareto_rank"] == 1
    assert bool(by_sequence.loc["FFFFF", "feasible"]) is False
    assert pd.isna(by_sequence.loc["FFFFF", "pareto_rank"])
    assert np.isinf(by_sequence.loc["AAAAA", "crowding_distance"])

    summary = pareto_summary(ranked, objectives, constraints)
    assert summary["n_candidates"] == 5
    assert summary["n_feasible"] == 4
    assert summary["n_pareto_front"] == 2
    assert summary["front_sizes"] == {"0": 2, "1": 2}


def test_pareto_requires_two_unique_objectives() -> None:
    candidates = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="at least two"):
        rank_candidates(candidates, [parse_objective("x:min")])
    with pytest.raises(ValueError, match="unique"):
        rank_candidates(candidates, [parse_objective("x:min"), parse_objective("x:max")])
