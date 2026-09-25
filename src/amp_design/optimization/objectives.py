"""Declarative objectives and constraints for scored AMP candidates."""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Direction = Literal["min", "max"]


@dataclass(frozen=True)
class ObjectiveSpec:
    """A numeric candidate column and whether it should be minimized or maximized."""

    column: str
    direction: Direction

    def to_minimization(self, values: pd.Series) -> np.ndarray:
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        return numeric if self.direction == "min" else -numeric


@dataclass(frozen=True)
class ConstraintSpec:
    """A per-candidate numeric feasibility constraint."""

    column: str
    operator: str
    threshold: float

    def satisfied(self, values: pd.Series) -> np.ndarray:
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        operation = {
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
            "==": operator.eq,
        }[self.operator]
        return np.isfinite(numeric) & operation(numeric, self.threshold)


def parse_objective(specification: str) -> ObjectiveSpec:
    """Parse ``COLUMN:min`` or ``COLUMN:max``."""

    try:
        column, direction = (part.strip() for part in specification.rsplit(":", 1))
    except ValueError as error:
        raise ValueError(f"Invalid objective {specification!r}; expected COLUMN:min|max") from error
    if not column or direction not in {"min", "max"}:
        raise ValueError(f"Invalid objective {specification!r}; expected COLUMN:min|max")
    return ObjectiveSpec(column=column, direction=direction)  # type: ignore[arg-type]


_CONSTRAINT_PATTERN = re.compile(
    r"^\s*(?P<column>[A-Za-z_][A-Za-z0-9_.-]*)\s*"
    r"(?P<operator><=|>=|==|<|>)\s*"
    r"(?P<threshold>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


def parse_constraint(specification: str) -> ConstraintSpec:
    """Parse a numeric constraint such as ``toxicity<=0.2``."""

    match = _CONSTRAINT_PATTERN.fullmatch(specification)
    if match is None:
        raise ValueError(
            f"Invalid constraint {specification!r}; expected COLUMN<op>NUMBER"
        )
    return ConstraintSpec(
        column=match.group("column"),
        operator=match.group("operator"),
        threshold=float(match.group("threshold")),
    )
