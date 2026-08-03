"""Pure helpers for benchmark endpoint summarization."""

from __future__ import annotations

import pandas as pd


def unique_sequences_within_cell(group: pd.DataFrame) -> pd.DataFrame:
    """De-duplicate sequences inside one model/seed/length benchmark cell."""

    required = {
        "model",
        "sampling_seed",
        "requested_length",
        "sequence_canonical",
    }
    if missing := required.difference(group.columns):
        raise ValueError(f"Benchmark cell lacks columns: {sorted(missing)}")
    for column in ("model", "sampling_seed", "requested_length"):
        if group[column].nunique(dropna=False) != 1:
            raise ValueError(f"Benchmark cell contains multiple {column} values")
    return group.drop_duplicates("sequence_canonical", keep="first")
