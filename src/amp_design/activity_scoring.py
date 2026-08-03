"""Backend-neutral activity scoring and target aggregation contracts."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .apex import APEX_STRAINS, strain_slug
from .apex_pathogen import APEX_PATHOGEN_STRAINS

COMMAND_RESERVED_COLUMNS = {
    "sequence_sha256",
    "length",
    "activity_scorer",
    "toxicity_score",
    "hemolysis_score",
    "formal_safety_ad_pass",
    "safety_gate_pass",
    "formal_genetic_feasible",
    "pareto_rank",
    "crowding_distance",
    "candidate_origin",
    "genetic_seed",
    "search_protocol",
}


def _target_columns(backend: str, targets: Sequence[str]) -> list[str]:
    if backend == "apex":
        strains = APEX_STRAINS
        prefix = "apex_mic__"
    elif backend == "apex_pathogen":
        strains = APEX_PATHOGEN_STRAINS
        prefix = "apex_pathogen_mic__"
    else:
        raise ValueError(f"Target aggregation is unsupported for activity backend {backend!r}")

    by_name = {strain: f"{prefix}{strain_slug(strain)}" for strain in strains}
    by_slug = {strain_slug(strain): column for strain, column in by_name.items()}
    columns = []
    unknown = []
    for target in targets:
        if target in by_name:
            columns.append(by_name[target])
        elif target in by_slug:
            columns.append(by_slug[target])
        elif target.startswith(prefix):
            columns.append(target)
        else:
            unknown.append(target)
    if unknown:
        available = ", ".join(strains)
        raise ValueError(
            f"Unknown {backend} activity targets: {unknown}. Available targets: {available}"
        )
    return columns


def _target_uncertainty_columns(backend: str, mic_columns: Sequence[str]) -> list[str]:
    """Return per-target log10(MIC) ensemble-SD columns for selected MIC columns."""

    if backend == "apex":
        mic_prefix = "apex_mic__"
        uncertainty_prefix = "apex_log10_mic_sd__"
    elif backend == "apex_pathogen":
        mic_prefix = "apex_pathogen_mic__"
        uncertainty_prefix = "apex_pathogen_log10_mic_sd__"
    else:
        raise ValueError(f"Target aggregation is unsupported for activity backend {backend!r}")
    return [
        f"{uncertainty_prefix}{column.removeprefix(mic_prefix)}" for column in mic_columns
    ]


def aggregate_activity_targets(
    scores: pd.DataFrame,
    *,
    backend: str,
    targets: Sequence[str],
    aggregation: str,
    quantile: float,
    output_column: str,
) -> pd.DataFrame:
    """Aggregate selected target MICs on the log10 scale into one optimizer score."""

    if not targets:
        if output_column not in scores:
            raise ValueError(
                f"Activity scorer {backend!r} did not produce required column {output_column!r}"
            )
        return scores

    columns = _target_columns(backend, targets)
    missing = [column for column in columns if column not in scores]
    if missing:
        raise ValueError(f"Activity scorer output lacks selected target columns: {missing}")
    uncertainty_columns = _target_uncertainty_columns(backend, columns)
    missing_uncertainty = [column for column in uncertainty_columns if column not in scores]
    if missing_uncertainty:
        raise ValueError(
            "Activity scorer output lacks selected-target uncertainty columns: "
            f"{missing_uncertainty}"
        )
    mic = scores[columns].to_numpy(dtype=float)
    if not np.isfinite(mic).all() or np.any(mic <= 0):
        raise ValueError("Selected target MIC values must be finite and positive")
    uncertainty = scores[uncertainty_columns].to_numpy(dtype=float)
    if not np.isfinite(uncertainty).all() or np.any(uncertainty < 0):
        raise ValueError("Selected-target uncertainty values must be finite and nonnegative")
    log_mic = np.log10(mic)
    reducers = {
        "median": lambda values: np.median(values, axis=1),
        "mean": lambda values: np.mean(values, axis=1),
        "max": lambda values: np.max(values, axis=1),
        "min": lambda values: np.min(values, axis=1),
        "quantile": lambda values: np.quantile(values, quantile, axis=1),
    }
    result = scores.copy()
    result[output_column] = reducers[aggregation](log_mic)
    # These backend-neutral aliases are consumed by the optimizer's uncertainty
    # feasibility contract.  When targets are selected, they must describe the
    # same target panel as the activity objective rather than all backend outputs.
    result["apex_median_log10_mic_sd"] = np.median(uncertainty, axis=1)
    result["apex_max_log10_mic_sd"] = np.max(uncertainty, axis=1)
    result["activity_target_count"] = len(targets)
    result["activity_targets"] = json.dumps(list(targets), ensure_ascii=False)
    result["activity_aggregation"] = aggregation
    if aggregation == "quantile":
        result["activity_quantile"] = quantile
    return result


def score_activity_command(
    sequences: Sequence[str],
    command: Sequence[str],
    *,
    output_column: str,
    work_dir: Path,
) -> pd.DataFrame:
    """Run a no-shell CSV scorer command and validate its one-row-per-sequence output.

    ``command`` must contain separate ``{input}`` and ``{output}`` tokens. The input CSV has a
    single ``sequence`` column. The output CSV must contain ``sequence`` and ``output_column``;
    additional prediction, uncertainty, or provenance columns are retained.
    """

    normalized = [str(sequence).strip().upper() for sequence in sequences]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Command activity scorer requires unique input sequences")
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="activity_command_", dir=work_dir) as temporary:
        root = Path(temporary)
        input_path = root / "input.csv"
        output_path = root / "output.csv"
        pd.DataFrame({"sequence": normalized}).to_csv(input_path, index=False)
        resolved = [
            str(input_path)
            if token == "{input}"
            else str(output_path)
            if token == "{output}"
            else token
            for token in command
        ]
        completed = subprocess.run(resolved, capture_output=True, text=True, check=False)
        if completed.returncode:
            detail = (completed.stderr or completed.stdout)[-4000:]
            raise RuntimeError(
                f"Activity scorer command failed with exit code {completed.returncode}: {detail}"
            )
        if not output_path.is_file():
            raise RuntimeError(f"Activity scorer command did not create {output_path}")
        result = pd.read_csv(output_path)

    required = {"sequence", output_column}
    if missing := required.difference(result.columns):
        raise ValueError(f"Activity scorer command output lacks columns: {sorted(missing)}")
    if collisions := COMMAND_RESERVED_COLUMNS.intersection(result.columns):
        raise ValueError(
            f"Activity scorer command output uses reserved columns: {sorted(collisions)}"
        )
    result["sequence"] = result["sequence"].astype(str).str.strip().str.upper()
    if result["sequence"].duplicated().any():
        raise ValueError("Activity scorer command returned duplicate sequences")
    if set(result["sequence"]) != set(normalized) or len(result) != len(normalized):
        raise ValueError("Activity scorer command output does not match its input sequences")
    values = pd.to_numeric(result[output_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Activity scorer command produced non-finite {output_column!r} values")
    return result.set_index("sequence").loc[normalized].reset_index()
