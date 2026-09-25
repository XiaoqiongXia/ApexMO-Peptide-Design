"""Validated wrappers for the frozen HemoPI2 and ToxinPred3 predictors."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_fasta(frame: pd.DataFrame, path: Path, id_column: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in frame[[id_column, "sequence"]].itertuples(index=False):
            handle.write(f">{row[0]}\n{row.sequence}\n")


def score_hemopi2(
    candidates: pd.DataFrame,
    work: Path,
    *,
    id_column: str,
    root: Path,
    wrapper: Path,
    threshold: float,
    quiet: bool = True,
) -> pd.DataFrame:
    """Run the frozen HemoPI2 Hybrid2 classifier and validate its output."""

    runtime = work / "hemopi2_runtime"
    runtime.mkdir()
    (runtime / "model").symlink_to(root / "Model", target_is_directory=True)
    (runtime / "merci").symlink_to(root / "merci", target_is_directory=True)
    (runtime / "motif").symlink_to(root / "motif", target_is_directory=True)
    fasta = work / "hemopi2_input.fasta"
    _write_fasta(candidates, fasta, id_column)
    raw = work / "hemopi2_raw.csv"
    command = [
        sys.executable,
        str(wrapper),
        "--official-script",
        str(root / "hemopi2_classification.py"),
        "--runtime-dir",
        str(runtime),
        "-i",
        str(fasta),
        "-o",
        raw.name,
        "-j",
        "1",
        "-m",
        "4",
        "-t",
        str(threshold),
        "-wd",
        str(work),
        "-d",
        "2",
    ]
    subprocess.run(
        command,
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
    )
    result = pd.read_csv(raw).rename(
        columns={
            "SeqID": id_column,
            "Sequence": "hemopi2_sequence",
            "ESM Score": "hemopi2_esm_score",
            "MERCI Score": "hemopi2_merci_score",
            "Hybrid Score": "hemopi2_score",
            "Prediction": "hemopi2_prediction",
        }
    )
    result[id_column] = result[id_column].astype(str).str.lstrip(">")
    columns = [
        id_column,
        "hemopi2_sequence",
        "hemopi2_esm_score",
        "hemopi2_merci_score",
        "hemopi2_score",
        "hemopi2_prediction",
    ]
    if missing := set(columns).difference(result.columns):
        raise ValueError(f"HemoPI2 output lacks columns: {sorted(missing)}")
    result = result[columns]
    if len(result) != len(candidates) or not result[id_column].is_unique:
        raise RuntimeError("HemoPI2 output cardinality or identifier validation failed")
    expected = candidates.set_index(id_column)["sequence"]
    observed = result.set_index(id_column)["hemopi2_sequence"]
    if not observed.reindex(expected.index).eq(expected).all():
        raise RuntimeError("HemoPI2 output sequence validation failed")
    return result


def score_toxinpred3(
    candidates: pd.DataFrame,
    work: Path,
    *,
    id_column: str,
    root: Path,
    runner: Path,
    python: Path,
    threshold: float,
) -> pd.DataFrame:
    """Run the frozen ToxinPred3 hybrid classifier and validate its output."""

    runtime = work / "toxinpred3_runtime"
    runtime.mkdir()
    (runtime / "model").symlink_to(root / "model", target_is_directory=True)
    fasta = work / "toxinpred3_input.fasta"
    _write_fasta(candidates, fasta, id_column)
    raw = work / "toxinpred3_raw.csv"
    command = [
        sys.executable,
        str(runner),
        "--input",
        str(fasta),
        "--output",
        str(raw),
        "--runtime-dir",
        str(runtime),
        "--script",
        str(root / "toxinpred3.py"),
        "--python",
        str(python),
        "--threshold",
        str(threshold),
        "--chunks-dir",
        str(work / "toxinpred3_chunks"),
    ]
    subprocess.run(command, check=True)
    result = pd.read_csv(raw).rename(
        columns={
            "ID": id_column,
            "Subject": id_column,
            "ML Score": "toxinpred3_ml_score",
            "MERCI Score Pos": "toxinpred3_merci_score_pos",
            "MERCI Score Neg": "toxinpred3_merci_score_neg",
            "Hybrid Score": "toxinpred3_score",
            "Prediction": "toxinpred3_prediction",
            "PPV": "toxinpred3_ppv",
        }
    )
    columns = [
        id_column,
        "toxinpred3_ml_score",
        "toxinpred3_merci_score_pos",
        "toxinpred3_merci_score_neg",
        "toxinpred3_score",
        "toxinpred3_prediction",
        "toxinpred3_ppv",
    ]
    if missing := set(columns).difference(result.columns):
        raise ValueError(f"ToxinPred3 output lacks columns: {sorted(missing)}")
    result = result[columns]
    if len(result) != len(candidates) or not result[id_column].is_unique:
        raise RuntimeError("ToxinPred3 output cardinality or identifier validation failed")
    return result
