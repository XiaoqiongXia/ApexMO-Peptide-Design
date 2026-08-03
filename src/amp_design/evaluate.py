"""Generator-only validity, diversity, memorization, and composition metrics."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .tokenizer import AMINO_ACIDS

HYDROPHOBIC_AMINO_ACIDS = frozenset("AVILMFWY")


def approximate_net_charge(sequence: str) -> float:
    """Approximate side-chain charge near neutral pH for an unmodified peptide."""

    counts = Counter(sequence)
    return float(counts["K"] + counts["R"] + 0.1 * counts["H"] - counts["D"] - counts["E"])


def hydrophobic_fraction(sequence: str) -> float:
    return sum(amino_acid in HYDROPHOBIC_AMINO_ACIDS for amino_acid in sequence) / len(sequence)


def _write_fasta(frame: pd.DataFrame, path: Path, id_column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for identifier, sequence in frame[[id_column, "sequence"]].itertuples(
            index=False, name=None
        ):
            handle.write(f">{identifier}\n{sequence}\n")


def novelty_search_command(
    mmseqs_command: Sequence[str],
    query_fasta: Path,
    train_fasta: Path,
    result_path: Path,
    tmp_dir: Path,
    *,
    threads: int,
) -> list[str]:
    return [
        *mmseqs_command,
        "easy-search",
        str(query_fasta),
        str(train_fasta),
        str(result_path),
        str(tmp_dir),
        "--min-seq-id",
        "0.5",
        "-c",
        "0.8",
        "--cov-mode",
        "0",
        "--mask",
        "0",
        "--comp-bias-corr",
        "0",
        "--min-ungapped-score",
        "0",
        "--max-seqs",
        "20000",
        "-k",
        "5",
        "--spaced-kmer-mode",
        "0",
        "-e",
        "1e6",
        "-s",
        "7.5",
        "--threads",
        str(threads),
        "--format-output",
        "query,target,fident,qcov,tcov,alnlen",
        "--remove-tmp-files",
        "1",
    ]


def _normalize_identity(value: float) -> float:
    return value / 100.0 if value > 1.0 else value


def parse_novelty_hits(result_path: Path) -> pd.DataFrame:
    columns = ["candidate_id", "target_id", "identity", "qcov", "tcov", "alnlen"]
    if not result_path.exists() or result_path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    hits = pd.read_csv(result_path, sep="\t", names=columns)
    for column in ("identity", "qcov", "tcov"):
        hits[column] = pd.to_numeric(hits[column], errors="raise").map(_normalize_identity)
    hits["alnlen"] = pd.to_numeric(hits["alnlen"], errors="raise").astype(int)
    return hits


def annotate_mmseqs_novelty(
    candidates: pd.DataFrame,
    train: pd.DataFrame,
    work_dir: Path,
    *,
    mmseqs_command: Sequence[str],
    threads: int,
) -> pd.DataFrame:
    """Annotate candidate matches at >=50% identity and >=80% bidirectional coverage."""

    work_dir.mkdir(parents=True, exist_ok=True)
    result_path = work_dir / "novelty_hits.tsv"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path}")
    candidate_frame = candidates.copy().reset_index(drop=True)
    if "candidate_id" in candidate_frame:
        candidate_frame["candidate_id"] = candidate_frame["candidate_id"].astype(str)
        if candidate_frame["candidate_id"].duplicated().any():
            raise ValueError("candidate_id values must be unique for novelty annotation")
    else:
        candidate_frame["candidate_id"] = [
            f"candidate_{index:08d}" for index in range(len(candidate_frame))
        ]
    train_frame = train.copy()
    if "sequence_id" not in train_frame:
        train_frame["sequence_id"] = [f"train_{index:08d}" for index in range(len(train_frame))]
    query_fasta = work_dir / "candidates.fasta"
    train_fasta = work_dir / "train.fasta"
    _write_fasta(candidate_frame, query_fasta, "candidate_id")
    _write_fasta(train_frame, train_fasta, "sequence_id")

    command = novelty_search_command(
        mmseqs_command,
        query_fasta,
        train_fasta,
        result_path,
        work_dir / "tmp",
        threads=threads,
    )
    subprocess.run(command, check=True)
    hits = parse_novelty_hits(result_path)
    best = (
        hits.sort_values(
            ["candidate_id", "identity", "qcov", "tcov"],
            ascending=[True, False, False, False],
        )
        .drop_duplicates("candidate_id")
        .set_index("candidate_id")
    )
    target_sequences = train_frame.set_index("sequence_id")["sequence"].to_dict()
    candidate_frame["has_ge_50_identity_match"] = candidate_frame["candidate_id"].isin(best.index)
    candidate_frame["max_train_identity"] = candidate_frame["candidate_id"].map(
        best["identity"] if not best.empty else {}
    )
    candidate_frame["nearest_train_sequence_id"] = candidate_frame["candidate_id"].map(
        best["target_id"] if not best.empty else {}
    )
    candidate_frame["nearest_train_sequence"] = candidate_frame["nearest_train_sequence_id"].map(
        target_sequences
    )
    candidate_frame["identity_search_upper_bound_if_missing"] = 0.5
    metadata = {
        "mmseqs_version": subprocess.run(
            [*mmseqs_command, "version"], check=True, text=True, capture_output=True
        ).stdout.strip(),
        "command": shlex.join(command),
        "n_candidates": len(candidate_frame),
        "n_candidates_with_ge_50_match": int(candidate_frame["has_ge_50_identity_match"].sum()),
    }
    (work_dir / "novelty_search.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return candidate_frame


def evaluate_candidates(candidates: pd.DataFrame, train_sequences: set[str]) -> dict[str, Any]:
    required = {"sequence", "length"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Missing candidate columns: {sorted(missing)}")
    frame = candidates.copy()
    frame["sequence"] = frame["sequence"].astype(str).str.strip().str.upper()
    frame["length"] = pd.to_numeric(frame["length"], errors="raise").astype(int)
    valid_alphabet = frame["sequence"].map(
        lambda sequence: bool(sequence) and set(sequence).issubset(set(AMINO_ACIDS))
    )
    valid_length = frame.apply(lambda row: len(row["sequence"]) == row["length"], axis=1)
    exact_train_match = frame["sequence"].isin(train_sequences)
    frame["net_charge"] = frame["sequence"].map(approximate_net_charge)
    frame["hydrophobic_fraction"] = frame["sequence"].map(hydrophobic_fraction)

    by_length = {}
    for length, group in frame.groupby("length", sort=True):
        by_length[str(int(length))] = {
            "n": int(len(group)),
            "unique_fraction": float(group["sequence"].nunique() / len(group)),
            "mean_net_charge": float(group["net_charge"].mean()),
            "mean_hydrophobic_fraction": float(group["hydrophobic_fraction"].mean()),
        }
    metrics = {
        "n_candidates": int(len(frame)),
        "valid_amino_acid_fraction": float(valid_alphabet.mean()),
        "length_condition_fraction": float(valid_length.mean()),
        "unique_fraction": float(frame["sequence"].nunique() / len(frame)),
        "exact_train_match_fraction": float(exact_train_match.mean()),
        "mean_net_charge": float(frame["net_charge"].mean()),
        "mean_hydrophobic_fraction": float(frame["hydrophobic_fraction"].mean()),
        "amino_acid_frequencies": {
            amino_acid: count / sum(Counter("".join(frame["sequence"])).values())
            for amino_acid, count in sorted(Counter("".join(frame["sequence"])).items())
        },
        "by_length": by_length,
    }
    if "has_ge_50_identity_match" in frame:
        has_match = frame["has_ge_50_identity_match"].astype(bool)
        identity = pd.to_numeric(frame["max_train_identity"], errors="coerce")
        metrics["identity_lt_50_fraction"] = float((~has_match).mean())
        metrics["identity_lt_80_fraction"] = float(((~has_match) | identity.lt(0.8)).mean())
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotated-output", type=Path)
    parser.add_argument("--novelty-work-dir", type=Path)
    parser.add_argument("--mmseqs-command", default="conda run -n bg mmseqs")
    parser.add_argument("--threads", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    candidates = pd.read_csv(args.candidates)
    train = pd.read_csv(args.train)
    if args.novelty_work_dir:
        candidates = annotate_mmseqs_novelty(
            candidates,
            train,
            args.novelty_work_dir,
            mmseqs_command=shlex.split(args.mmseqs_command),
            threads=args.threads,
        )
        annotated_output = args.annotated_output or args.candidates.with_name(
            args.candidates.stem + "_annotated.csv"
        )
        annotated_output.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(annotated_output, index=False)
    train_sequences = set(train["sequence"].astype(str))
    metrics = evaluate_candidates(candidates, train_sequences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
