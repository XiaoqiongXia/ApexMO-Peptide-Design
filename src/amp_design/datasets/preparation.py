"""Sequence validation, MMseqs2 clustering, and cluster-based dataset splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
REQUIRED_COLUMNS = frozenset(
    {"sequence_id", "sequence", "length", "sources", "is_strict_positive_prior"}
)
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


@dataclass(frozen=True)
class ValidationSummary:
    n_rows: int
    n_unique_ids: int
    n_unique_sequences: int
    min_length: int
    max_length: int
    alphabet: str
    input_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def load_and_validate_amp_csv(
    path: Path,
    *,
    min_length: int = 5,
    max_length: int = 64,
    require_positive_prior: bool = True,
) -> tuple[pd.DataFrame, ValidationSummary]:
    """Load the AMP CSV and check its columns and sequences."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["sequence_id"] = frame["sequence_id"].astype(str).str.strip()
    frame["sequence"] = frame["sequence"].astype(str).str.strip().str.upper()
    frame["sources"] = frame["sources"].fillna("").astype(str).str.strip()

    if frame["sequence_id"].eq("").any():
        raise ValueError("Empty sequence_id values are not allowed")
    if frame["sequence"].eq("").any():
        raise ValueError("Empty sequences are not allowed")
    if frame["sequence_id"].duplicated().any():
        duplicates = frame.loc[frame["sequence_id"].duplicated(), "sequence_id"].tolist()
        raise ValueError(f"Duplicate sequence_id values: {duplicates[:5]}")
    if frame["sequence"].duplicated().any():
        duplicates = frame.loc[frame["sequence"].duplicated(), "sequence"].tolist()
        raise ValueError(f"Duplicate sequences: {duplicates[:5]}")

    computed_lengths = frame["sequence"].str.len().astype(int)
    declared_lengths = pd.to_numeric(frame["length"], errors="raise").astype(int)
    mismatch = declared_lengths.ne(computed_lengths)
    if mismatch.any():
        examples = frame.loc[mismatch, ["sequence_id", "sequence", "length"]].head()
        raise ValueError(f"Declared length mismatch:\n{examples.to_string(index=False)}")
    frame["length"] = computed_lengths

    invalid_lengths = ~frame["length"].between(min_length, max_length)
    if invalid_lengths.any():
        examples = frame.loc[invalid_lengths, ["sequence_id", "length"]].head()
        raise ValueError(f"Sequences outside [{min_length}, {max_length}]:\n{examples}")

    invalid_sequences = frame["sequence"].map(
        lambda sequence: sorted(set(sequence).difference(CANONICAL_AMINO_ACIDS))
    )
    invalid_mask = invalid_sequences.map(bool)
    if invalid_mask.any():
        examples = [
            (frame.loc[index, "sequence_id"], invalid_sequences.loc[index])
            for index in frame.index[invalid_mask][:5]
        ]
        raise ValueError(f"Non-canonical amino acids found: {examples}")

    positive_prior = frame["is_strict_positive_prior"].map(_parse_bool)
    frame["is_strict_positive_prior"] = positive_prior
    if require_positive_prior and not positive_prior.all():
        count = int((~positive_prior).sum())
        raise ValueError(f"Expected a positive-prior-only dataset; found {count} non-positive rows")

    alphabet = "".join(sorted(set("".join(frame["sequence"]))))
    summary = ValidationSummary(
        n_rows=len(frame),
        n_unique_ids=frame["sequence_id"].nunique(),
        n_unique_sequences=frame["sequence"].nunique(),
        min_length=int(frame["length"].min()),
        max_length=int(frame["length"].max()),
        alphabet=alphabet,
        input_sha256=sha256_file(path),
    )
    return frame, summary


def write_fasta(frame: pd.DataFrame, path: Path) -> None:
    """Write sequence identifiers and sequences as a two-line FASTA."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sequence_id, sequence in frame[["sequence_id", "sequence"]].itertuples(
            index=False, name=None
        ):
            handle.write(f">{sequence_id}\n{sequence}\n")


def _run_checked(command: Sequence[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.strip()


def mmseqs_cluster_command(
    mmseqs_command: Sequence[str],
    fasta_path: Path,
    output_prefix: Path,
    tmp_dir: Path,
    *,
    threads: int,
    min_sequence_identity: float = 0.67,
) -> list[str]:
    """Build the MMseqs2 clustering command for short peptides."""

    return [
        *mmseqs_command,
        "easy-cluster",
        str(fasta_path),
        str(output_prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(min_sequence_identity),
        "-c",
        "0.8",
        "--cov-mode",
        "0",
        "--cluster-mode",
        "1",
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
        "--remove-tmp-files",
        "1",
    ]


def run_mmseqs_clustering(
    fasta_path: Path,
    output_prefix: Path,
    tmp_dir: Path,
    *,
    mmseqs_command: Sequence[str],
    threads: int,
    min_sequence_identity: float,
) -> tuple[Path, list[str], str]:
    """Run MMseqs2 and return the cluster TSV, command, and version."""

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cluster_tsv = Path(f"{output_prefix}_cluster.tsv")
    if cluster_tsv.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing clustering result: {cluster_tsv}. "
            "Move it aside or choose a new output directory."
        )

    version = _run_checked([*mmseqs_command, "version"])
    command = mmseqs_cluster_command(
        mmseqs_command,
        fasta_path,
        output_prefix,
        tmp_dir,
        threads=threads,
        min_sequence_identity=min_sequence_identity,
    )
    _run_checked(command)
    if not cluster_tsv.is_file():
        raise RuntimeError(f"MMseqs2 did not create {cluster_tsv}")
    return cluster_tsv, command, version


def read_cluster_map(cluster_tsv: Path, expected_ids: Iterable[str]) -> pd.DataFrame:
    """Read and validate the MMseqs representative/member mapping."""

    mapping = pd.read_csv(
        cluster_tsv,
        sep="\t",
        names=["cluster_id", "sequence_id"],
        dtype=str,
    )
    if mapping.empty:
        raise ValueError("MMseqs2 cluster map is empty")
    if mapping["sequence_id"].duplicated().any():
        duplicates = mapping.loc[mapping["sequence_id"].duplicated(), "sequence_id"].tolist()
        raise ValueError(f"MMseqs2 assigned sequences more than once: {duplicates[:5]}")

    expected = set(expected_ids)
    observed = set(mapping["sequence_id"])
    missing = expected.difference(observed)
    unexpected = observed.difference(expected)
    if missing or unexpected:
        raise ValueError(
            f"Cluster map coverage mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected"
        )
    return mapping.sort_values(["cluster_id", "sequence_id"]).reset_index(drop=True)


def audit_clusters(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize cluster sizes and sequence lengths."""

    sizes = frame.groupby("cluster_id", sort=True).size().sort_values()
    singleton_clusters = int((sizes == 1).sum())
    largest_size = int(sizes.iloc[-1])
    percentiles = {
        str(percentile): float(np.percentile(sizes.to_numpy(), percentile))
        for percentile in (0, 25, 50, 75, 90, 95, 99, 100)
    }
    length_counts = frame.groupby("length").size()
    length_cluster_counts = frame.groupby("length")["cluster_id"].nunique()
    return {
        "n_sequences": int(len(frame)),
        "n_clusters": int(len(sizes)),
        "singleton_clusters": singleton_clusters,
        "singleton_cluster_fraction": singleton_clusters / len(sizes),
        "sequences_in_singleton_clusters_fraction": singleton_clusters / len(frame),
        "largest_cluster_size": largest_size,
        "largest_cluster_fraction": largest_size / len(frame),
        "cluster_size_percentiles": percentiles,
        "length_counts": {str(int(key)): int(value) for key, value in length_counts.items()},
        "length_cluster_counts": {
            str(int(key)): int(value) for key, value in length_cluster_counts.items()
        },
    }


def _cluster_features(frame: pd.DataFrame) -> tuple[list[int], list[str]]:
    return sorted(frame["length"].unique().tolist()), sorted(frame["sources"].unique().tolist())


def assign_cluster_splits(
    frame: pd.DataFrame,
    *,
    seed: int,
    ratios: dict[str, float] | None = None,
) -> dict[str, str]:
    """Greedily assign complete clusters while balancing size, length, and source."""

    ratios = ratios or SPLIT_RATIOS
    if set(ratios) != {"train", "validation", "test"}:
        raise ValueError("ratios must define train, validation, and test")
    if not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError("split ratios must sum to one")
    if frame["cluster_id"].isna().any():
        raise ValueError("cluster_id cannot be missing")

    lengths, sources = _cluster_features(frame)
    total_size = len(frame)
    global_length = frame["length"].value_counts().reindex(lengths, fill_value=0).to_numpy(float)
    global_source = frame["sources"].value_counts().reindex(sources, fill_value=0).to_numpy(float)

    groups: list[dict[str, Any]] = []
    for cluster_id, group in frame.groupby("cluster_id", sort=True):
        groups.append(
            {
                "cluster_id": str(cluster_id),
                "size": len(group),
                "length": group["length"]
                .value_counts()
                .reindex(lengths, fill_value=0)
                .to_numpy(float),
                "source": group["sources"]
                .value_counts()
                .reindex(sources, fill_value=0)
                .to_numpy(float),
            }
        )

    rng = np.random.default_rng(seed)
    tie_breakers = {group["cluster_id"]: float(rng.random()) for group in groups}
    groups.sort(key=lambda group: (-group["size"], tie_breakers[group["cluster_id"]]))

    split_names = list(ratios)
    target_size = {name: total_size * ratios[name] for name in split_names}
    target_length = {name: global_length * ratios[name] for name in split_names}
    target_source = {name: global_source * ratios[name] for name in split_names}
    state = {
        name: {
            "size": 0.0,
            "length": np.zeros(len(lengths), dtype=float),
            "source": np.zeros(len(sources), dtype=float),
        }
        for name in split_names
    }

    def candidate_score(candidate: str, group: dict[str, Any]) -> float:
        score_size = 0.0
        score_length = 0.0
        score_source = 0.0
        overflow_penalty = 0.0
        for name in split_names:
            add = group if name == candidate else None
            size = state[name]["size"] + (add["size"] if add else 0.0)
            length = state[name]["length"] + (add["length"] if add else 0.0)
            source = state[name]["source"] + (add["source"] if add else 0.0)
            score_size += ((size - target_size[name]) / max(target_size[name], 1.0)) ** 2
            score_length += float(
                np.mean((length - target_length[name]) ** 2 / (target_length[name] + 5.0))
            )
            score_source += float(
                np.mean((source - target_source[name]) ** 2 / (target_source[name] + 5.0))
            )
            overflow = max(0.0, size - target_size[name] * 1.05)
            overflow_penalty += (overflow / max(target_size[name], 1.0)) ** 2
        return score_size + 0.05 * score_length + 0.05 * score_source + 100.0 * overflow_penalty

    assignments: dict[str, str] = {}
    for group in groups:
        scores = {name: candidate_score(name, group) for name in split_names}
        best = min(
            split_names,
            key=lambda name: (
                scores[name],
                -(target_size[name] - state[name]["size"]),
                name,
            ),
        )
        assignments[group["cluster_id"]] = best
        state[best]["size"] += group["size"]
        state[best]["length"] += group["length"]
        state[best]["source"] += group["source"]

    return assignments


def _split_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for split, group in frame.groupby("split", sort=True):
        stats[str(split)] = {
            "n_sequences": int(len(group)),
            "n_clusters": int(group["cluster_id"].nunique()),
            "length_counts": {
                str(int(key)): int(value)
                for key, value in group["length"].value_counts().sort_index().items()
            },
            "source_counts": {
                str(key): int(value)
                for key, value in group["sources"].value_counts().sort_index().items()
            },
        }
    return stats


def write_cluster_splits(
    frame: pd.DataFrame,
    assignments: dict[str, str],
    output_dir: Path,
) -> dict[str, Path]:
    """Write split CSVs and verify that no cluster crosses a split."""

    output_dir.mkdir(parents=True, exist_ok=True)
    result = frame.copy()
    result["split"] = result["cluster_id"].map(assignments)
    if result["split"].isna().any():
        raise ValueError("At least one cluster has no split assignment")
    leakage = result.groupby("cluster_id")["split"].nunique()
    if (leakage != 1).any():
        raise RuntimeError("Cluster leakage detected")

    paths: dict[str, Path] = {}
    for split in SPLIT_RATIOS:
        split_path = output_dir / f"{split}.csv"
        result.loc[result["split"] == split].sort_values("sequence_id").to_csv(
            split_path, index=False
        )
        paths[split] = split_path
    return paths


def write_random_split(frame: pd.DataFrame, output_path: Path, *, seed: int) -> None:
    """Write an auxiliary random split assignment for comparison only."""

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(frame))
    n_train = round(len(frame) * SPLIT_RATIOS["train"])
    n_validation = round(len(frame) * SPLIT_RATIOS["validation"])
    labels = np.empty(len(frame), dtype=object)
    labels[indices[:n_train]] = "train"
    labels[indices[n_train : n_train + n_validation]] = "validation"
    labels[indices[n_train + n_validation :]] = "test"
    pd.DataFrame({"sequence_id": frame["sequence_id"], "split": labels}).sort_values(
        "sequence_id"
    ).to_csv(output_path, index=False)


def _write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def prepare_data(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, validation = load_and_validate_amp_csv(input_path)
    fasta_path = output_dir / "sequences.fasta"
    write_fasta(frame, fasta_path)

    mmseqs_dir = output_dir / "mmseqs"
    prefix = mmseqs_dir / "clusters"
    tmp_dir = mmseqs_dir / "tmp"
    mmseqs_base = shlex.split(args.mmseqs_command)
    cluster_tsv, command, version = run_mmseqs_clustering(
        fasta_path,
        prefix,
        tmp_dir,
        mmseqs_command=mmseqs_base,
        threads=args.threads,
        min_sequence_identity=args.min_sequence_identity,
    )

    cluster_map = read_cluster_map(cluster_tsv, frame["sequence_id"])
    cluster_map_path = output_dir / "cluster_map.csv"
    cluster_map.to_csv(cluster_map_path, index=False)
    clustered = frame.merge(cluster_map, on="sequence_id", how="left", validate="one_to_one")
    audit = audit_clusters(clustered)
    audit_path = output_dir / "cluster_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if audit["largest_cluster_fraction"] > args.max_cluster_fraction:
        raise RuntimeError(
            "Cluster audit gate failed: largest cluster contains "
            f"{audit['largest_cluster_fraction']:.2%} of sequences, above "
            f"the {args.max_cluster_fraction:.2%} limit. Audit {audit_path} before continuing."
        )

    assignments = assign_cluster_splits(clustered, seed=args.split_seed)
    split_paths = write_cluster_splits(clustered, assignments, output_dir)
    random_split_path = output_dir / "random_split_assignments.csv"
    write_random_split(frame, random_split_path, seed=args.split_seed)

    assigned = clustered.copy()
    assigned["split"] = assigned["cluster_id"].map(assignments)
    artifacts = {
        "fasta": fasta_path,
        "cluster_map": cluster_map_path,
        "cluster_audit": audit_path,
        "random_split_assignments": random_split_path,
        **split_paths,
    }
    artifact_hashes = {name: sha256_file(path) for name, path in artifacts.items()}
    command_string = shlex.join(command)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(input_path),
            "sha256": validation.input_sha256,
            "validation": validation.as_dict(),
        },
        "mmseqs": {
            "version": version,
            "command": command_string,
            "threads": args.threads,
        },
        "clustering": {
            "min_sequence_identity": args.min_sequence_identity,
            "coverage": 0.8,
            "coverage_mode": 0,
            "cluster_mode": 1,
            "audit": audit,
        },
        "splitting": {
            "seed": args.split_seed,
            "ratios": SPLIT_RATIOS,
            "statistics": _split_statistics(assigned),
        },
        "artifacts": {
            name: {"path": str(path), "sha256": artifact_hashes[name]}
            for name, path in artifacts.items()
        },
    }
    local_manifest = output_dir / "split_manifest.json"
    local_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    safe_manifest = {
        "schema_version": manifest["schema_version"],
        "input_sha256": validation.input_sha256,
        "mmseqs": manifest["mmseqs"],
        "clustering": manifest["clustering"],
        "splitting": manifest["splitting"],
        "artifact_sha256": artifact_hashes,
    }
    _write_yaml(safe_manifest, args.manifest.resolve())
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Validate, cluster, audit, and split AMP data")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--mmseqs-command", default="conda run -n bg mmseqs")
    prepare.add_argument("--threads", type=int, default=64)
    prepare.add_argument("--min-sequence-identity", type=float, default=0.67)
    prepare.add_argument("--split-seed", type=int, default=20260717)
    prepare.add_argument("--max-cluster-fraction", type=float, default=0.10)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_data(args)
        audit = manifest["clustering"]["audit"]
        stats = manifest["splitting"]["statistics"]
        print(
            json.dumps(
                {
                    "status": "ok",
                    "n_sequences": audit["n_sequences"],
                    "n_clusters": audit["n_clusters"],
                    "largest_cluster_fraction": audit["largest_cluster_fraction"],
                    "split_sizes": {
                        split: values["n_sequences"] for split, values in stats.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
