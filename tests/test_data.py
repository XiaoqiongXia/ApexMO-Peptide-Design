from pathlib import Path

import pandas as pd
import pytest

from amp_design.data import (
    assign_cluster_splits,
    audit_clusters,
    load_and_validate_amp_csv,
    mmseqs_cluster_command,
    write_cluster_splits,
)


def _frame() -> pd.DataFrame:
    rows = []
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    for index in range(30):
        sequence = amino_acids[index % 20] + amino_acids[index // 20] + "ACD"
        rows.append(
            {
                "sequence_id": f"seq_{index:03d}",
                "sequence": sequence,
                "length": len(sequence),
                "sources": "source_a" if index % 2 == 0 else "source_b",
                "is_strict_positive_prior": True,
                "cluster_id": f"cluster_{index // 3:03d}",
            }
        )
    return pd.DataFrame(rows)


def test_load_and_validate_amp_csv(tmp_path: Path) -> None:
    path = tmp_path / "amp.csv"
    _frame().drop(columns="cluster_id").to_csv(path, index=False)
    frame, summary = load_and_validate_amp_csv(path)
    assert len(frame) == 30
    assert summary.n_unique_sequences == 30
    assert summary.min_length == 5
    assert summary.alphabet == "ACDEFGHIKLMNPQRSTVWY"


def test_validation_rejects_noncanonical_sequence(tmp_path: Path) -> None:
    frame = _frame().drop(columns="cluster_id")
    frame.loc[0, "sequence"] = "ABCDX"
    path = tmp_path / "amp.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Non-canonical"):
        load_and_validate_amp_csv(path)


def test_cluster_assignment_is_deterministic_and_leak_free(tmp_path: Path) -> None:
    frame = _frame()
    first = assign_cluster_splits(frame, seed=20260717)
    second = assign_cluster_splits(frame, seed=20260717)
    assert first == second
    assert set(first.values()) == {"train", "validation", "test"}

    paths = write_cluster_splits(frame, first, tmp_path)
    observed = pd.concat(pd.read_csv(path) for path in paths.values())
    assert observed.groupby("cluster_id")["split"].nunique().max() == 1
    assert len(observed) == len(frame)


def test_cluster_audit() -> None:
    audit = audit_clusters(_frame())
    assert audit["n_sequences"] == 30
    assert audit["n_clusters"] == 10
    assert audit["largest_cluster_size"] == 3
    assert audit["largest_cluster_fraction"] == pytest.approx(0.1)


def test_reviewed_mmseqs_command_contains_short_peptide_options() -> None:
    command = mmseqs_cluster_command(
        ["mmseqs"],
        Path("sequences.fasta"),
        Path("clusters"),
        Path("tmp"),
        threads=64,
    )
    joined = " ".join(command)
    for expected in (
        "--cluster-mode 1",
        "--min-seq-id 0.67",
        "--mask 0",
        "--comp-bias-corr 0",
        "--min-ungapped-score 0",
        "--max-seqs 20000",
        "-k 5",
        "--spaced-kmer-mode 0",
        "-e 1e6",
        "-s 7.5",
        "--threads 64",
    ):
        assert expected in joined
