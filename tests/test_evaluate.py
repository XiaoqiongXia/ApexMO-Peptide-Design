from pathlib import Path

import pandas as pd
import pytest

from amp_design.evaluate import (
    annotate_mmseqs_novelty,
    approximate_net_charge,
    evaluate_candidates,
    hydrophobic_fraction,
    novelty_search_command,
    parse_novelty_hits,
)


def test_physicochemical_metrics() -> None:
    assert approximate_net_charge("KKRDE") == 1.0
    assert hydrophobic_fraction("AVKR") == 0.5


def test_candidate_evaluation() -> None:
    frame = pd.DataFrame(
        {
            "sequence": ["ACDEF", "ACDEF", "KKKKK", "ACDEX"],
            "length": [5, 5, 5, 5],
        }
    )
    metrics = evaluate_candidates(frame, {"ACDEF"})
    assert metrics["n_candidates"] == 4
    assert metrics["valid_amino_acid_fraction"] == 0.75
    assert metrics["length_condition_fraction"] == 1.0
    assert metrics["unique_fraction"] == 0.75
    assert metrics["exact_train_match_fraction"] == 0.5


def test_parse_novelty_hits_normalizes_percent_identity(tmp_path: Path) -> None:
    result = tmp_path / "hits.tsv"
    result.write_text("candidate_00000000\ttrain_1\t80.0\t0.9\t0.9\t10\n")
    hits = parse_novelty_hits(result)
    assert hits.loc[0, "identity"] == 0.8
    assert hits.loc[0, "qcov"] == 0.9


def test_novelty_command_matches_split_thresholds() -> None:
    command = novelty_search_command(
        ["mmseqs"],
        Path("query.fasta"),
        Path("train.fasta"),
        Path("hits.tsv"),
        Path("tmp"),
        threads=64,
    )
    joined = " ".join(command)
    for option in (
        "--min-seq-id 0.5",
        "-c 0.8",
        "--cov-mode 0",
        "--mask 0",
        "-k 5",
        "--spaced-kmer-mode 0",
    ):
        assert option in joined


def test_novelty_annotation_requires_unique_candidate_ids(tmp_path: Path) -> None:
    candidates = pd.DataFrame(
        {"candidate_id": ["duplicate", "duplicate"], "sequence": ["AAAAA", "CCCCC"]}
    )
    train = pd.DataFrame({"sequence_id": ["known"], "sequence": ["AAAAA"]})
    with pytest.raises(ValueError, match="candidate_id values must be unique"):
        annotate_mmseqs_novelty(
            candidates,
            train,
            tmp_path,
            mmseqs_command=["mmseqs"],
            threads=1,
        )
