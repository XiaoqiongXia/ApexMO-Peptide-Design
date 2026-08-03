from pathlib import Path

import pandas as pd
import pytest

from amp_design.benchmark import (
    RAW_COLUMNS,
    BenchmarkTask,
    build_tasks,
    derive_seed,
    load_protocol,
    make_raw_frame,
    validate_raw_frame,
    write_immutable_shard,
)

PROTOCOL = Path("configs/generative_model_benchmark_v1.yaml")


def test_frozen_protocol_builds_expected_pilot_and_formal_budgets() -> None:
    protocol = load_protocol(PROTOCOL)
    pilot = build_tasks(protocol, PROTOCOL, phase="pilot")
    formal = build_tasks(protocol, PROTOCOL, phase="formal")
    assert len(pilot) == 3 * 3 * 21
    assert sum(task.count for task in pilot) == 3 * 1260
    assert len(formal) == 3 * 3 * 21 * 5
    assert sum(task.count for task in formal) == 3 * 31500


def test_derived_seeds_are_stable_and_distinct() -> None:
    assert derive_seed(42, 10, 0) == 42_010_000
    values = {
        derive_seed(seed, length, shard)
        for seed in (42, 123, 2025)
        for length in range(10, 31)
        for shard in range(5)
    }
    assert len(values) == 3 * 21 * 5


def test_raw_frame_retains_invalid_attempts() -> None:
    task = BenchmarkTask("flow_matching", "pilot", 42, 42_010_000, 10, 0, 3, Path("x"))
    frame = make_raw_frame(
        ["A" * 10, "AX" + "A" * 8, "A" * 9],
        task=task,
        checkpoint_hash="hash",
        runtime_seconds=3.0,
        device="cpu",
        sampling_parameters={"steps": 2},
    )
    assert tuple(frame.columns) == RAW_COLUMNS
    assert frame["generation_status"].tolist() == ["success", "failed", "failed"]
    assert frame.loc[1, "failure_reason"] == "noncanonical_amino_acid"
    assert frame.loc[2, "failure_reason"] == "length_mismatch"
    assert not validate_raw_frame(frame, task, "hash")


def test_immutable_writer_refuses_different_existing_content(tmp_path: Path) -> None:
    task = BenchmarkTask(
        "flow_matching", "pilot", 42, 42_010_000, 10, 0, 1, tmp_path / "shard.csv"
    )
    frame = make_raw_frame(
        ["A" * 10],
        task=task,
        checkpoint_hash="hash",
        runtime_seconds=1.0,
        device="cpu",
        sampling_parameters={},
    )
    write_immutable_shard(frame, task, "hash")
    write_immutable_shard(frame, task, "hash")
    changed = pd.read_csv(task.output, keep_default_na=False)
    changed.loc[0, "sequence_raw"] = "C" * 10
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_immutable_shard(changed, task, "hash")

