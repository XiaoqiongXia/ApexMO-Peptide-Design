from pathlib import Path

import pandas as pd
import pytest

from amp_design.production_sampling import (
    build_tasks,
    derive_generation_seed,
    validate_completed_shard,
)


def test_generation_seeds_are_deterministic_and_distinct() -> None:
    values = {
        derive_generation_seed(seed, length, shard)
        for seed in (42, 123, 2025)
        for length in range(10, 31)
        for shard in range(10)
    }
    assert len(values) == 3 * 21 * 10
    assert derive_generation_seed(42, 10, 0) == 73_421_000
    assert derive_generation_seed(42, 10, 100) == derive_generation_seed(42, 10, 100)
    assert derive_generation_seed(42, 10, 100) != derive_generation_seed(42, 10, 101)


def test_build_tasks_uses_checkpoint_length_shards(tmp_path: Path) -> None:
    tasks = build_tasks(
        tmp_path,
        training_seed=42,
        lengths=[11, 10],
        samples_per_length=2000,
        shard_size=1000,
    )
    assert len(tasks) == 4
    assert tasks[0].output == tmp_path / "raw/seed_42/length_10/shard_000.csv"
    with pytest.raises(ValueError, match="divisible"):
        build_tasks(
            tmp_path,
            training_seed=42,
            lengths=[10],
            samples_per_length=1500,
            shard_size=1000,
        )


def test_build_tasks_supports_more_than_one_hundred_shards(tmp_path: Path) -> None:
    tasks = build_tasks(
        tmp_path,
        training_seed=42,
        lengths=[10],
        samples_per_length=101,
        shard_size=1,
    )
    assert len(tasks) == 101
    assert tasks[-1].shard_id == 100


def test_validate_completed_shard_checks_provenance(tmp_path: Path) -> None:
    task = build_tasks(
        tmp_path,
        training_seed=42,
        lengths=[10],
        samples_per_length=2,
        shard_size=2,
    )[0]
    task.output.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "sequence": ["A" * 10, "C" * 10],
            "length": [10, 10],
            "training_seed": [42, 42],
            "generation_seed": [task.generation_seed] * 2,
            "shard_id": [0, 0],
            "checkpoint_sha256": ["hash", "hash"],
            "sampling_steps": [256, 256],
            "use_ema": [True, True],
        }
    ).to_csv(task.output, index=False)
    assert validate_completed_shard(task.output, task, "hash")
    assert not validate_completed_shard(task.output, task, "different")
