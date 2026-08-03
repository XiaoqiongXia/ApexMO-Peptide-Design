import pandas as pd

from amp_design.benchmark_summary import unique_sequences_within_cell


def test_seed_length_summary_deduplicates_within_each_cell() -> None:
    frame = pd.DataFrame(
        {
            "model": ["model", "model", "model"],
            "sampling_seed": [42, 42, 123],
            "requested_length": [10, 10, 10],
            "sequence_canonical": ["A" * 10, "A" * 10, "A" * 10],
        }
    )
    counts = []
    for _, group in frame.groupby(
        ["model", "sampling_seed", "requested_length"], sort=True
    ):
        counts.append(len(unique_sequences_within_cell(group)))
    assert counts == [1, 1]
