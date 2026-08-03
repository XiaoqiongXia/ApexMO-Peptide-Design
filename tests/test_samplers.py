from collections import Counter

import pandas as pd

from amp_design.samplers import BufferedSortishBatchProvider, SortishBatchConfig
from amp_design.tokenizer import PeptideTokenizer


def _sampler_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": [f"s{index}" for index in range(6)],
            "sequence": ["ACDEF", "ACDEFG", "ACDEFGH", "CDEFG", "DEFGHI", "EFGHIKL"],
            "length": [5, 6, 7, 5, 6, 7],
            "cluster_id": ["large", "large", "large", "large", "large", "small"],
        }
    )


def test_sortish_batches_are_deterministic() -> None:
    config = SortishBatchConfig(batch_size=2, buffer_size=8, seed=42, sampling="sequence")
    first = BufferedSortishBatchProvider(_sampler_frame(), PeptideTokenizer(), config)
    second = BufferedSortishBatchProvider(_sampler_frame(), PeptideTokenizer(), config)
    for step in range(20):
        assert first.indices_for_step(step).tolist() == second.indices_for_step(step).tolist()


def test_cluster_sampling_is_cluster_balanced() -> None:
    config = SortishBatchConfig(batch_size=2, buffer_size=20, seed=7, sampling="cluster")
    provider = BufferedSortishBatchProvider(_sampler_frame(), PeptideTokenizer(), config)
    counts: Counter[str] = Counter()
    for step in range(100):
        batch = provider.batch_for_step(step)
        counts.update(batch.cluster_ids)
    small_fraction = counts["small"] / sum(counts.values())
    assert 0.35 < small_fraction < 0.65


def test_batch_padding_is_reduced_by_sorting() -> None:
    config = SortishBatchConfig(batch_size=2, buffer_size=8, seed=11, sampling="sequence")
    provider = BufferedSortishBatchProvider(_sampler_frame(), PeptideTokenizer(), config)
    for step in range(4):
        batch = provider.batch_for_step(step)
        assert int(batch.lengths.max() - batch.lengths.min()) <= 2
