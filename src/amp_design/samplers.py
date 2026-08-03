"""Deterministic sequence- and cluster-balanced sortish batch providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .tokenizer import PeptideBatch, PeptideTokenizer

SamplingMode = Literal["sequence", "cluster"]


@dataclass(frozen=True)
class SortishBatchConfig:
    batch_size: int = 256
    buffer_size: int = 2048
    seed: int = 42
    sampling: SamplingMode = "sequence"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.buffer_size < self.batch_size:
            raise ValueError("buffer_size must be at least batch_size")
        if self.buffer_size % self.batch_size:
            raise ValueError("buffer_size must be divisible by batch_size")
        if self.sampling not in {"sequence", "cluster"}:
            raise ValueError(f"Unknown sampling mode: {self.sampling}")


class BufferedSortishBatchProvider:
    """Produce a deterministic batch for each optimizer step.

    Sampling happens before length sorting, so sortish batching does not alter
    sequence- or cluster-balanced marginal probabilities. Batches are a pure
    function of ``seed`` and ``step``, making checkpoint resume exact without
    serializing a mutable sampler RNG.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: PeptideTokenizer,
        config: SortishBatchConfig,
    ) -> None:
        required = {"sequence_id", "sequence", "length", "cluster_id"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing sampler columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("Training frame cannot be empty")
        self.frame = frame.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.config = config
        self.batches_per_buffer = config.buffer_size // config.batch_size
        self.lengths = self.frame["length"].to_numpy(dtype=np.int64)

        grouped = self.frame.groupby("cluster_id", sort=True).indices
        self.cluster_ids = tuple(str(cluster_id) for cluster_id in grouped)
        self.cluster_members = tuple(
            np.asarray(grouped[cluster_id], dtype=np.int64) for cluster_id in grouped
        )

    def _draw_indices(self, rng: np.random.Generator) -> np.ndarray:
        if self.config.sampling == "sequence":
            return rng.integers(0, len(self.frame), size=self.config.buffer_size)

        chosen_clusters = rng.integers(
            0, len(self.cluster_members), size=self.config.buffer_size
        )
        indices = np.empty(self.config.buffer_size, dtype=np.int64)
        for position, cluster_index in enumerate(chosen_clusters):
            members = self.cluster_members[int(cluster_index)]
            indices[position] = members[int(rng.integers(0, len(members)))]
        return indices

    def indices_for_step(self, step: int) -> np.ndarray:
        if step < 0:
            raise ValueError("step cannot be negative")
        buffer_index, position = divmod(step, self.batches_per_buffer)
        seed_sequence = np.random.SeedSequence([self.config.seed, buffer_index])
        rng = np.random.default_rng(seed_sequence)
        sampled = self._draw_indices(rng)
        order = np.argsort(self.lengths[sampled], kind="stable")
        sorted_indices = sampled[order]
        batches = sorted_indices.reshape(self.batches_per_buffer, self.config.batch_size)
        batch_order = rng.permutation(self.batches_per_buffer)
        return batches[int(batch_order[position])].copy()

    def batch_for_step(self, step: int) -> PeptideBatch:
        rows = self.frame.iloc[self.indices_for_step(step)]
        return self.tokenizer.collate(
            rows["sequence"].tolist(),
            sequence_ids=rows["sequence_id"].astype(str).tolist(),
            cluster_ids=rows["cluster_id"].astype(str).tolist(),
        )
