"""Tokenization and padding for canonical antimicrobial peptide sequences."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
ID_TO_AA = {index: amino_acid for amino_acid, index in AA_TO_ID.items()}
MASK_TOKEN_ID = len(AMINO_ACIDS)
PAD_TOKEN_ID = MASK_TOKEN_ID + 1
INPUT_VOCAB_SIZE = PAD_TOKEN_ID + 1
OUTPUT_VOCAB_SIZE = len(AMINO_ACIDS)
FLOW_VOCAB_SIZE = OUTPUT_VOCAB_SIZE + 1


@dataclass(frozen=True)
class PeptideBatch:
    input_ids: Tensor
    attention_mask: Tensor
    lengths: Tensor
    sequence_ids: tuple[str, ...]
    cluster_ids: tuple[str, ...]

    def to(self, device: torch.device | str) -> PeptideBatch:
        return PeptideBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            lengths=self.lengths.to(device),
            sequence_ids=self.sequence_ids,
            cluster_ids=self.cluster_ids,
        )


class PeptideTokenizer:
    """A fixed tokenizer for the 20 canonical amino acids plus MASK/PAD."""

    amino_acids = AMINO_ACIDS
    mask_token_id = MASK_TOKEN_ID
    pad_token_id = PAD_TOKEN_ID
    input_vocab_size = INPUT_VOCAB_SIZE
    output_vocab_size = OUTPUT_VOCAB_SIZE
    flow_vocab_size = FLOW_VOCAB_SIZE

    def encode(self, sequence: str) -> list[int]:
        normalized = sequence.strip().upper()
        if not normalized:
            raise ValueError("Peptide sequence cannot be empty")
        try:
            return [AA_TO_ID[amino_acid] for amino_acid in normalized]
        except KeyError as error:
            raise ValueError(f"Non-canonical amino acid: {error.args[0]!r}") from error

    def decode(self, token_ids: Sequence[int] | Tensor) -> str:
        values = token_ids.tolist() if isinstance(token_ids, Tensor) else list(token_ids)
        invalid = [token_id for token_id in values if token_id not in ID_TO_AA]
        if invalid:
            raise ValueError(f"Cannot decode non-amino-acid token ids: {invalid[:5]}")
        return "".join(ID_TO_AA[token_id] for token_id in values)

    def collate(
        self,
        sequences: Sequence[str],
        *,
        sequence_ids: Sequence[str] | None = None,
        cluster_ids: Sequence[str] | None = None,
    ) -> PeptideBatch:
        if not sequences:
            raise ValueError("Cannot collate an empty batch")
        encoded = [self.encode(sequence) for sequence in sequences]
        lengths = torch.tensor([len(tokens) for tokens in encoded], dtype=torch.long)
        max_length = int(lengths.max().item())
        input_ids = torch.full(
            (len(encoded), max_length), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(encoded), max_length), dtype=torch.bool)
        for row, tokens in enumerate(encoded):
            length = len(tokens)
            input_ids[row, :length] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[row, :length] = True

        ids = tuple(sequence_ids or (f"sequence_{index}" for index in range(len(encoded))))
        clusters = tuple(cluster_ids or ("" for _ in encoded))
        if len(ids) != len(encoded) or len(clusters) != len(encoded):
            raise ValueError("Sequence and cluster metadata must match batch size")
        return PeptideBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lengths=lengths,
            sequence_ids=ids,
            cluster_ids=clusters,
        )
