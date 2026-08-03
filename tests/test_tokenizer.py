import pytest
import torch

from amp_design.tokenizer import (
    AMINO_ACIDS,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    PeptideTokenizer,
)


def test_tokenizer_round_trip() -> None:
    tokenizer = PeptideTokenizer()
    assert tokenizer.decode(tokenizer.encode(AMINO_ACIDS)) == AMINO_ACIDS


def test_tokenizer_rejects_invalid_tokens() -> None:
    tokenizer = PeptideTokenizer()
    with pytest.raises(ValueError, match="Non-canonical"):
        tokenizer.encode("ACDX")
    with pytest.raises(ValueError, match="non-amino-acid"):
        tokenizer.decode([0, MASK_TOKEN_ID, PAD_TOKEN_ID])


def test_collate_masks_padding() -> None:
    tokenizer = PeptideTokenizer()
    batch = tokenizer.collate(
        ["ACDEF", "ACDEFGH"], sequence_ids=["a", "b"], cluster_ids=["c1", "c2"]
    )
    assert batch.input_ids.shape == (2, 7)
    assert torch.equal(batch.lengths, torch.tensor([5, 7]))
    assert batch.attention_mask.sum().item() == 12
    assert batch.input_ids[0, 5:].eq(PAD_TOKEN_ID).all()
