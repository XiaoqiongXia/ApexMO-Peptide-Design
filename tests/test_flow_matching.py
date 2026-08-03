import pytest
import torch

from amp_design.flow_matching import (
    AminoAcidPosteriorWrapper,
    build_probability_path,
    flow_matching_loss,
    generate_token_ids,
    masked_cross_entropy,
    sample_masked_path,
)
from amp_design.model import AMPDDiT, AMPDDiTConfig
from amp_design.tokenizer import MASK_TOKEN_ID, PAD_TOKEN_ID, PeptideTokenizer


def _small_model() -> AMPDDiT:
    return AMPDDiT(
        AMPDDiTConfig(
            hidden_size=32,
            n_blocks=1,
            n_heads=4,
            mlp_ratio=2,
            dropout=0.0,
            time_embedding_dim=16,
            length_embedding_dim=16,
            condition_dim=32,
        )
    )


def test_masked_path_keeps_padding_fixed() -> None:
    batch = PeptideTokenizer().collate(["ACDEF", "ACDEFGH"])
    path = build_probability_path()
    x_t = sample_masked_path(
        path,
        batch.input_ids,
        batch.attention_mask,
        torch.tensor([0.0, 1.0]),
        uniform=torch.full(batch.input_ids.shape, 0.5),
    )
    assert x_t[0, :5].eq(MASK_TOKEN_ID).all()
    assert x_t[0, 5:].eq(PAD_TOKEN_ID).all()
    assert torch.equal(x_t[1], batch.input_ids[1])


def test_masked_cross_entropy_ignores_padding() -> None:
    batch = PeptideTokenizer().collate(["ACDEF", "ACDEFGH"])
    logits = torch.randn(2, 7, 20)
    baseline = masked_cross_entropy(logits, batch.input_ids, batch.attention_mask)
    changed = logits.clone()
    changed[0, 5:] = 10_000
    torch.testing.assert_close(
        baseline,
        masked_cross_entropy(changed, batch.input_ids, batch.attention_mask),
    )


def test_flow_loss_is_finite() -> None:
    batch = PeptideTokenizer().collate(["ACDEF", "ACDEFGH"])
    loss, time = flow_matching_loss(_small_model(), batch, build_probability_path())
    assert torch.isfinite(loss)
    assert time.shape == (2,)


def test_wrapper_assigns_zero_mask_probability() -> None:
    batch = PeptideTokenizer().collate(["ACDEF", "ACDEFGH"])
    wrapper = AminoAcidPosteriorWrapper(_small_model())
    probability = wrapper(
        batch.input_ids,
        torch.tensor([0.2, 0.7]),
        lengths=batch.lengths,
        attention_mask=batch.attention_mask,
    )
    assert probability.shape == (2, 7, 21)
    assert probability[..., -1].eq(0).all()
    torch.testing.assert_close(probability.sum(-1), torch.ones(2, 7))


def test_generation_returns_exact_lengths_without_mask() -> None:
    model = _small_model().eval()
    samples = generate_token_ids(model, lengths=[5, 7, 5], steps=4, device="cpu")
    assert [len(sample) for sample in samples] == [5, 7, 5]
    assert all(sample.lt(20).all() for sample in samples)


def test_generation_uses_model_max_length_instead_of_hard_coded_64() -> None:
    model = AMPDDiT(
        AMPDDiTConfig(
            hidden_size=32,
            n_blocks=1,
            n_heads=4,
            mlp_ratio=2,
            dropout=0.0,
            time_embedding_dim=16,
            length_embedding_dim=16,
            condition_dim=32,
            max_length=8,
        )
    ).eval()
    with pytest.raises(ValueError, match=r"\[1, 8\]"):
        generate_token_ids(model, lengths=[9], steps=4, device="cpu")
