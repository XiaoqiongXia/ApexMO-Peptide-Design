import torch

from amp_design.model import AMPDDiT, AMPDDiTConfig
from amp_design.tokenizer import PeptideTokenizer


def _small_model() -> AMPDDiT:
    return AMPDDiT(
        AMPDDiTConfig(
            hidden_size=32,
            n_blocks=2,
            n_heads=4,
            mlp_ratio=2,
            dropout=0.0,
            time_embedding_dim=16,
            length_embedding_dim=16,
            condition_dim=32,
        )
    )


def test_model_shapes_and_parameter_count() -> None:
    tokenizer = PeptideTokenizer()
    batch = tokenizer.collate(["ACDEF", "ACDEFGH"])
    model = _small_model()
    logits = model(
        batch.input_ids,
        torch.tensor([0.2, 0.8]),
        batch.lengths,
        batch.attention_mask,
    )
    assert logits.shape == (2, 7, 20)
    assert sum(parameter.numel() for parameter in model.parameters()) > 10_000


def test_padding_does_not_change_valid_logits() -> None:
    tokenizer = PeptideTokenizer()
    model = _small_model().eval()
    alone = tokenizer.collate(["ACDEF"])
    padded = tokenizer.collate(["ACDEF", "ACDEFGHIK"])
    with torch.no_grad():
        alone_logits = model(
            alone.input_ids,
            torch.tensor([0.4]),
            alone.lengths,
            alone.attention_mask,
        )[0]
        padded_logits = model(
            padded.input_ids,
            torch.tensor([0.4, 0.7]),
            padded.lengths,
            padded.attention_mask,
        )[0, :5]
    torch.testing.assert_close(alone_logits, padded_logits, atol=1e-6, rtol=1e-5)


def test_length_condition_is_present() -> None:
    model = _small_model()
    assert model.length_embedding.num_embeddings == 65
