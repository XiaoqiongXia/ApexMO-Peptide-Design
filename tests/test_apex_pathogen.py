from pathlib import Path

import numpy as np
import pytest
import torch

from amp_design.apex import update_ensemble_statistics
from amp_design.apex_pathogen import (
    APEX_PATHOGEN_STRAINS,
    discover_pathogen_models,
    finalize_pathogen_scores,
    score_apex_pathogen_ensemble,
)


def test_apex_pathogen_discovers_exactly_eight_models(tmp_path: Path) -> None:
    model_dir = tmp_path / "APEX_pathogen_models"
    model_dir.mkdir()
    for index in range(8):
        (model_dir / f"APEX_{index}.pt").write_bytes(b"checkpoint")

    models = discover_pathogen_models(tmp_path)

    assert len(models) == 8
    assert models == sorted(models)


def test_apex_pathogen_statistics_preserve_raw_mic_mean() -> None:
    shape = (2, len(APEX_PATHOGEN_STRAINS))
    mic_sum = np.zeros(shape)
    log_sum = np.zeros(shape)
    square_sum = np.zeros(shape)
    update_ensemble_statistics(mic_sum, log_sum, square_sum, np.full(shape, 1.0))
    update_ensemble_statistics(mic_sum, log_sum, square_sum, np.full(shape, 3.0))
    scores = finalize_pathogen_scores(mic_sum, log_sum, square_sum, 2)
    mic_columns = scores.filter(like="apex_pathogen_mic__")
    assert mic_columns.shape[1] == 11
    assert np.allclose(mic_columns.to_numpy(), 505.0)
    assert np.allclose(scores["apex_pathogen_median_log10_mic_sd"], np.sqrt(2.0))
    assert np.allclose(scores["apex_median_log10_mic_sd"], np.sqrt(2.0))
    assert np.allclose(scores["apex_pathogen_median_log10_mic"], np.log10(505.0))


def test_apex_pathogen_rejects_nonfinite_ensemble_statistics() -> None:
    shape = (1, len(APEX_PATHOGEN_STRAINS))
    mic_sum = np.full(shape, np.nan)
    with pytest.raises(ValueError, match="non-finite or non-positive"):
        finalize_pathogen_scores(mic_sum, np.zeros(shape), np.zeros(shape), 8)


@pytest.mark.parametrize("sequence", ["ACDX", "A" * 51, ""])
def test_apex_pathogen_rejects_invalid_direct_inputs(
    tmp_path: Path, sequence: str
) -> None:
    with pytest.raises(ValueError, match="canonical sequences"):
        score_apex_pathogen_ensemble(
            [sequence],
            tmp_path,
            tmp_path / "state.npz",
            device=torch.device("cpu"),
            batch_size=1,
        )
