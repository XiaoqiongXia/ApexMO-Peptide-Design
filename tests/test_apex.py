import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amp_design.apex import (
    APEX_STRAINS,
    annotate_apex_eligibility,
    encode_apex_sequences,
    finalize_scores,
    load_strain_groups,
    update_ensemble_statistics,
    validate_apex_state,
    validate_sequences,
)


def test_strain_groups_are_complete() -> None:
    groups = load_strain_groups(Path("configs/apex_strain_groups.yaml"))
    assert len(groups["pathogen"]) == 15
    assert len(groups["commensal"]) == 18
    assert len(groups["ambiguous"]) == 1


def test_validate_sequences_normalizes_and_rejects_invalid() -> None:
    result = validate_sequences(pd.DataFrame({"sequence": [" acdef "]}))
    assert result.loc[0, "sequence"] == "ACDEF"
    with pytest.raises(ValueError, match="canonical sequences"):
        validate_sequences(pd.DataFrame({"sequence": ["AX"]}))


def test_apex_eligibility_does_not_silently_truncate() -> None:
    frame = annotate_apex_eligibility(pd.DataFrame({"sequence": ["ACDEF", "A" * 51]}))
    assert frame["apex_eligible"].tolist() == [True, False]
    assert frame.loc[1, "apex_exclusion_reason"] == "length_gt_50"


def test_apex_encoding_adds_start_end_and_padding() -> None:
    encoded = encode_apex_sequences(["ACD"])
    assert encoded.shape == (1, 52)
    assert encoded[0, :5].tolist() == [1, 3, 4, 5, 2]
    assert encoded[0, 5:].sum() == 0


def test_streaming_statistics_preserve_raw_mic_mean() -> None:
    shape = (2, len(APEX_STRAINS))
    mic_sum = np.zeros(shape)
    log_sum = np.zeros(shape)
    square_sum = np.zeros(shape)
    first = np.full(shape, 1.0)
    second = np.full(shape, 3.0)
    update_ensemble_statistics(mic_sum, log_sum, square_sum, first)
    update_ensemble_statistics(mic_sum, log_sum, square_sum, second)
    groups = load_strain_groups(Path("configs/apex_strain_groups.yaml"))
    scores = finalize_scores(mic_sum, log_sum, square_sum, 2, groups)
    mic_columns = scores.filter(like="apex_mic__")
    assert np.allclose(mic_columns.to_numpy(), 505.0)
    assert np.allclose(scores["apex_median_log10_mic_sd"], np.sqrt(2.0))
    assert np.allclose(scores["apex_commensal_selectivity_margin"], 0.0)


def test_resumable_state_is_bound_to_complete_inference_contract(tmp_path: Path) -> None:
    contract = {"model_paths": ["m1", "m2"], "identity": "frozen"}
    contract_json = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    shape = (2, len(APEX_STRAINS))
    state_path = tmp_path / "state.npz"
    np.savez(
        state_path,
        schema_version=np.asarray(2),
        completed_models=np.asarray(1),
        sequence_hash=np.asarray("sequences"),
        contract_json=np.asarray(contract_json),
        mic_sum=np.zeros(shape, dtype=np.float64),
        log_sum=np.zeros(shape, dtype=np.float64),
        log_square_sum=np.zeros(shape, dtype=np.float64),
    )
    with np.load(state_path, allow_pickle=False) as state:
        completed, *_ = validate_apex_state(
            state,
            sequence_hash="sequences",
            contract=contract,
            shape=shape,
        )
    assert completed == 1
    changed_contract = {**contract, "identity": "changed"}
    with np.load(state_path, allow_pickle=False) as state:
        with pytest.raises(ValueError, match="different inference contract"):
            validate_apex_state(
                state,
                sequence_hash="sequences",
                contract=changed_contract,
                shape=shape,
            )
