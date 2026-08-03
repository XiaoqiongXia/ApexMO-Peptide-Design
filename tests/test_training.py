import subprocess
import sys

import pandas as pd
import torch
from omegaconf import OmegaConf

from amp_design.flow_matching import build_probability_path
from amp_design.model import AMPDDiT, AMPDDiTConfig
from amp_design.tokenizer import PeptideTokenizer
from amp_design.training import _resume_contract, deterministic_validation


def test_deterministic_validation_is_repeatable() -> None:
    frame = pd.DataFrame(
        {
            "sequence_id": ["s1", "s2", "s3"],
            "sequence": ["ACDEF", "GHIKLM", "NPQRSTV"],
            "length": [5, 6, 7],
            "cluster_id": ["c1", "c2", "c3"],
        }
    )
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
        )
    )
    kwargs = {
        "path": build_probability_path(),
        "split_hash": "abc123",
        "batch_size": 2,
        "device": torch.device("cpu"),
        "precision": "fp32",
    }
    first = deterministic_validation(model, frame, PeptideTokenizer(), **kwargs)
    second = deterministic_validation(model, frame, PeptideTokenizer(), **kwargs)
    assert first == second


def test_hydra_config_is_resolvable_outside_repository(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from amp_design.training import main; main()",
            "--cfg",
            "job",
        ],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "sampling: sequence" in result.stdout
    assert "data/processed/splits_v2" in result.stdout


def test_resume_contract_ignores_resume_path_but_binds_flow_configuration() -> None:
    first = OmegaConf.create(
        {"training": {"resume": None, "max_steps": 10}, "flow": {"exponent": 1.0}}
    )
    second = OmegaConf.create(
        {"training": {"resume": "checkpoint.pt", "max_steps": 10}, "flow": {"exponent": 1.0}}
    )
    assert _resume_contract(first) == _resume_contract(second)
    second.flow.exponent = 2.0
    assert _resume_contract(first) != _resume_contract(second)
