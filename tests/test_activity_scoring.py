import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from amp_design.activity_scoring import aggregate_activity_targets, score_activity_command
from amp_design.apex import strain_slug
from amp_design.release_optimization import _aggregate_configured_activity


def _pathogen_column(name: str) -> str:
    return f"apex_pathogen_mic__{strain_slug(name)}"


def _pathogen_uncertainty_column(name: str) -> str:
    return f"apex_pathogen_log10_mic_sd__{strain_slug(name)}"


def test_single_target_activity_is_direct_log10_mic() -> None:
    target = "P. aeruginosa PA14"
    scores = pd.DataFrame(
        {
            _pathogen_column(target): [1.0, 10.0, 100.0],
            _pathogen_uncertainty_column(target): [0.1, 0.2, 0.3],
        }
    )

    result = aggregate_activity_targets(
        scores,
        backend="apex_pathogen",
        targets=[target],
        aggregation="median",
        quantile=0.5,
        output_column="activity_score",
    )

    assert np.allclose(result["activity_score"], [0.0, 1.0, 2.0])
    assert np.allclose(result["apex_median_log10_mic_sd"], [0.1, 0.2, 0.3])
    assert result["activity_target_count"].eq(1).all()


def test_multi_target_max_optimizes_worst_log10_mic() -> None:
    first = "P. aeruginosa PA14"
    second = "K. pneumoniae ATCC 13883"
    scores = pd.DataFrame(
        {
            _pathogen_column(first): [4.0, 64.0],
            _pathogen_column(second): [16.0, 8.0],
            _pathogen_uncertainty_column(first): [0.1, 0.4],
            _pathogen_uncertainty_column(second): [0.3, 0.2],
        }
    )

    result = aggregate_activity_targets(
        scores,
        backend="apex_pathogen",
        targets=[first, second],
        aggregation="max",
        quantile=0.5,
        output_column="activity_score",
    )

    assert np.allclose(result["activity_score"], np.log10([16.0, 64.0]))
    assert np.allclose(result["apex_median_log10_mic_sd"], [0.2, 0.3])
    assert np.allclose(result["apex_max_log10_mic_sd"], [0.3, 0.4])


def test_legacy_apex_selected_targets_replace_global_uncertainty() -> None:
    first = "P. aeruginosa PAO1"
    second = "A. baumannii ATCC19606"
    scores = pd.DataFrame(
        {
            f"apex_mic__{strain_slug(first)}": [8.0],
            f"apex_mic__{strain_slug(second)}": [32.0],
            f"apex_log10_mic_sd__{strain_slug(first)}": [0.1],
            f"apex_log10_mic_sd__{strain_slug(second)}": [0.3],
            "apex_median_log10_mic_sd": [9.9],
            "apex_max_log10_mic_sd": [9.9],
        }
    )

    result = aggregate_activity_targets(
        scores,
        backend="apex",
        targets=[first, second],
        aggregation="median",
        quantile=0.5,
        output_column="apex_pathogen_median_log10_mic",
    )

    assert np.allclose(result["apex_pathogen_median_log10_mic"], np.log10(16.0))
    assert np.allclose(result["apex_median_log10_mic_sd"], [0.2])
    assert np.allclose(result["apex_max_log10_mic_sd"], [0.3])


def test_activity_target_rejects_unknown_pathogen() -> None:
    with pytest.raises(ValueError, match="Unknown apex_pathogen activity targets"):
        aggregate_activity_targets(
            pd.DataFrame(),
            backend="apex_pathogen",
            targets=["not-a-pathogen"],
            aggregation="median",
            quantile=0.5,
            output_column="activity_score",
        )


def test_configured_activity_allows_empty_fresh_scoring_cache() -> None:
    empty = pd.DataFrame(columns=["sequence", "toxicity_score"])
    config = SimpleNamespace(
        activity_scorer="apex",
        activity_targets=("P. aeruginosa PAO1",),
        activity_aggregation="median",
        activity_quantile=0.5,
        effective_activity_output_column="activity_score",
    )

    result = _aggregate_configured_activity(empty, config)

    assert result.empty
    assert result.columns.tolist() == empty.columns.tolist()


def test_command_activity_scorer_uses_csv_contract(tmp_path: Path) -> None:
    script = tmp_path / "score.py"
    script.write_text(
        "import sys\n"
        "import pandas as pd\n"
        "frame = pd.read_csv(sys.argv[1])\n"
        "frame['custom_score'] = frame.sequence.str.len() / 10\n"
        "frame.to_csv(sys.argv[2], index=False)\n",
        encoding="utf-8",
    )

    result = score_activity_command(
        ["ACDE", "KLMNPQ"],
        [sys.executable, str(script), "{input}", "{output}"],
        output_column="custom_score",
        work_dir=tmp_path / "work",
    )

    assert result["sequence"].tolist() == ["ACDE", "KLMNPQ"]
    assert np.allclose(result["custom_score"], [0.4, 0.6])


def test_command_activity_scorer_rejects_reserved_columns(tmp_path: Path) -> None:
    script = tmp_path / "score.py"
    script.write_text(
        "import sys\n"
        "import pandas as pd\n"
        "frame = pd.read_csv(sys.argv[1])\n"
        "frame['custom_score'] = 1.0\n"
        "frame['length'] = 999\n"
        "frame.to_csv(sys.argv[2], index=False)\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reserved columns"):
        score_activity_command(
            ["ACDE"],
            [sys.executable, str(script), "{input}", "{output}"],
            output_column="custom_score",
            work_dir=tmp_path / "work",
        )
