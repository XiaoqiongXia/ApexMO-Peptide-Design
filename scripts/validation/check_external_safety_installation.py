#!/usr/bin/env python3
"""Check the installation by running one batch through HemoPI2 and ToxinPred3."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from amp_design.predictors.safety import score_hemopi2, score_toxinpred3
from amp_design.workflows.config import PublicationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = PublicationConfig.from_file(args.config)
    candidates = pd.DataFrame(
        {
            "smoke_id": ["SMOKE001", "SMOKE002"],
            "sequence": ["GIGKFLHSAKKFGKAFVGEIMKS", "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"],
        }
    )
    with tempfile.TemporaryDirectory(prefix="amp_external_safety_smoke_", dir="/tmp") as tmp:
        work = Path(tmp)
        hemopi2 = score_hemopi2(
            candidates,
            work,
            id_column="smoke_id",
            root=config.hemopi2_root,
            wrapper=config.hemopi2_wrapper,
            threshold=config.hemopi2_threshold,
        )
        toxinpred3 = score_toxinpred3(
            candidates,
            work,
            id_column="smoke_id",
            root=config.toxinpred3_root,
            runner=config.toxinpred3_runner,
            python=config.toxinpred3_python,
            threshold=config.toxinpred3_threshold,
        )
    print(
        json.dumps(
            {
                "passed": True,
                "rows": len(candidates),
                "hemopi2_predictions": hemopi2["hemopi2_prediction"].tolist(),
                "toxinpred3_predictions": toxinpred3["toxinpred3_prediction"].tolist(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
