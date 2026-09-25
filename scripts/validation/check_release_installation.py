#!/usr/bin/env python3
"""Check configuration files and external tools without running model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from amp_design.workflows.cli import check_optimization
from amp_design.workflows.config import OptimizationConfig, PipelineConfig, PublicationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    pipeline = PipelineConfig.from_file(args.config)
    optimization = OptimizationConfig.from_file(
        args.config, require_search_inputs=False
    )
    publication = PublicationConfig.from_file(args.config)
    report = check_optimization(optimization)
    print(
        json.dumps(
            {
                "passed": True,
                "sampling_manifests": len(pipeline.sampling_manifests),
                "optimization_seeds": list(optimization.seeds),
                "publication_output": str(publication.output_dir),
                "activity_scorer": report["activity_scorer"],
                "n_apex_models": report["n_apex_models"],
                "mmseqs_version": report["mmseqs_version"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
