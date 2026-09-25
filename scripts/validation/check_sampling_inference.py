#!/usr/bin/env python3
"""Load a released Flow Matching checkpoint and generate a tiny validated batch."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from amp_design.generation.sharded_sampling import sample_tasks
from amp_design.workflows.cli import resolve_device
from amp_design.workflows.config import SamplingRunsConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    sampling = SamplingRunsConfig.from_file(args.config).runs[0]
    length = min(sampling.lengths)
    device = resolve_device(sampling.device)
    with tempfile.TemporaryDirectory(prefix="amp_sampling_smoke_", dir="/tmp") as tmp:
        manifest = sample_tasks(
            sampling.checkpoint,
            Path(tmp),
            training_seed=sampling.training_seed,
            lengths=(length,),
            samples_per_length=2,
            shard_size=2,
            steps=4,
            batch_size=2,
            device=device,
        )
        shard = Path(manifest["tasks"][0]["output"])
        frame = pd.read_csv(shard)
        if len(frame) != 2 or not frame["sequence"].str.len().eq(length).all():
            raise RuntimeError("Flow Matching smoke output failed length/cardinality validation")
        sequences = frame["sequence"].tolist()
    print(
        json.dumps(
            {
                "passed": True,
                "checkpoint": str(sampling.checkpoint),
                "device": str(device),
                "length": length,
                "sequences": sequences,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
