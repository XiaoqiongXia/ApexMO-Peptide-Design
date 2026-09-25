"""Sample fixed-length sequences in resumable shards from one checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

from amp_design import __version__
from amp_design.generation.sampling import generate_candidates, load_checkpoint_model
from amp_design.generation.tokenizer import AMINO_ACIDS

GENERATION_SEED_BASE = 73_000_000


@dataclass(frozen=True)
class SamplingTask:
    training_seed: int
    length: int
    shard_id: int
    count: int
    generation_seed: int
    output: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_generation_seed(training_seed: int, length: int, shard_id: int) -> int:
    if training_seed < 0 or not 5 <= length <= 64 or shard_id < 0:
        raise ValueError("Invalid training seed, length, or shard id")
    # Preserve every seed emitted by the v1.3 sampler while removing its
    # artificial 100-shard ceiling.  Larger shard ids use a stable digest so
    # the mapping remains deterministic without integer-field collisions.
    if shard_id < 100:
        return GENERATION_SEED_BASE + training_seed * 10_000 + length * 100 + shard_id
    payload = f"{GENERATION_SEED_BASE}:{training_seed}:{length}:{shard_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def build_tasks(
    output_root: Path,
    *,
    training_seed: int,
    lengths: Sequence[int],
    samples_per_length: int,
    shard_size: int,
) -> list[SamplingTask]:
    if samples_per_length < 1 or shard_size < 1:
        raise ValueError("samples_per_length and shard_size must be positive")
    if samples_per_length % shard_size:
        raise ValueError("samples_per_length must be divisible by shard_size")
    if len(set(lengths)) != len(lengths) or any(not 5 <= length <= 64 for length in lengths):
        raise ValueError("lengths must be unique integers in [5, 64]")
    tasks = []
    for length in sorted(lengths):
        for shard_id in range(samples_per_length // shard_size):
            tasks.append(
                SamplingTask(
                    training_seed=training_seed,
                    length=length,
                    shard_id=shard_id,
                    count=shard_size,
                    generation_seed=derive_generation_seed(training_seed, length, shard_id),
                    output=(
                        output_root
                        / "raw"
                        / f"seed_{training_seed}"
                        / f"length_{length:02d}"
                        / f"shard_{shard_id:03d}.csv"
                    ),
                )
            )
    return tasks


def validate_completed_shard(path: Path, task: SamplingTask, checkpoint_hash: str) -> bool:
    if not path.is_file():
        return False
    frame = pd.read_csv(path)
    required = {
        "candidate_id",
        "sequence",
        "length",
        "training_seed",
        "generation_seed",
        "shard_id",
        "checkpoint_sha256",
        "sampling_steps",
        "use_ema",
    }
    if not required.issubset(frame.columns) or len(frame) != task.count:
        return False
    return bool(
        frame["candidate_id"].nunique() == task.count
        and frame["sequence"].map(lambda value: set(str(value)).issubset(set(AMINO_ACIDS))).all()
        and frame["sequence"].str.len().eq(task.length).all()
        and frame["length"].eq(task.length).all()
        and frame["training_seed"].eq(task.training_seed).all()
        and frame["generation_seed"].eq(task.generation_seed).all()
        and frame["shard_id"].eq(task.shard_id).all()
        and frame["checkpoint_sha256"].eq(checkpoint_hash).all()
        and frame["use_ema"].eq(True).all()  # noqa: E712
    )


def sample_tasks(
    checkpoint: Path,
    output_root: Path,
    *,
    training_seed: int,
    lengths: Sequence[int],
    samples_per_length: int,
    shard_size: int,
    steps: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    checkpoint_hash = sha256(checkpoint)
    tasks = build_tasks(
        output_root,
        training_seed=training_seed,
        lengths=lengths,
        samples_per_length=samples_per_length,
        shard_size=shard_size,
    )
    pending = [
        task for task in tasks if not validate_completed_shard(task.output, task, checkpoint_hash)
    ]
    model = None
    if pending:
        model, checkpoint_payload = load_checkpoint_model(checkpoint, device=device, use_ema=True)
    else:
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    flow_exponent = float(checkpoint_payload["config"]["flow"]["exponent"])
    time_epsilon = float(checkpoint_payload["config"]["generation"]["time_epsilon"])
    max_length = int(checkpoint_payload["config"]["model"]["max_length"])
    for task_index, task in enumerate(pending, start=1):
        candidates = generate_candidates(
            model,
            lengths=[task.length] * task.count,
            checkpoint_path=checkpoint,
            steps=steps,
            seed=task.generation_seed,
            device=device,
            batch_size=batch_size,
            path_exponent=flow_exponent,
            time_epsilon=time_epsilon,
            max_length=max_length,
        )
        candidates.insert(
            0,
            "candidate_id",
            [
                f"amp_s{training_seed}_l{task.length:02d}_h{task.shard_id:03d}_i{index:04d}"
                for index in range(task.count)
            ],
        )
        candidates["training_seed"] = training_seed
        candidates["generation_seed"] = task.generation_seed
        candidates["shard_id"] = task.shard_id
        candidates["checkpoint_sha256"] = checkpoint_hash
        candidates["use_ema"] = True
        task.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = task.output.with_suffix(task.output.suffix + ".tmp")
        candidates.to_csv(temporary, index=False)
        os.replace(temporary, task.output)
        if not validate_completed_shard(task.output, task, checkpoint_hash):
            raise RuntimeError(f"Completed shard failed validation: {task.output}")
        print(f"Completed shard {task_index}/{len(pending)}: {task.output}", flush=True)

    manifest = {
        "batch_size": batch_size,
        "amp_design_version": __version__,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "device": str(device),
        "generation_seed_base": GENERATION_SEED_BASE,
        "flow_path_exponent": flow_exponent,
        "lengths": sorted(lengths),
        "n_candidates": sum(task.count for task in tasks),
        "n_shards": len(tasks),
        "python": platform.python_version(),
        "samples_per_length": samples_per_length,
        "sampling_steps": steps,
        "shard_size": shard_size,
        "time_epsilon": time_epsilon,
        "model_max_length": max_length,
        "tasks": [{**asdict(task), "output": str(task.output.resolve())} for task in tasks],
        "torch": torch.__version__,
        "training_seed": training_seed,
        "use_ema": True,
    }
    manifest_path = output_root / "manifests" / f"seed_{training_seed}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)
    return manifest


def parse_lengths(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("At least one length is required")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--lengths", type=parse_lengths, required=True)
    parser.add_argument("--samples-per-length", type=int, default=1000)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sample_tasks(
        args.checkpoint,
        args.output_root,
        training_seed=args.training_seed,
        lengths=args.lengths,
        samples_per_length=args.samples_per_length,
        shard_size=args.shard_size,
        steps=args.steps,
        batch_size=args.batch_size,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
