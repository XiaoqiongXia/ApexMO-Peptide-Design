"""Plan and validate immutable raw outputs for the generative-model benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
RAW_COLUMNS = (
    "sample_id",
    "model",
    "checkpoint_sha256",
    "sampling_seed",
    "derived_seed",
    "requested_length",
    "shard_id",
    "sample_index",
    "sequence_raw",
    "sequence_canonical",
    "observed_length",
    "generation_status",
    "failure_reason",
    "runtime_seconds",
    "device",
    "sampling_parameters_json",
    "created_at",
)


@dataclass(frozen=True)
class BenchmarkTask:
    model: str
    phase: str
    sampling_seed: int
    derived_seed: int
    requested_length: int
    shard_id: int
    count: int
    output: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict[str, object]:
    protocol = yaml.safe_load(path.read_text())
    if not isinstance(protocol, dict) or protocol.get("version") != "generative_model_benchmark_v1":
        raise ValueError("Expected the frozen generative_model_benchmark_v1 protocol")
    required_models = {"flow_matching", "amp_diffusion", "arcadiamp"}
    if set(protocol["scope"]["included_models"]) != required_models:
        raise ValueError("Frozen v1 model set has changed")
    lengths = protocol["sampling"]["lengths"]
    seeds = protocol["sampling"]["sampling_seeds"]
    if lengths != list(range(10, 31)) or seeds != [42, 123, 2025]:
        raise ValueError("Frozen v1 lengths or sampling seeds have changed")
    return protocol


def derive_seed(sampling_seed: int, requested_length: int, shard_id: int) -> int:
    if sampling_seed < 0 or not 10 <= requested_length <= 30 or shard_id < 0:
        raise ValueError("Invalid sampling seed, requested length, or shard id")
    return sampling_seed * 1_000_000 + requested_length * 1_000 + shard_id


def build_tasks(
    protocol: Mapping[str, object], protocol_path: Path, *, phase: str
) -> list[BenchmarkTask]:
    if phase not in {"pilot", "formal"}:
        raise ValueError("phase must be pilot or formal")
    sampling = protocol["sampling"]
    per_cell = int(sampling[f"{phase}_samples_per_seed_length"])
    shard_size = int(sampling[f"{phase}_shard_size"])
    if per_cell % shard_size:
        raise ValueError("Samples per seed/length must be divisible by shard size")
    project_root = protocol_path.resolve().parent.parent
    output_root = project_root / protocol["outputs"]["root"] / phase / "raw"
    tasks = []
    for model in protocol["scope"]["included_models"]:
        for sampling_seed in sampling["sampling_seeds"]:
            for length in sampling["lengths"]:
                for shard_id in range(per_cell // shard_size):
                    tasks.append(
                        BenchmarkTask(
                            model=model,
                            phase=phase,
                            sampling_seed=sampling_seed,
                            derived_seed=derive_seed(sampling_seed, length, shard_id),
                            requested_length=length,
                            shard_id=shard_id,
                            count=shard_size,
                            output=(
                                output_root
                                / model
                                / f"seed_{sampling_seed}"
                                / f"length_{length:02d}"
                                / f"shard_{shard_id:03d}.csv"
                            ),
                        )
                    )
    return tasks


def canonicalize_sequence(value: object) -> str:
    return str(value).strip().upper()


def make_raw_frame(
    sequences: Sequence[object],
    *,
    task: BenchmarkTask,
    checkpoint_hash: str,
    runtime_seconds: float,
    device: str,
    sampling_parameters: Mapping[str, object],
    errors: Sequence[str | None] | None = None,
) -> pd.DataFrame:
    if len(sequences) != task.count:
        raise ValueError(f"Adapter returned {len(sequences)} rows; expected {task.count}")
    if errors is None:
        errors = [None] * task.count
    if len(errors) != task.count:
        raise ValueError("errors must have one entry per raw attempt")
    created_at = datetime.now(timezone.utc).isoformat()
    per_sample_runtime = runtime_seconds / task.count
    parameters_json = json.dumps(dict(sampling_parameters), sort_keys=True, separators=(",", ":"))
    rows = []
    for index, (raw, adapter_error) in enumerate(zip(sequences, errors, strict=True)):
        canonical = canonicalize_sequence(raw) if raw is not None else ""
        reasons = []
        if adapter_error:
            reasons.append(str(adapter_error))
        if not canonical:
            reasons.append("empty_sequence")
        if canonical and not set(canonical).issubset(CANONICAL_AMINO_ACIDS):
            reasons.append("noncanonical_amino_acid")
        if len(canonical) != task.requested_length:
            reasons.append("length_mismatch")
        status = "success" if not reasons else "failed"
        rows.append(
            {
                "sample_id": (
                    f"{task.model}_s{task.sampling_seed}_l{task.requested_length:02d}_"
                    f"h{task.shard_id:03d}_i{index:04d}"
                ),
                "model": task.model,
                "checkpoint_sha256": checkpoint_hash,
                "sampling_seed": task.sampling_seed,
                "derived_seed": task.derived_seed,
                "requested_length": task.requested_length,
                "shard_id": task.shard_id,
                "sample_index": index,
                "sequence_raw": "" if raw is None else str(raw),
                "sequence_canonical": canonical,
                "observed_length": len(canonical),
                "generation_status": status,
                "failure_reason": ";".join(dict.fromkeys(reasons)),
                "runtime_seconds": per_sample_runtime,
                "device": device,
                "sampling_parameters_json": parameters_json,
                "created_at": created_at,
            }
        )
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def validate_raw_frame(frame: pd.DataFrame, task: BenchmarkTask, checkpoint_hash: str) -> list[str]:
    errors = []
    if list(frame.columns) != list(RAW_COLUMNS):
        errors.append("raw columns do not exactly match the frozen schema")
        return errors
    if len(frame) != task.count:
        errors.append(f"expected {task.count} rows, observed {len(frame)}")
    checks = {
        "sample_id uniqueness": frame["sample_id"].nunique() == len(frame),
        "model": frame["model"].eq(task.model).all(),
        "checkpoint": frame["checkpoint_sha256"].eq(checkpoint_hash).all(),
        "sampling seed": frame["sampling_seed"].eq(task.sampling_seed).all(),
        "derived seed": frame["derived_seed"].eq(task.derived_seed).all(),
        "requested length": frame["requested_length"].eq(task.requested_length).all(),
        "shard id": frame["shard_id"].eq(task.shard_id).all(),
        "sample indices": frame["sample_index"].tolist() == list(range(len(frame))),
        "status values": frame["generation_status"].isin(["success", "failed"]).all(),
    }
    errors.extend(f"invalid {label}" for label, valid in checks.items() if not valid)
    return errors


def write_immutable_shard(frame: pd.DataFrame, task: BenchmarkTask, checkpoint_hash: str) -> None:
    errors = validate_raw_frame(frame, task, checkpoint_hash)
    if errors:
        raise ValueError("; ".join(errors))
    task.output.parent.mkdir(parents=True, exist_ok=True)
    if task.output.exists():
        existing = pd.read_csv(task.output, keep_default_na=False)
        existing_errors = validate_raw_frame(existing, task, checkpoint_hash)
        if existing_errors or sha256(task.output) != _csv_hash(frame):
            raise FileExistsError(f"Refusing to overwrite differing raw shard: {task.output}")
        return
    temporary = task.output.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, task.output)


def _csv_hash(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


def task_from_args(args: argparse.Namespace) -> BenchmarkTask:
    return BenchmarkTask(
        model=args.model,
        phase=args.phase,
        sampling_seed=args.sampling_seed,
        derived_seed=derive_seed(args.sampling_seed, args.requested_length, args.shard_id),
        requested_length=args.requested_length,
        shard_id=args.shard_id,
        count=args.count,
        output=args.output,
    )


def write_plan(protocol_path: Path, phase: str, output: Path) -> None:
    protocol = load_protocol(protocol_path)
    tasks = build_tasks(protocol, protocol_path, phase=phase)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256(protocol_path),
        "phase": phase,
        "n_tasks": len(tasks),
        "n_raw_attempts": sum(task.count for task in tasks),
        "tasks": [{**asdict(task), "output": str(task.output)} for task in tasks],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Write the immutable task plan")
    plan.add_argument("--protocol", type=Path, required=True)
    plan.add_argument("--phase", choices=["pilot", "formal"], required=True)
    plan.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate", help="Validate all existing shards")
    validate.add_argument("--protocol", type=Path, required=True)
    validate.add_argument("--phase", choices=["pilot", "formal"], required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        write_plan(args.protocol, args.phase, args.output)
        return
    protocol = load_protocol(args.protocol)
    tasks = build_tasks(protocol, args.protocol, phase=args.phase)
    invalid = []
    complete = 0
    for task in tasks:
        if not task.output.exists():
            continue
        model_config = protocol["models"][task.model]
        checkpoint_hash = model_config.get("checkpoint_sha256") or model_config.get(
            "checkpoint_lfs_oid_sha256"
        )
        errors = validate_raw_frame(
            pd.read_csv(task.output, keep_default_na=False), task, checkpoint_hash
        )
        if errors:
            invalid.append({"output": str(task.output), "errors": errors})
        else:
            complete += 1
    print(json.dumps({"complete": complete, "invalid": invalid, "total": len(tasks)}, indent=2))
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
