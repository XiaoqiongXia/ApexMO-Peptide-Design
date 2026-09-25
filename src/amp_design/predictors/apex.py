"""APEX ensemble MIC scoring with per-model checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from amp_design.generation.tokenizer import AMINO_ACIDS

APEX_STRAINS = (
    "E. coli ATCC11775",
    "P. aeruginosa PAO1",
    "P. aeruginosa PA14",
    "S. aureus ATCC12600",
    "E. coli AIG221",
    "E. coli AIG222",
    "K. pneumoniae ATCC13883",
    "A. baumannii ATCC19606",
    "A. muciniphila ATCC BAA-835",
    "B. fragilis ATCC25285",
    "B. vulgatus ATCC8482",
    "C. aerofaciens ATCC25986",
    "C. scindens ATCC35704",
    "B. thetaiotaomicron ATCC29148",
    "B. thetaiotaomicron Complemmented",
    "B. thetaiotaomicron Mutant",
    "B. uniformis ATCC8492",
    "B. eggerthi ATCC27754",
    "C. spiroforme ATCC29900",
    "P. distasonis ATCC8503",
    "P. copri DSMZ18205",
    "B. ovatus ATCC8483",
    "E. rectale ATCC33656",
    "C. symbiosum",
    "R. obeum",
    "R. torques",
    "S. aureus (ATCC BAA-1556) - MRSA",
    "vancomycin-resistant E. faecalis ATCC700802",
    "vancomycin-resistant E. faecium ATCC700221",
    "E. coli Nissle",
    "Salmonella enterica ATCC 9150 (BEIRES NR-515)",
    "Salmonella enterica (BEIRES NR-170)",
    "Salmonella enterica ATCC 9150 (BEIRES NR-174)",
    "L. monocytogenes ATCC 19111 (BEIRES NR-106)",
)
APEX_STATE_SCHEMA_VERSION = 2


def strain_slug(name: str) -> str:
    """Return a stable machine-readable strain label."""

    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_strain_groups(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    groups = {name: list(payload[name]) for name in ("pathogen", "commensal", "ambiguous")}
    flattened = [strain for group in groups.values() for strain in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError("APEX strain groups contain duplicate entries")
    if set(flattened) != set(APEX_STRAINS):
        missing = sorted(set(APEX_STRAINS).difference(flattened))
        extra = sorted(set(flattened).difference(APEX_STRAINS))
        raise ValueError(f"APEX strain grouping mismatch; missing={missing}, extra={extra}")
    if len(groups["pathogen"]) != 15 or len(groups["commensal"]) != 18:
        raise ValueError("Expected 15 pathogen and 18 commensal APEX strains")
    return {"version": str(payload["version"]), **groups}


def validate_sequences(frame: pd.DataFrame) -> pd.DataFrame:
    if "sequence" not in frame:
        raise ValueError("Input CSV must contain a sequence column")
    result = frame.copy()
    result["sequence"] = result["sequence"].astype(str).str.strip().str.upper()
    invalid = ~result["sequence"].map(
        lambda value: bool(value) and len(value) <= 50 and set(value).issubset(set(AMINO_ACIDS))
    )
    if invalid.any():
        examples = result.loc[invalid, "sequence"].head().tolist()
        raise ValueError(f"APEX requires canonical sequences of length 1-50; examples={examples}")
    return result


def annotate_apex_eligibility(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize sequences and explicitly label inputs outside the APEX domain."""

    if "sequence" not in frame:
        raise ValueError("Input CSV must contain a sequence column")
    result = frame.copy()
    result["sequence"] = result["sequence"].astype(str).str.strip().str.upper()
    canonical = result["sequence"].map(
        lambda value: bool(value) and set(value).issubset(set(AMINO_ACIDS))
    )
    supported_length = result["sequence"].str.len().between(1, 50)
    result["apex_eligible"] = canonical & supported_length
    result["apex_exclusion_reason"] = ""
    result.loc[~canonical, "apex_exclusion_reason"] = "noncanonical_or_empty"
    result.loc[canonical & ~supported_length, "apex_exclusion_reason"] = "length_gt_50"
    return result


def discover_models(apex_root: Path) -> list[Path]:
    keys = [
        line.strip()
        for line in (apex_root / "best_key_list").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    models = [
        apex_root / "trained_models" / f"trained_all_model_{key}_ensemble_{repeat}"
        for key in keys
        for repeat in range(5)
    ]
    missing = [str(path) for path in models if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing APEX checkpoints: {missing}")
    if len(models) != 40:
        raise ValueError(f"Expected 40 APEX checkpoints, found {len(models)}")
    return models


def _load_apex_utils(apex_root: Path) -> Any:
    root = str(apex_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("AMP_DL_model_twohead")


def encode_apex_sequences(sequences: Sequence[str], max_length: int = 52) -> np.ndarray:
    """Encode sequences exactly as APEX utils.onehot_encoding for valid inputs."""

    vocabulary = {token: index for index, token in enumerate("012ACDEFGHIKLMNPQRSTVWY")}
    encoded = np.zeros((len(sequences), max_length), dtype=np.int64)
    for row, sequence in enumerate(sequences):
        framed = "1" + sequence[: max_length - 2].upper() + "2"
        encoded[row, : len(framed)] = [vocabulary[token] for token in framed]
    return encoded


def _atomic_save_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _apex_commit(apex_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(apex_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_apex_state_contract(
    models: Sequence[Path],
    apex_root: Path,
    groups: dict[str, Any],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Build the immutable inference identity stored in every resumable state."""

    ordered_models = [Path(path).resolve() for path in models]
    group_payload = json.dumps(groups, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": APEX_STATE_SCHEMA_VERSION,
        "apex_commit": _apex_commit(apex_root),
        "apex_inference_code_sha256": sha256(apex_root / "AMP_DL_model_twohead.py"),
        "apex_model_key_list_sha256": sha256(apex_root / "best_key_list"),
        "model_paths": [str(path) for path in ordered_models],
        "model_sha256": [sha256(path) for path in ordered_models],
        "strain_order": list(APEX_STRAINS),
        "strain_groups": groups,
        "strain_groups_sha256": hashlib.sha256(group_payload.encode()).hexdigest(),
        "inference_code_sha256": sha256(Path(__file__)),
        "batch_size": batch_size,
        "torch_version": torch.__version__,
    }


def validate_apex_state(
    state: Any,
    *,
    sequence_hash: str,
    contract: dict[str, Any],
    shape: tuple[int, int],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Validate state schema, inference identity, bounds, shapes, and finite arrays."""

    required = {
        "schema_version",
        "completed_models",
        "sequence_hash",
        "contract_json",
        "mic_sum",
        "log_sum",
        "log_square_sum",
    }
    missing = required.difference(state.files)
    if missing:
        raise ValueError(f"Existing APEX state lacks fields: {sorted(missing)}")
    if int(state["schema_version"].item()) != APEX_STATE_SCHEMA_VERSION:
        raise ValueError("Existing APEX state has an incompatible schema version")
    if str(state["sequence_hash"].item()) != sequence_hash:
        raise ValueError("Existing APEX state belongs to different input sequences")
    expected_contract = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    if str(state["contract_json"].item()) != expected_contract:
        raise ValueError("Existing APEX state belongs to a different inference contract")
    completed = int(state["completed_models"].item())
    if not 0 <= completed <= len(contract["model_paths"]):
        raise ValueError("Existing APEX state has an invalid completed-model count")
    arrays = tuple(
        np.asarray(state[name]).copy() for name in ("mic_sum", "log_sum", "log_square_sum")
    )
    for name, array in zip(("mic_sum", "log_sum", "log_square_sum"), arrays, strict=True):
        if array.shape != shape:
            raise ValueError(f"Existing APEX state {name} shape mismatch")
        if array.dtype != np.float64:
            raise ValueError(f"Existing APEX state {name} must use float64")
        if not np.isfinite(array).all():
            raise ValueError(f"Existing APEX state {name} contains nonfinite values")
    return completed, *arrays


def update_ensemble_statistics(
    mic_sum: np.ndarray,
    log_sum: np.ndarray,
    log_square_sum: np.ndarray,
    log_mic: np.ndarray,
) -> None:
    mic = np.power(10.0, log_mic, dtype=np.float64)
    mic_sum += mic
    log_sum += log_mic
    log_square_sum += np.square(log_mic)


def finalize_scores(
    mic_sum: np.ndarray,
    log_sum: np.ndarray,
    log_square_sum: np.ndarray,
    count: int,
    groups: dict[str, Any],
) -> pd.DataFrame:
    if count < 2:
        raise ValueError("At least two ensemble members are required")
    mean_mic = mic_sum / count
    variance = np.maximum((log_square_sum - np.square(log_sum) / count) / (count - 1), 0.0)
    log_sd = np.sqrt(variance)
    values: dict[str, np.ndarray] = {}
    for index, strain in enumerate(APEX_STRAINS):
        slug = strain_slug(strain)
        values[f"apex_mic__{slug}"] = mean_mic[:, index]
        values[f"apex_log10_mic_sd__{slug}"] = log_sd[:, index]
    pathogen_indices = [APEX_STRAINS.index(strain) for strain in groups["pathogen"]]
    commensal_indices = [APEX_STRAINS.index(strain) for strain in groups["commensal"]]
    log_mean_mic = np.log10(mean_mic)
    pathogen_median = np.median(log_mean_mic[:, pathogen_indices], axis=1)
    commensal_median = np.median(log_mean_mic[:, commensal_indices], axis=1)
    values["apex_pathogen_median_log10_mic"] = pathogen_median
    values["apex_commensal_median_log10_mic"] = commensal_median
    values["apex_commensal_selectivity_margin"] = commensal_median - pathogen_median
    values["apex_median_log10_mic_sd"] = np.median(log_sd, axis=1)
    values["apex_max_log10_mic_sd"] = np.max(log_sd, axis=1)
    return pd.DataFrame(values)


def score_apex_ensemble(
    sequences: Sequence[str],
    apex_root: Path,
    groups: dict[str, Any],
    state_path: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, list[Path]]:
    """Score sequences with all APEX models, checkpointing after every model."""

    models = discover_models(apex_root)
    contract = build_apex_state_contract(models, apex_root, groups, batch_size=batch_size)
    _load_apex_utils(apex_root)
    encoded = encode_apex_sequences(sequences)
    sequence_hash = hashlib.sha256("\n".join(sequences).encode()).hexdigest()
    shape = (len(sequences), len(APEX_STRAINS))
    start = 0
    mic_sum = np.zeros(shape, dtype=np.float64)
    log_sum = np.zeros(shape, dtype=np.float64)
    log_square_sum = np.zeros(shape, dtype=np.float64)
    if state_path.exists():
        with np.load(state_path, allow_pickle=False) as state:
            start, mic_sum, log_sum, log_square_sum = validate_apex_state(
                state,
                sequence_hash=sequence_hash,
                contract=contract,
                shape=shape,
            )

    for model_index, model_path in enumerate(models[start:], start=start):
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model.to(device).eval()
        for offset in range(0, len(sequences), batch_size):
            batch = torch.as_tensor(
                encoded[offset : offset + batch_size], dtype=torch.long, device=device
            )
            with torch.inference_mode():
                model_output = model(batch).detach().cpu().numpy().astype(np.float64)
            log_mic = 6.0 - model_output
            update_ensemble_statistics(
                mic_sum[offset : offset + len(batch)],
                log_sum[offset : offset + len(batch)],
                log_square_sum[offset : offset + len(batch)],
                log_mic,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _atomic_save_npz(
            state_path,
            schema_version=np.asarray(APEX_STATE_SCHEMA_VERSION),
            completed_models=np.asarray(model_index + 1),
            sequence_hash=np.asarray(sequence_hash),
            contract_json=np.asarray(json.dumps(contract, sort_keys=True, separators=(",", ":"))),
            mic_sum=mic_sum,
            log_sum=log_sum,
            log_square_sum=log_square_sum,
        )
        print(f"Completed APEX model {model_index + 1}/{len(models)}", flush=True)
    return finalize_scores(mic_sum, log_sum, log_square_sum, len(models), groups), models


def build_summary(scores: pd.DataFrame) -> dict[str, Any]:
    uncertainty = scores["apex_median_log10_mic_sd"]
    activity = scores["apex_pathogen_median_log10_mic"]
    return {
        "n_sequences": int(len(scores)),
        "n_strains": len(APEX_STRAINS),
        "n_ensemble_models": 40,
        "all_mic_finite_positive": bool(
            np.isfinite(scores.filter(like="apex_mic__").to_numpy()).all()
            and scores.filter(like="apex_mic__").gt(0).all().all()
        ),
        "uncertainty_quantiles": {
            str(key): float(value)
            for key, value in uncertainty.quantile([0.5, 0.9, 0.95, 0.99]).items()
        },
        "pathogen_activity_quantiles": {
            str(key): float(value) for key, value in activity.quantile([0.1, 0.5, 0.9]).items()
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--apex-root", type=Path, default=Path("external/apex"))
    parser.add_argument(
        "--strain-groups", type=Path, default=Path("configs/apex_strain_groups.yaml")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=3000)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    frame = annotate_apex_eligibility(pd.read_csv(args.input))
    eligible = frame.loc[frame["apex_eligible"]].copy()
    if eligible.empty:
        raise ValueError("Input contains no APEX-eligible sequences")
    groups = load_strain_groups(args.strain_groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    scores, models = score_apex_ensemble(
        eligible["sequence"].tolist(),
        args.apex_root,
        groups,
        args.state,
        device=device,
        batch_size=args.batch_size,
    )
    score_columns = scores.columns.tolist()
    combined = frame.copy()
    combined[score_columns] = np.nan
    combined.loc[eligible.index, score_columns] = scores.to_numpy()
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    combined.to_csv(temporary_output, index=False)
    os.replace(temporary_output, args.output)
    summary = build_summary(scores)
    summary["n_input_sequences"] = int(len(frame))
    summary["n_scored_sequences"] = int(len(eligible))
    summary["n_excluded_sequences"] = int((~frame["apex_eligible"]).sum())
    summary["exclusion_reasons"] = {
        str(key): int(value)
        for key, value in frame.loc[~frame["apex_eligible"], "apex_exclusion_reason"]
        .value_counts()
        .items()
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "apex_commit": _apex_commit(args.apex_root),
        "apex_license": "non-profit research only",
        "batch_size": args.batch_size,
        "device": str(device),
        "group_config": str(args.strain_groups.resolve()),
        "group_config_sha256": sha256(args.strain_groups),
        "group_version": groups["version"],
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "model_sha256": {path.name: sha256(path) for path in models},
        "numpy": np.__version__,
        "output": str(args.output.resolve()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
