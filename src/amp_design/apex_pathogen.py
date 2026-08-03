"""Reproducible, resumable scoring with the 8-model APEX-pathogen ensemble."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .apex import (
    annotate_apex_eligibility,
    encode_apex_sequences,
    sha256,
    strain_slug,
    update_ensemble_statistics,
    validate_apex_state,
    validate_sequences,
)

APEX_PATHOGEN_STRAINS = (
    "A. baumannii ATCC 19606",
    "E. coli ATCC 11775",
    "E. coli AIC221",
    "E. coli AIC222",
    "K. pneumoniae ATCC 13883",
    "P. aeruginosa PA01",
    "P. aeruginosa PA14",
    "S. aureus ATCC 12600",
    "S. aureus (ATCC BAA-1556) - MRSA",
    "vancomycin-resistant E. faecalis ATCC 700802",
    "vancomycin-resistant E. faecium ATCC 700221",
)
EXPECTED_MODEL_COUNT = 8


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def discover_pathogen_models(apex_pathogen_root: Path) -> list[Path]:
    """Return the frozen APEX-pathogen checkpoints in deterministic name order."""

    model_dir = apex_pathogen_root / "APEX_pathogen_models"
    models = sorted(path for path in model_dir.glob("APEX_*") if path.is_file())
    if len(models) != EXPECTED_MODEL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MODEL_COUNT} APEX-pathogen checkpoints in {model_dir}, "
            f"found {len(models)}"
        )
    return models


def _load_model_module(apex_pathogen_root: Path) -> Any:
    """Load the checkpoint class under the module name used by the pickles."""

    source = (apex_pathogen_root / "APEX_models.py").resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing APEX-pathogen model definition: {source}")
    existing = sys.modules.get("APEX_models")
    if existing is not None and Path(existing.__file__).resolve() == source:
        return existing
    spec = importlib.util.spec_from_file_location("APEX_models", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load APEX-pathogen model definition: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["APEX_models"] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _repository_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _state_contract(
    models: Sequence[Path],
    apex_pathogen_root: Path,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "backend": "apex-pathogen",
        "repository_commit": _repository_commit(apex_pathogen_root),
        "model_code_sha256": sha256(apex_pathogen_root / "APEX_models.py"),
        "model_paths": [str(path.resolve()) for path in models],
        "model_sha256": [sha256(path) for path in models],
        "strain_order": list(APEX_PATHOGEN_STRAINS),
        "inference_code_sha256": sha256(Path(__file__)),
        "batch_size": batch_size,
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def finalize_pathogen_scores(
    mic_sum: np.ndarray,
    log_sum: np.ndarray,
    log_square_sum: np.ndarray,
    count: int,
) -> pd.DataFrame:
    """Create per-pathogen MICs, disagreement, and the aggregate activity score."""

    expected_shape = (mic_sum.shape[0], len(APEX_PATHOGEN_STRAINS))
    if mic_sum.shape != expected_shape or log_sum.shape != expected_shape:
        raise ValueError(f"APEX-pathogen statistic arrays must have shape {expected_shape}")
    if log_square_sum.shape != expected_shape:
        raise ValueError(f"APEX-pathogen statistic arrays must have shape {expected_shape}")
    if count < 2:
        raise ValueError("At least two ensemble members are required")
    mean_mic = mic_sum / count
    if not np.isfinite(mean_mic).all() or not np.greater(mean_mic, 0).all():
        raise ValueError("APEX-pathogen ensemble produced non-finite or non-positive MIC values")
    variance = np.maximum((log_square_sum - np.square(log_sum) / count) / (count - 1), 0.0)
    log_sd = np.sqrt(variance)
    values: dict[str, np.ndarray] = {}
    for index, strain in enumerate(APEX_PATHOGEN_STRAINS):
        slug = strain_slug(strain)
        values[f"apex_pathogen_mic__{slug}"] = mean_mic[:, index]
        values[f"apex_pathogen_log10_mic_sd__{slug}"] = log_sd[:, index]
    values["apex_pathogen_median_log10_mic"] = np.median(np.log10(mean_mic), axis=1)
    median_sd = np.median(log_sd, axis=1)
    max_sd = np.max(log_sd, axis=1)
    values["apex_pathogen_median_log10_mic_sd"] = median_sd
    values["apex_pathogen_max_log10_mic_sd"] = max_sd
    # Backend-neutral aliases retained by the release feasibility and hard-filter contracts.
    values["apex_median_log10_mic_sd"] = median_sd
    values["apex_max_log10_mic_sd"] = max_sd
    return pd.DataFrame(values)


def score_apex_pathogen_ensemble(
    sequences: Sequence[str],
    apex_pathogen_root: Path,
    state_path: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, list[Path]]:
    """Score canonical peptides with all eight APEX-pathogen models."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    normalized = validate_sequences(pd.DataFrame({"sequence": list(sequences)}))
    sequences = normalized["sequence"].tolist()
    if not sequences:
        raise ValueError("At least one APEX-pathogen sequence is required")
    models = discover_pathogen_models(apex_pathogen_root)
    contract = _state_contract(
        models,
        apex_pathogen_root,
        batch_size=batch_size,
        device=device,
    )
    _load_model_module(apex_pathogen_root)
    encoded = encode_apex_sequences(sequences)
    sequence_hash = hashlib.sha256("\n".join(sequences).encode()).hexdigest()
    shape = (len(sequences), len(APEX_PATHOGEN_STRAINS))
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

    state_path.parent.mkdir(parents=True, exist_ok=True)
    for model_index, model_path in enumerate(models[start:], start=start):
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model.to(device).eval()
        for offset in range(0, len(sequences), batch_size):
            batch = torch.as_tensor(
                encoded[offset : offset + batch_size], dtype=torch.long, device=device
            )
            with torch.inference_mode():
                model_output = model(batch).detach().cpu().numpy().astype(np.float64)
            if model_output.shape != (len(batch), len(APEX_PATHOGEN_STRAINS)):
                raise ValueError(
                    f"Unexpected APEX-pathogen output shape {model_output.shape} from {model_path}"
                )
            if not np.isfinite(model_output).all():
                raise ValueError(f"Non-finite APEX-pathogen model output from {model_path}")
            log_mic = 6.0 - model_output
            with np.errstate(over="ignore", invalid="ignore"):
                mic = np.power(10.0, log_mic, dtype=np.float64)
            if not np.isfinite(mic).all() or not np.greater(mic, 0).all():
                raise ValueError(f"Invalid APEX-pathogen MIC transformation from {model_path}")
            update_ensemble_statistics(
                mic_sum[offset : offset + len(batch)],
                log_sum[offset : offset + len(batch)],
                log_square_sum[offset : offset + len(batch)],
                log_mic,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        temporary = state_path.with_suffix(state_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                schema_version=np.asarray(2),
                completed_models=np.asarray(model_index + 1),
                sequence_hash=np.asarray(sequence_hash),
                contract_json=np.asarray(
                    json.dumps(contract, sort_keys=True, separators=(",", ":"))
                ),
                mic_sum=mic_sum,
                log_sum=log_sum,
                log_square_sum=log_square_sum,
            )
        os.replace(temporary, state_path)
        print(f"Completed APEX-pathogen model {model_index + 1}/{len(models)}", flush=True)
    return finalize_pathogen_scores(
        mic_sum, log_sum, log_square_sum, len(models)
    ), models


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device cpu")
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with a sequence column")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument(
        "--apex-pathogen-root", type=Path, default=Path("external/apex-pathogen")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=3000)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = _resolve_device(args.device)
    frame = annotate_apex_eligibility(pd.read_csv(args.input))
    eligible = frame.loc[frame["apex_eligible"]].copy()
    if eligible.empty:
        raise ValueError("Input contains no APEX-pathogen-eligible sequences")
    for path in (args.output, args.summary, args.manifest, args.state):
        path.parent.mkdir(parents=True, exist_ok=True)
    scores, models = score_apex_pathogen_ensemble(
        eligible["sequence"].tolist(),
        args.apex_pathogen_root,
        args.state,
        device=device,
        batch_size=args.batch_size,
    )
    combined = frame.copy()
    combined[scores.columns] = np.nan
    combined.loc[eligible.index, scores.columns] = scores.to_numpy()
    combined["activity_scorer"] = "apex_pathogen"
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    combined.to_csv(temporary_output, index=False)
    os.replace(temporary_output, args.output)

    summary = {
        "n_input_sequences": int(len(frame)),
        "n_scored_sequences": int(len(eligible)),
        "n_excluded_sequences": int((~frame["apex_eligible"]).sum()),
        "n_pathogens": len(APEX_PATHOGEN_STRAINS),
        "n_ensemble_models": len(models),
        "pathogen_activity_quantiles": {
            str(key): float(value)
            for key, value in scores["apex_pathogen_median_log10_mic"]
            .quantile([0.1, 0.5, 0.9])
            .items()
        },
        "uncertainty_quantiles": {
            str(key): float(value)
            for key, value in scores["apex_pathogen_median_log10_mic_sd"]
            .quantile([0.5, 0.9, 0.95, 0.99])
            .items()
        },
    }
    _atomic_text(args.summary, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": "apex-pathogen",
        "repository": "https://gitlab.com/machine-biology-group-public/apex-pathogen",
        "repository_commit": _repository_commit(args.apex_pathogen_root),
        "license": "MIT",
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "summary": str(args.summary.resolve()),
        "summary_sha256": sha256(args.summary),
        "state": str(args.state.resolve()),
        "state_sha256": sha256(args.state),
        "inference_code": str(Path(__file__).resolve()),
        "inference_code_sha256": sha256(Path(__file__)),
        "model_sha256": {path.name: sha256(path) for path in models},
        "batch_size": args.batch_size,
        "device": str(device),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
    }
    _atomic_text(args.manifest, json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
