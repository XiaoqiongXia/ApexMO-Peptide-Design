"""Self-contained, resumable three-objective genetic Pareto optimization runtime."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pymoo
import torch
from pymoo.indicators.hv import HV

from amp_design import __version__
from amp_design.optimization.genetic import (
    AMINO_ACIDS,
    MAX_LENGTH,
    MIN_LENGTH,
    allowed_initial_edits,
    amino_acid_frequencies,
    assign_rank_and_crowding,
    generate_offspring,
    hash_sorted_fasta_records,
    sequence_sha256,
    survivor_order,
)
from amp_design.predictors.activity import aggregate_activity_targets, score_activity_command
from amp_design.predictors.apex import discover_models, load_strain_groups, score_apex_ensemble
from amp_design.predictors.apex_pathogen import (
    discover_pathogen_models,
    score_apex_pathogen_ensemble,
)
from amp_design.utils.paths import PACKAGE_ROOT
from amp_design.workflows.config import OptimizationConfig

OBJECTIVES = (
    "apex_pathogen_median_log10_mic",
    "toxicity_score",
    "hemolysis_score",
)
PREFLIGHT_ABSOLUTE_TOLERANCES = {
    "apex_pathogen_median_log10_mic": 5e-4,
    # ESM embeddings and downstream probability scores vary at roughly 1e-6
    # across otherwise equivalent GPU batch sizes. This remains far below a
    # meaningful threshold/model change while avoiding false preflight drift.
    "toxicity_score": 1e-5,
    "hemolysis_score": 1e-5,
}
PREFLIGHT_RELATIVE_TOLERANCE = 1e-7
ANCHOR_COLUMNS = (
    "anchor_initial_sequence",
    "anchor_initial_sequence_sha256",
    "anchor_source",
    "initial_edit_distance",
    "normalized_initial_edit_distance",
    "max_allowed_initial_edits",
    "initial_edit_distance_pass",
)


def _objective_columns(config: OptimizationConfig) -> tuple[str, ...]:
    return tuple(getattr(config, "objective_columns", OBJECTIVES))


def _objective_directions(config: OptimizationConfig) -> tuple[str, ...]:
    return tuple(
        getattr(config, "objective_directions", ("minimize",) * len(_objective_columns(config)))
    )


def _objective_tolerance(column: str, config: OptimizationConfig) -> float:
    if column == getattr(config, "effective_activity_output_column", OBJECTIVES[0]):
        return 5e-4
    return PREFLIGHT_ABSOLUTE_TOLERANCES.get(column, 1e-8)


def _initialize_anchor_metadata(
    frame: pd.DataFrame,
    config: OptimizationConfig,
) -> pd.DataFrame:
    """Attach zero-distance self anchors to the initial population."""

    if not config.enforce_initial_edit_distance:
        return frame
    result = frame.copy()
    result["anchor_initial_sequence"] = result["sequence"].astype(str)
    result["anchor_initial_sequence_sha256"] = result["sequence_sha256"].astype(str)
    result["anchor_source"] = "initial_self"
    result["initial_edit_distance"] = 0
    result["normalized_initial_edit_distance"] = 0.0
    result["max_allowed_initial_edits"] = result["sequence"].map(
        lambda sequence: allowed_initial_edits(
            str(sequence),
            max_initial_edit_fraction=config.max_initial_edit_fraction,
            max_initial_edits=config.max_initial_edits,
        )
    )
    result["initial_edit_distance_pass"] = True
    return result


def _merge_child_anchor_metadata(
    child_scores: pd.DataFrame,
    lineage: list[dict[str, object]],
    config: OptimizationConfig,
) -> pd.DataFrame:
    """Join accepted child anchor annotations produced before expensive scoring."""

    if not config.enforce_initial_edit_distance:
        return child_scores
    annotations = pd.DataFrame(lineage)[["sequence", *ANCHOR_COLUMNS]]
    if annotations["sequence"].duplicated().any():
        raise RuntimeError("Accepted offspring lineage contains duplicate sequences")
    result = child_scores.merge(
        annotations,
        on="sequence",
        how="left",
        validate="one_to_one",
    )
    if result[list(ANCHOR_COLUMNS)].isna().any(axis=None):
        raise RuntimeError("Accepted offspring lack complete initial-anchor metadata")
    if not result["initial_edit_distance_pass"].astype(bool).all():
        raise RuntimeError("An offspring exceeded its initial edit-distance budget")
    return result


def _aggregate_configured_activity(frame: pd.DataFrame, config: OptimizationConfig) -> pd.DataFrame:
    if config.activity_scorer not in {"apex", "apex_pathogen"}:
        return frame
    # ObjectiveEvaluator is intentionally initialized with an empty cache when
    # scoring a fresh generated pool. Target columns only exist after model
    # inference, so aggregation must be a no-op for that empty bootstrap frame.
    if frame.empty:
        return frame.copy()
    return aggregate_activity_targets(
        frame,
        backend=config.activity_scorer,
        targets=config.activity_targets,
        aggregation=config.activity_aggregation,
        quantile=config.activity_quantile,
        output_column=config.effective_activity_output_column,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    """Serialize cross-process initialization of shared optimization artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def read_table(path: Path, *, low_memory: bool = False) -> pd.DataFrame:
    """Read a frozen CSV or Parquet table without format guessing."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=low_memory)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format for {path}; expected CSV or Parquet")


def _validated_candidate_identities(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Normalize sequence identity fields and reject ambiguous or stale records."""

    if "sequence" not in frame:
        raise ValueError(f"{label} lacks sequence column")
    result = frame.copy()
    result["sequence"] = result["sequence"].astype(str).str.strip().str.upper()
    if result["sequence"].eq("").any():
        raise ValueError(f"{label} contains empty sequences")
    if result["sequence"].duplicated().any():
        raise ValueError(f"{label} contains duplicate normalized sequences")
    derived_hash = result["sequence"].map(sequence_sha256)
    if (
        "sequence_sha256" in result
        and not result["sequence_sha256"].astype(str).eq(derived_hash).all()
    ):
        raise ValueError(f"{label} contains sequence SHA-256 mismatches")
    derived_length = result["sequence"].str.len()
    if "length" in result:
        observed_length = pd.to_numeric(result["length"], errors="coerce")
        if not observed_length.eq(derived_length).all():
            raise ValueError(f"{label} contains sequence length mismatches")
    result["sequence_sha256"] = derived_hash
    result["length"] = derived_length
    return result


def _device(requested: str) -> torch.device:
    value = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if value == "auto":
        value = "cpu"
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _configure_deterministic_inference() -> None:
    """Use deterministic inference settings where supported by the runtime."""

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _runtime_versions() -> dict[str, object]:
    import sklearn
    import transformers

    return {
        "amp-design": __version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pymoo": pymoo.__version__,
        "scikit-learn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": transformers.__version__,
    }


def _required_esm_files(root: Path) -> list[Path]:
    names = (
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
    )
    files = [root / name for name in names if (root / name).is_file()]
    if not any(path.name in {"model.safetensors", "pytorch_model.bin"} for path in files):
        raise FileNotFoundError(f"No ESM weight file found under {root}")
    return files


def _mean5_cosine_distance(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return the mean cosine distance to five reference neighbours."""

    if len(reference) < 5:
        raise ValueError("Safety AD reference must contain at least five embeddings")
    query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    ref_norm = reference / (np.linalg.norm(reference, axis=1, keepdims=True) + 1e-12)
    output = []
    for start in range(0, len(query_norm), 512):
        similarity = query_norm[start : start + 512] @ ref_norm.T
        nearest = np.partition(similarity, -5, axis=1)[:, -5:]
        output.append(1.0 - nearest.mean(axis=1))
    return np.concatenate(output) if output else np.asarray([], dtype=float)


def _write_sequence_fasta(sequences: list[str], path: Path, prefix: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, sequence in enumerate(sequences):
            handle.write(f">{prefix}{index:08d}\n{sequence}\n")


def _novelty_annotations(
    sequences: list[str],
    training_sequences: list[str],
    config: OptimizationConfig,
    state_dir: Path,
) -> pd.DataFrame:
    """Search candidates against the frozen training set with MMseqs2."""

    columns = [
        "has_ge_50_identity_match",
        "max_train_identity",
        "nearest_train_sequence_id",
        "novelty_pass",
    ]
    if not sequences:
        return pd.DataFrame(columns=columns)
    state_dir.mkdir(parents=True, exist_ok=True)
    cache_contract = json.dumps(
        {
            "query_sequences": sequences,
            "training_sequences_sha256": sequence_sha256("|".join(training_sequences)),
            "novelty_max_identity": config.novelty_max_identity,
            "novelty_min_coverage": config.novelty_min_coverage,
            "mmseqs_command": list(config.mmseqs_command),
        },
        sort_keys=True,
    )
    digest = sequence_sha256(cache_contract)
    cached = state_dir / f"novelty_{digest}.parquet"
    if cached.is_file():
        result = pd.read_parquet(cached)
        if result["sequence"].tolist() != sequences:
            raise RuntimeError(f"Novelty cache sequence mismatch: {cached}")
        return result
    with tempfile.TemporaryDirectory(prefix="amp_novelty_") as temporary:
        work = Path(temporary)
        query, target = work / "query.fasta", work / "training.fasta"
        output, mmseqs_tmp = work / "hits.tsv", work / "tmp"
        _write_sequence_fasta(sequences, query, "query_")
        _write_sequence_fasta(training_sequences, target, "train_")
        command = [
            *config.mmseqs_command,
            "easy-search",
            str(query),
            str(target),
            str(output),
            str(mmseqs_tmp),
            "--min-seq-id",
            str(config.novelty_max_identity),
            "-c",
            str(config.novelty_min_coverage),
            "--cov-mode",
            "0",
            "--mask",
            "0",
            "--comp-bias-corr",
            "0",
            "--min-ungapped-score",
            "0",
            "--max-seqs",
            "20000",
            "-k",
            "5",
            "--spaced-kmer-mode",
            "0",
            "-e",
            "1e6",
            "-s",
            "7.5",
            "--threads",
            str(config.mmseqs_threads),
            "--format-output",
            "query,target,fident,qcov,tcov,alnlen",
            "--remove-tmp-files",
            "1",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode:
            raise RuntimeError(f"MMseqs novelty search failed: {completed.stderr[-2000:]}")
        hits = pd.DataFrame()
        if output.is_file() and output.stat().st_size:
            hits = pd.read_csv(
                output,
                sep="\t",
                names=["query", "target", "identity", "qcov", "tcov", "alnlen"],
            )
            hits["query_index"] = hits["query"].str.removeprefix("query_").astype(int)
            for column in ("identity", "qcov", "tcov"):
                hits[column] = pd.to_numeric(hits[column], errors="raise")
                hits.loc[hits[column] > 1, column] /= 100.0
            hits = hits.sort_values(
                ["query_index", "identity", "qcov", "tcov"],
                ascending=[True, False, False, False],
            ).drop_duplicates("query_index")
        best = hits.set_index("query_index") if not hits.empty else pd.DataFrame()
        rows = []
        for index, sequence in enumerate(sequences):
            matched = not best.empty and index in best.index
            row = best.loc[index] if matched else None
            rows.append(
                {
                    "sequence": sequence,
                    "has_ge_50_identity_match": matched,
                    "max_train_identity": float(row["identity"]) if matched else np.nan,
                    "nearest_train_sequence_id": str(row["target"]) if matched else pd.NA,
                    "novelty_pass": not matched,
                }
            )
    result = pd.DataFrame(rows)
    _atomic_parquet(result, cached)
    return result


def frozen_hashes(config: OptimizationConfig) -> dict[str, str]:
    """Hash every input and weight that can change optimization results."""

    if config.activity_scorer == "apex":
        activity_paths = [
            (PACKAGE_ROOT / "predictors/apex.py").resolve(),
            config.apex_root / "AMP_DL_model_twohead.py",
            config.apex_root / "best_key_list",
            *discover_models(config.apex_root),
        ]
    elif config.activity_scorer == "apex_pathogen":
        activity_paths = [
            (PACKAGE_ROOT / "predictors/apex_pathogen.py").resolve(),
            config.apex_root / "APEX_models.py",
            *discover_pathogen_models(config.apex_root),
        ]
    else:
        activity_paths = [
            (PACKAGE_ROOT / "predictors/activity.py").resolve(),
            *config.activity_model_files,
        ]
    paths = [
        Path(__file__).resolve(),
        (PACKAGE_ROOT / "predictors/activity.py").resolve(),
        (PACKAGE_ROOT / "optimization/genetic.py").resolve(),
        (PACKAGE_ROOT / "workflows/config.py").resolve(),
        config.ranked_candidates,
        config.initial_population,
        config.training_sequences,
        config.promotion_gate,
        config.stability_model,
        config.safety_ad_root / "safety_ad_config.json",
        config.safety_ad_root / "toxicity_esm_reference.npz",
        config.safety_ad_root / "hemolysis_esm_reference.npz",
        *config.novelty_references,
        config.scorer_root / "toxicity_model.pkl",
        config.scorer_root / "hemolysis_model.pkl",
        *activity_paths,
        *_required_esm_files(config.esm_model),
    ]
    if config.activity_scorer == "apex":
        if config.strain_groups is None:
            raise ValueError("strain_groups is required for the legacy APEX backend")
        paths.append(config.strain_groups)
    elif config.activity_scorer == "apex_pathogen" and config.activity_thresholds is not None:
        paths.append(config.activity_thresholds)
    if config.activity_scorer == "apex" and config.activity_thresholds is not None:
        paths.append(config.activity_thresholds)
    return {str(path.resolve()): _sha256_file(path) for path in paths}


def _external_runtime_contract(config: OptimizationConfig) -> dict[str, str | None]:
    """Bind non-Python executables and external scorer repositories to preflight."""

    completed = subprocess.run(
        [*config.mmseqs_command, "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Configured MMseqs command failed: {(completed.stderr or completed.stdout)[-1000:]}"
        )
    lines = (completed.stdout or completed.stderr).strip().splitlines()
    mmseqs_version = lines[-1] if lines else "unknown"
    repository_commit: str | None = None
    if config.activity_scorer in {"apex", "apex_pathogen"}:
        commit = subprocess.run(
            ["git", "-C", str(config.apex_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if commit.returncode or not commit.stdout.strip():
            raise RuntimeError(f"Cannot resolve activity repository commit: {config.apex_root}")
        repository_commit = commit.stdout.strip()
    return {
        "mmseqs_version": mmseqs_version,
        "activity_repository_commit": repository_commit,
    }


def _read_complete_cluster_assignments(
    cluster_file: Path,
    expected_members: set[str],
) -> dict[str, str]:
    """Parse MMseqs clusters and require exactly one assignment per input sequence."""

    assignments: dict[str, str] = {}
    for line_number, line in enumerate(cluster_file.read_text().splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) < 2:
            raise ValueError(f"Malformed MMseqs cluster row {line_number}: {line!r}")
        representative, member = fields[:2]
        if member in assignments:
            raise ValueError(f"Duplicate MMseqs member assignment: {member}")
        assignments[member] = representative
    observed = set(assignments)
    missing = expected_members.difference(observed)
    extra = observed.difference(expected_members)
    invalid_representatives = set(assignments.values()).difference(expected_members)
    if missing or extra or invalid_representatives:
        raise ValueError(
            "MMseqs cluster membership mismatch: "
            f"missing={len(missing)}, extra={len(extra)}, "
            f"invalid_representatives={len(invalid_representatives)}"
        )
    return assignments


def optimization_contract(config: OptimizationConfig) -> dict[str, object]:
    """Return result-affecting settings that must remain fixed after preflight."""

    return {
        "protocol": config.protocol,
        "activity_scorer": config.activity_scorer,
        "activity_targets": list(getattr(config, "activity_targets", ())),
        "activity_aggregation": getattr(config, "activity_aggregation", "median"),
        "activity_quantile": getattr(config, "activity_quantile", 0.5),
        "activity_output_column": getattr(
            config, "effective_activity_output_column", OBJECTIVES[0]
        ),
        "activity_command": list(getattr(config, "activity_command", ())),
        "activity_model_files": [
            str(path.resolve()) for path in getattr(config, "activity_model_files", ())
        ],
        "seeds": list(config.seeds),
        "generations": config.generations,
        "population_size": config.population_size,
        "batch_size_apex": config.batch_size_apex,
        "batch_size_esm": config.batch_size_esm,
        "device": config.device,
        "mmseqs_command": list(config.mmseqs_command),
        "mmseqs_threads": config.mmseqs_threads,
        "cluster_max_members": config.cluster_max_members,
        "apex_uncertainty_max": config.apex_uncertainty_max,
        "novelty_max_identity": config.novelty_max_identity,
        "novelty_min_coverage": config.novelty_min_coverage,
        "enforce_safety_ad": config.enforce_safety_ad,
        "enforce_apex_uncertainty": config.enforce_apex_uncertainty,
        "enforce_novelty": config.enforce_novelty,
        "enforce_initial_edit_distance": getattr(
            config, "enforce_initial_edit_distance", False
        ),
        "max_initial_edit_fraction": getattr(config, "max_initial_edit_fraction", 0.25),
        "max_initial_edits": getattr(config, "max_initial_edits", 5),
        "same_anchor_crossover": getattr(config, "same_anchor_crossover", True),
        "max_offspring_attempts": getattr(config, "max_offspring_attempts", 20),
        "early_stopping_patience": getattr(config, "early_stopping_patience", 0),
        "early_stopping_min_delta": getattr(config, "early_stopping_min_delta", 0.001),
        "early_stopping_metric": getattr(config, "early_stopping_metric", "hypervolume"),
        "novelty_references": [str(path.resolve()) for path in config.novelty_references],
        "objectives": [
            {"column": column, "direction": direction}
            for column, direction in zip(
                _objective_columns(config), _objective_directions(config), strict=True
            )
        ],
        "minimum_length": MIN_LENGTH,
        "maximum_length": MAX_LENGTH,
    }


class ObjectiveEvaluator:
    """Cached ESM safety and APEX activity evaluator for novel offspring."""

    def __init__(self, original: pd.DataFrame, config: OptimizationConfig) -> None:
        try:
            from transformers import AutoTokenizer, EsmModel
        except ImportError as error:
            raise RuntimeError(
                "Optimization dependencies are missing; install amp-design[optimization]"
            ) from error

        original = _validated_candidate_identities(original, "Ranked candidates")
        original = _aggregate_configured_activity(original, config)
        required = {"sequence", *_objective_columns(config)}
        if len(original) and (missing := required.difference(original.columns)):
            raise ValueError(f"Ranked candidates lack required columns: {sorted(missing)}")
        if len(original) and config.activity_scorer in {"apex", "apex_pathogen"}:
            if "activity_scorer" not in original:
                raise ValueError(
                    f"{config.activity_scorer} optimization refuses candidate scores without "
                    "activity_scorer provenance; rescore the candidate pool first"
                )
            backends = set(original["activity_scorer"].dropna().astype(str))
            if backends != {config.activity_scorer}:
                raise ValueError(
                    f"{config.activity_scorer} optimization received incompatible scorer "
                    "provenance: "
                    f"{sorted(backends)}"
                )
        self.config = config
        self.device = _device(config.device)
        _configure_deterministic_inference()
        self.groups = (
            load_strain_groups(config.strain_groups)  # type: ignore[arg-type]
            if config.activity_scorer == "apex"
            else None
        )
        gate = json.loads(config.promotion_gate.read_text(encoding="utf-8"))
        self.safety_thresholds = {
            task: float(gate["tasks"][task]["dehomologized_metrics"]["threshold"])
            for task in ("toxicity", "hemolysis")
        }
        ad_config = json.loads(
            (config.safety_ad_root / "safety_ad_config.json").read_text(encoding="utf-8")
        )
        self.ad_thresholds = {
            task: float(ad_config["esm_5nn_thresholds"][task]) for task in ("toxicity", "hemolysis")
        }
        self.ad_length_ranges = {
            task: tuple(int(value) for value in ad_config["length_ranges"][task])
            for task in ("toxicity", "hemolysis")
        }
        self.ad_references = {
            task: np.load(config.safety_ad_root / f"{task}_esm_reference.npz")["X"]
            for task in ("toxicity", "hemolysis")
        }
        reference_tables = [
            read_table(config.training_sequences),
            *[read_table(path) for path in config.novelty_references],
        ]
        training = (
            pd.concat([table[["sequence"]] for table in reference_tables], ignore_index=True)[
                "sequence"
            ]
            .astype(str)
            .str.strip()
            .str.upper()
        )
        self.training_sequences = list(dict.fromkeys(training))
        self.forbidden_sequences = set(self.training_sequences)
        self.cache: dict[str, dict[str, object]] = {}
        for row in original.to_dict("records"):
            sequence = str(row["sequence"])
            self.cache[sequence] = row
            self.forbidden_sequences.add(sequence)
        self.tokenizer = AutoTokenizer.from_pretrained(config.esm_model, local_files_only=True)
        self.encoder = (
            EsmModel.from_pretrained(config.esm_model, local_files_only=True).to(self.device).eval()
        )
        self.models = {}
        for task in ("toxicity", "hemolysis"):
            with (config.scorer_root / f"{task}_model.pkl").open("rb") as handle:
                self.models[task] = pickle.load(handle)
        with config.stability_model.open("rb") as handle:
            self.stability_model = pickle.load(handle)

    def embed(self, sequences: list[str]) -> np.ndarray:
        embeddings = []
        with torch.inference_mode():
            for start in range(0, len(sequences), self.config.batch_size_esm):
                values = sequences[start : start + self.config.batch_size_esm]
                encoded = {
                    key: value.to(self.device)
                    for key, value in self.tokenizer(
                        values, return_tensors="pt", padding=True
                    ).items()
                }
                hidden = self.encoder(**encoded).last_hidden_state
                mask = encoded["attention_mask"].bool()
                mask[:, 0] = False
                for index, sequence in enumerate(values):
                    mask[index, len(sequence) + 1] = False
                    embeddings.append(hidden[index][mask[index]].mean(0).cpu().numpy())
        return np.asarray(embeddings, dtype=np.float32)

    def evaluate(
        self,
        sequences: list[str],
        state_dir: Path,
        *,
        force: bool = False,
    ) -> pd.DataFrame:
        normalized = [str(value).strip().upper() for value in sequences]
        unique = list(dict.fromkeys(normalized))
        if force:
            for value in unique:
                self.cache.pop(value, None)
        missing = [value for value in unique if value not in self.cache]
        if missing:
            embeddings = self.embed(missing)
            features = np.column_stack(
                [embeddings, np.asarray([len(value) for value in missing], dtype=float)]
            )
            output = pd.DataFrame(
                {
                    "sequence": missing,
                    "sequence_sha256": [sequence_sha256(value) for value in missing],
                    "length": [len(value) for value in missing],
                }
            )
            for task in ("toxicity", "hemolysis"):
                output[f"{task}_score"] = self.models[task].predict_proba(features)[:, 1]
                output[f"{task}_esm_5nn_distance"] = _mean5_cosine_distance(
                    embeddings, self.ad_references[task]
                )
                output[f"{task}_esm_ad_pass"] = (
                    output[f"{task}_esm_5nn_distance"] <= self.ad_thresholds[task]
                )
                minimum, maximum = self.ad_length_ranges[task]
                output[f"{task}_length_ad_pass"] = output["length"].between(minimum, maximum)
                output[f"{task}_ad_pass"] = (
                    output[f"{task}_esm_ad_pass"] & output[f"{task}_length_ad_pass"]
                )
            output["formal_safety_ad_pass"] = (
                output["toxicity_ad_pass"] & output["hemolysis_ad_pass"]
            )
            output["safety_ad_protocol"] = "portable_esm_length_v1"
            output["safety_gate_pass"] = output["toxicity_score"].lt(
                self.safety_thresholds["toxicity"]
            ) & output["hemolysis_score"].lt(self.safety_thresholds["hemolysis"])
            output["stability_model"] = "custom_esm2_ridge_stage_b_v1"
            output["stability_log10_half_life"] = np.asarray(
                self.stability_model.predict(embeddings), dtype=float
            ).reshape(-1)
            output["stability_half_life_hours"] = np.power(
                10.0, output["stability_log10_half_life"]
            )
            output["stability_prediction"] = np.where(
                output["stability_half_life_hours"].ge(1.0), "Stable", "Unstable"
            )
            state_dir.mkdir(parents=True, exist_ok=True)
            state = state_dir / (
                f"{self.config.activity_scorer}_{sequence_sha256('|'.join(missing))}.npz"
            )
            if self.config.activity_scorer == "apex_pathogen":
                apex, _ = score_apex_pathogen_ensemble(
                    missing,
                    self.config.apex_root,
                    state,
                    device=self.device,
                    batch_size=self.config.batch_size_apex,
                )
            elif self.config.activity_scorer == "apex":
                apex, _ = score_apex_ensemble(
                    missing,
                    self.config.apex_root,
                    self.groups,
                    state,
                    device=self.device,
                    batch_size=self.config.batch_size_apex,
                )
            else:
                apex = score_activity_command(
                    missing,
                    self.config.activity_command,
                    output_column=self.config.effective_activity_output_column,
                    work_dir=state_dir,
                )
            apex = _aggregate_configured_activity(apex, self.config)
            apex = apex.set_index("sequence") if "sequence" in apex else apex
            if isinstance(apex.index, pd.Index) and apex.index.name == "sequence":
                apex = apex.loc[missing].reset_index(drop=True)
            if collisions := (set(apex.columns) - {"sequence"}).intersection(output.columns):
                raise ValueError(
                    f"Activity scorer output attempts to overwrite evaluator columns: "
                    f"{sorted(collisions)}"
                )
            for column in apex.columns:
                if column != "sequence":
                    output[column] = apex[column].to_numpy()
            output["activity_scorer"] = self.config.activity_scorer
            if "apex_median_log10_mic_sd" in output:
                output["apex_uncertainty_feasible"] = output["apex_median_log10_mic_sd"].le(
                    self.config.apex_uncertainty_max
                )
            elif self.config.enforce_apex_uncertainty:
                raise ValueError(
                    "Configured activity scorer must return apex_median_log10_mic_sd "
                    "when enforce_apex_uncertainty is true"
                )
            novelty = _novelty_annotations(
                missing,
                self.training_sequences,
                self.config,
                state_dir,
            ).set_index("sequence")
            for column in novelty.columns:
                output[column] = output["sequence"].map(novelty[column])
            output["formal_genetic_feasible"] = _feasible(output, self.config)
            output["search_protocol"] = self.config.protocol
            for row in output.to_dict("records"):
                self.cache[str(row["sequence"])] = row
        return pd.DataFrame([self.cache[value] for value in normalized])


def _feasible(frame: pd.DataFrame, config: OptimizationConfig) -> np.ndarray:
    values = frame[list(_objective_columns(config))].to_numpy(dtype=float)
    feasible = (
        frame["sequence"].str.fullmatch(f"[{AMINO_ACIDS}]+", na=False).to_numpy()
        & frame["length"].between(MIN_LENGTH, MAX_LENGTH).to_numpy()
        & np.isfinite(values).all(axis=1)
    )
    if config.enforce_safety_ad:
        if "formal_safety_ad_pass" not in frame:
            raise ValueError("Candidate table lacks formal_safety_ad_pass")
        feasible &= frame["formal_safety_ad_pass"].fillna(False).astype(bool).to_numpy()
    if config.enforce_apex_uncertainty:
        if "apex_median_log10_mic_sd" not in frame:
            raise ValueError("Candidate table lacks apex_median_log10_mic_sd")
        feasible &= (
            frame["apex_median_log10_mic_sd"]
            .le(config.apex_uncertainty_max)
            .fillna(False)
            .to_numpy()
        )
    if config.enforce_novelty:
        if "novelty_pass" in frame:
            novelty = frame["novelty_pass"].fillna(False).astype(bool)
        elif "has_ge_50_identity_match" in frame:
            novelty = ~frame["has_ge_50_identity_match"].fillna(True).astype(bool)
        else:
            raise ValueError("Candidate table lacks a training-set novelty annotation")
        feasible &= novelty.to_numpy()
    return feasible


def _cluster_command(config: OptimizationConfig, fasta: Path, prefix: Path, tmp: Path) -> list[str]:
    return [
        *config.mmseqs_command,
        "easy-cluster",
        str(fasta),
        str(prefix),
        str(tmp),
        "--min-seq-id",
        "0.8",
        "-c",
        "0.8",
        "--cov-mode",
        "0",
        "--cluster-mode",
        "1",
        "--mask",
        "0",
        "--comp-bias-corr",
        "0",
        "--min-ungapped-score",
        "0",
        "--max-seqs",
        "20000",
        "-k",
        "5",
        "--spaced-kmer-mode",
        "0",
        "-e",
        "1e6",
        "-s",
        "7.5",
        "--threads",
        str(config.mmseqs_threads),
        "--remove-tmp-files",
        "1",
    ]


def select_survivors(
    frame: pd.DataFrame,
    config: OptimizationConfig,
    seed: int,
    generation: int,
) -> pd.DataFrame:
    ranked = assign_rank_and_crowding(
        frame,
        _objective_columns(config),
        feasible=_feasible(frame, config),
        directions=_objective_directions(config),
    )
    order = survivor_order(ranked)
    work = Path(tempfile.mkdtemp(prefix=f"amp_mmseqs_{seed}_{generation}_"))
    fasta, prefix, tmp = work / "input.fasta", work / "clusters", work / "tmp"
    with fasta.open("w", encoding="utf-8") as handle:
        for identifier, sequence in hash_sorted_fasta_records(ranked):
            handle.write(f">{identifier}\n{sequence}\n")
    command = _cluster_command(config, fasta, prefix, tmp)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(
                f"MMseqs failed with exit code {result.returncode}: {result.stderr[-2000:]}"
            )
        cluster_file = Path(f"{prefix}_cluster.tsv")
        if not cluster_file.is_file():
            raise RuntimeError(f"MMseqs output is missing: {cluster_file}")
        expected_members = set(ranked["sequence_sha256"].astype(str))
        assignments = _read_complete_cluster_assignments(cluster_file, expected_members)
        seed_output = config.output_root / f"seed_{seed}"
        seed_output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            cluster_file,
            seed_output / f"mmseqs_generation_{generation:03d}_cluster.tsv",
        )
        _atomic_text(
            seed_output / f"mmseqs_generation_{generation:03d}_metadata.json",
            json.dumps(
                {"seed": seed, "generation": generation, "command": command},
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    counts: dict[str, int] = {}
    chosen = []
    for index in order:
        sequence_hash = str(ranked.iloc[index]["sequence_sha256"])
        cluster_id = assignments[sequence_hash]
        if counts.get(cluster_id, 0) >= config.cluster_max_members:
            continue
        counts[cluster_id] = counts.get(cluster_id, 0) + 1
        chosen.append(index)
        if len(chosen) == config.population_size:
            break
    if len(chosen) != config.population_size:
        raise RuntimeError(
            f"Diversity cap retained {len(chosen)} rows; expected {config.population_size}"
        )
    return ranked.iloc[chosen].copy()


def _save_state(
    seed_output: Path,
    generation: int,
    rng: np.random.Generator,
    history: set[str],
) -> None:
    history_path = seed_output / f"history_generation_{generation:03d}.txt"
    _atomic_text(history_path, "\n".join(sorted(history)) + "\n")
    _atomic_text(
        seed_output / f"state_generation_{generation:03d}.json",
        json.dumps(
            {
                "schema_version": 1,
                "generation": generation,
                "rng_state": rng.bit_generator.state,
                "history_file": history_path.name,
                "history_count": len(history),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _load_state(seed_output: Path, generation: int, rng: np.random.Generator) -> set[str]:
    state = json.loads((seed_output / f"state_generation_{generation:03d}.json").read_text())
    if state.get("schema_version") != 1 or int(state["generation"]) != generation:
        raise ValueError("Resume state schema or generation mismatch")
    rng.bit_generator.state = state["rng_state"]
    history = {
        value for value in (seed_output / state["history_file"]).read_text().splitlines() if value
    }
    if len(history) != int(state["history_count"]):
        raise ValueError("Resume history count mismatch")
    return history


def _hypervolume_normalization(
    initial: pd.DataFrame,
    config: OptimizationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed minimization-space lower bounds and scales from generation zero."""

    values = initial[list(_objective_columns(config))].to_numpy(dtype=float)
    signs = np.asarray(
        [
            1.0 if direction == "minimize" else -1.0
            for direction in _objective_directions(config)
        ]
    )
    transformed = values * signs
    lower = np.min(transformed, axis=0)
    scale = np.max(transformed, axis=0) - lower
    scale[scale == 0] = 1.0
    return lower, scale


def _population_hypervolume(
    population: pd.DataFrame,
    config: OptimizationConfig,
    lower: np.ndarray,
    scale: np.ndarray,
) -> float:
    """Compute normalized feasible-population hypervolume against a fixed reference."""

    values = population[list(_objective_columns(config))].to_numpy(dtype=float)
    signs = np.asarray(
        [
            1.0 if direction == "minimize" else -1.0
            for direction in _objective_directions(config)
        ]
    )
    normalized = (values * signs - lower) / scale
    feasible = population["formal_genetic_feasible"].fillna(False).astype(bool).to_numpy()
    if "pareto_rank" in population:
        feasible &= population["pareto_rank"].eq(0).to_numpy()
    reference = np.full(normalized.shape[1], 1.05, dtype=float)
    points = normalized[feasible & np.isfinite(normalized).all(axis=1)]
    points = points[np.all(points < reference, axis=1)]
    return float(HV(ref_point=reference)(points)) if len(points) else 0.0


def _early_stopping_status(
    metrics: list[float],
    min_delta: float,
) -> tuple[float, int]:
    """Return best metric and consecutive generations without material improvement."""

    best = -np.inf
    stale = 0
    for metric in metrics:
        if metric > best + min_delta:
            best = metric
            stale = 0
        else:
            stale += 1
    return float(best), stale


def _write_cache(
    evaluator: ObjectiveEvaluator,
    history: set[str],
    initial_sequences: set[str],
    seed: int,
    path: Path,
) -> None:
    if missing := history.difference(evaluator.cache):
        raise RuntimeError(f"Evaluation cache is missing {len(missing)} historical sequences")
    cache = pd.DataFrame([evaluator.cache[value] for value in sorted(history)])
    cache["genetic_seed"] = seed
    cache["candidate_origin"] = np.where(
        cache["sequence"].isin(initial_sequences), "initial", "evolved"
    )
    _atomic_parquet(cache.drop_duplicates("sequence"), path)


def _preflight(
    config: OptimizationConfig,
    original: pd.DataFrame,
    initial: pd.DataFrame,
    frequencies: np.ndarray,
) -> dict[str, object]:
    if len(initial) != config.population_size:
        raise ValueError(
            f"Initial population has {len(initial)} rows; expected {config.population_size}"
        )
    evaluator = ObjectiveEvaluator(original, config)
    initial_sequences = initial["sequence"].astype(str).tolist()
    initial_scores = evaluator.evaluate(
        initial_sequences,
        config.output_root / "preflight_cache",
        force=True,
    )
    objectives = _objective_columns(config)
    directions = _objective_directions(config)
    differences = {
        objective: float(
            np.max(
                np.abs(
                    initial_scores[objective].to_numpy(float) - initial[objective].to_numpy(float)
                )
            )
        )
        for objective in objectives
    }
    consistent = all(
        np.allclose(
            initial_scores[objective].to_numpy(float),
            initial[objective].to_numpy(float),
            atol=_objective_tolerance(objective, config),
            rtol=PREFLIGHT_RELATIVE_TOLERANCE,
        )
        for objective in objectives
    )
    if not consistent:
        raise RuntimeError(f"Initial objective mismatch: {differences}")
    ranked = assign_rank_and_crowding(
        initial_scores,
        objectives,
        feasible=_feasible(initial_scores, config),
        directions=directions,
    )
    ranked = _initialize_anchor_metadata(ranked, config)
    probes, _ = generate_offspring(
        ranked,
        np.random.default_rng(42),
        frequencies,
        count=2,
        historical_sequences=set(initial_sequences),
        forbidden_sequences=evaluator.forbidden_sequences,
        generation=0,
        seed=42,
        max_attempts=config.max_offspring_attempts,
        enforce_initial_edit_distance=config.enforce_initial_edit_distance,
        max_initial_edit_fraction=config.max_initial_edit_fraction,
        max_initial_edits=config.max_initial_edits,
        same_anchor_crossover=config.same_anchor_crossover,
    )
    probe_scores = evaluator.evaluate(probes, config.output_root / "preflight_cache")
    if not np.isfinite(probe_scores[list(objectives)].to_numpy(float)).all():
        raise RuntimeError("Preflight probe produced non-finite objective values")
    report = {
        "schema_version": 1,
        "passed": True,
        "protocol": config.protocol,
        "optimization_contract": optimization_contract(config),
        "frozen_hashes": frozen_hashes(config),
        "device": str(evaluator.device),
        "deterministic_inference": {
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        },
        "package_versions": _runtime_versions(),
        "external_runtime_contract": _external_runtime_contract(config),
        "initial_rows": len(initial),
        "initial_objective_max_abs_difference": differences,
        "initial_objective_tolerance": {
            "absolute": {
                objective: _objective_tolerance(objective, config) for objective in objectives
            },
            "relative": PREFLIGHT_RELATIVE_TOLERANCE,
        },
        "probe_scores": probe_scores[["sequence", "sequence_sha256", *objectives]].to_dict(
            "records"
        ),
    }
    _atomic_text(
        config.output_root / "preflight_report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    return report


def run_optimization(
    config: OptimizationConfig,
    *,
    preflight_only: bool = False,
    only_seed: int | None = None,
    resume_generation: int | None = None,
) -> None:
    """Run or resume all configured independent genetic searches."""

    if only_seed is not None and only_seed not in config.seeds:
        raise ValueError(f"Requested seed {only_seed} is not in configured seeds {config.seeds}")
    selected_seeds = config.seeds if only_seed is None else (only_seed,)
    resume = config.resume_generation if resume_generation is None else resume_generation
    if not 1 <= resume <= config.generations:
        raise ValueError("resume_generation must lie within [1, generations]")

    config.output_root.mkdir(parents=True, exist_ok=True)
    original = read_table(config.ranked_candidates, low_memory=False)
    initial = read_table(config.initial_population, low_memory=False)
    original = _validated_candidate_identities(original, "Ranked candidates")
    initial = _validated_candidate_identities(initial, "Initial population")
    original = _aggregate_configured_activity(original, config)
    initial = _aggregate_configured_activity(initial, config)
    training = read_table(config.training_sequences, low_memory=False)[["sequence"]]
    frequencies = amino_acid_frequencies(training["sequence"])
    report_path = config.output_root / "preflight_report.json"
    lock_path = config.output_root / ".preflight.lock"
    with _exclusive_file_lock(lock_path):
        if not report_path.exists():
            report = _preflight(config, original, initial, frequencies)
        else:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                not report.get("passed")
                or report.get("protocol") != config.protocol
                or report.get("optimization_contract") != optimization_contract(config)
                or report.get("frozen_hashes") != frozen_hashes(config)
                or report.get("device") != str(_device(config.device))
                or report.get("package_versions") != _runtime_versions()
                or report.get("external_runtime_contract")
                != _external_runtime_contract(config)
            ):
                raise RuntimeError("Existing optimization preflight does not match frozen inputs")
    if preflight_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    for seed in selected_seeds:
        seed_output = config.output_root / f"seed_{seed}"
        for generation in range(resume, config.generations + 1):
            path = seed_output / f"generation_{generation:03d}.csv"
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite completed generation: {path}")

    initial_values = initial["sequence"].astype(str).tolist()
    initial_set = set(initial_values)

    for seed in selected_seeds:
        evaluator = ObjectiveEvaluator(original, config)
        rng = np.random.default_rng(seed)
        seed_output = config.output_root / f"seed_{seed}"
        seed_output.mkdir(parents=True, exist_ok=True)
        score_cache = seed_output / "score_cache"
        initial_scores = evaluator.evaluate(initial_values, score_cache, force=True)
        objectives = _objective_columns(config)
        directions = _objective_directions(config)
        if not all(
            np.allclose(
                initial_scores[objective].to_numpy(float),
                initial[objective].to_numpy(float),
                atol=_objective_tolerance(objective, config),
                rtol=PREFLIGHT_RELATIVE_TOLERANCE,
            )
            for objective in objectives
        ):
            differences = {
                objective: float(
                    np.max(
                        np.abs(
                            initial_scores[objective].to_numpy(float)
                            - initial[objective].to_numpy(float)
                        )
                    )
                )
                for objective in objectives
            }
            raise RuntimeError(f"Seed worker {seed} initial objective mismatch: {differences}")
        initial_population = assign_rank_and_crowding(
            initial_scores,
            objectives,
            feasible=_feasible(initial_scores, config),
            directions=directions,
        )
        initial_population = _initialize_anchor_metadata(initial_population, config)
        hypervolume_lower, hypervolume_scale = _hypervolume_normalization(
            initial_population, config
        )
        _atomic_text(
            seed_output / "worker_runtime.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "seed": seed,
                    "resume_generation": resume,
                    "logical_device": str(evaluator.device),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "pid": os.getpid(),
                    "package_versions": _runtime_versions(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        if resume > 1:
            previous = resume - 1
            population = pd.read_csv(seed_output / f"generation_{previous:03d}.csv")
            if config.enforce_initial_edit_distance:
                missing_anchor_columns = set(ANCHOR_COLUMNS).difference(population.columns)
                if missing_anchor_columns:
                    raise RuntimeError(
                        "Cannot resume edit-distance-constrained optimization because the "
                        f"population lacks anchor columns: {sorted(missing_anchor_columns)}"
                    )
                if not population["initial_edit_distance_pass"].astype(bool).all():
                    raise RuntimeError(
                        "Cannot resume from a population containing edit-distance failures"
                    )
            history = _load_state(seed_output, previous, rng)
            cache_path = seed_output / "evaluation_cache.parquet"
            if not cache_path.is_file():
                raise FileNotFoundError(f"Resume evaluation cache not found: {cache_path}")
            for row in pd.read_parquet(cache_path).to_dict("records"):
                evaluator.cache[str(row["sequence"])] = row
            if missing := history.difference(evaluator.cache):
                raise RuntimeError(f"Resume cache is missing {len(missing)} sequences")
        else:
            population = initial_population.copy()
            history = set(initial_values)

        early_history_path = seed_output / "early_stopping_history.csv"
        if resume > 1:
            previous = resume - 1
            if early_history_path.is_file():
                early_history = pd.read_csv(early_history_path)
                expected = list(range(1, previous + 1))
                if early_history["generation"].astype(int).tolist() != expected:
                    raise RuntimeError("Early-stopping history does not match resume generation")
            else:
                reconstructed = []
                for completed_generation in range(1, previous + 1):
                    completed_population = pd.read_csv(
                        seed_output / f"generation_{completed_generation:03d}.csv"
                    )
                    reconstructed.append(
                        {
                            "generation": completed_generation,
                            "hypervolume": _population_hypervolume(
                                completed_population,
                                config,
                                hypervolume_lower,
                                hypervolume_scale,
                            ),
                        }
                    )
                early_history = pd.DataFrame(reconstructed)
                _atomic_csv(early_history, early_history_path)
        else:
            early_history = pd.DataFrame(columns=["generation", "hypervolume"])

        existing_metrics = early_history.get("hypervolume", pd.Series(dtype=float)).tolist()
        best_hypervolume, stale_generations = _early_stopping_status(
            [float(value) for value in existing_metrics],
            config.early_stopping_min_delta,
        )
        completion_path = seed_output / "optimization_complete.json"
        if (
            config.early_stopping_patience > 0
            and stale_generations >= config.early_stopping_patience
        ):
            _atomic_text(
                completion_path,
                json.dumps(
                    {
                        "schema_version": 1,
                        "seed": seed,
                        "reason": "early_stopping",
                        "last_generation": resume - 1,
                        "best_hypervolume": best_hypervolume,
                        "stale_generations": stale_generations,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
            continue

        for generation in range(resume, config.generations + 1):
            children, lineage = generate_offspring(
                population,
                rng,
                frequencies,
                count=config.population_size,
                historical_sequences=history,
                forbidden_sequences=evaluator.forbidden_sequences,
                generation=generation,
                seed=seed,
                max_attempts=config.max_offspring_attempts,
                enforce_initial_edit_distance=config.enforce_initial_edit_distance,
                max_initial_edit_fraction=config.max_initial_edit_fraction,
                max_initial_edits=config.max_initial_edits,
                same_anchor_crossover=config.same_anchor_crossover,
            )
            child_scores = evaluator.evaluate(children, score_cache)
            child_scores = _merge_child_anchor_metadata(child_scores, lineage, config)
            combined = pd.concat([population, child_scores], ignore_index=True).drop_duplicates(
                "sequence", keep="first"
            )
            population = select_survivors(combined, config, seed, generation)
            _atomic_csv(pd.DataFrame(lineage), seed_output / f"lineage_{generation:03d}.csv")
            _save_state(seed_output, generation, rng, history)
            _write_cache(
                evaluator,
                history,
                initial_set,
                seed,
                seed_output / "evaluation_cache.parquet",
            )
            # The population CSV is the generation-complete marker. Write it
            # only after every resume artifact above has reached durable paths.
            _atomic_csv(population, seed_output / f"generation_{generation:03d}.csv")
            hypervolume = _population_hypervolume(
                population,
                config,
                hypervolume_lower,
                hypervolume_scale,
            )
            early_history = pd.concat(
                [
                    early_history,
                    pd.DataFrame(
                        [{"generation": generation, "hypervolume": hypervolume}]
                    ),
                ],
                ignore_index=True,
            )
            _atomic_csv(early_history, early_history_path)
            best_hypervolume, stale_generations = _early_stopping_status(
                early_history["hypervolume"].astype(float).tolist(),
                config.early_stopping_min_delta,
            )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "generation": generation,
                        "population": len(population),
                        "rank0": int((population["pareto_rank"] == 0).sum()),
                        "hypervolume": hypervolume,
                        "stale_generations": stale_generations,
                    }
                ),
                flush=True,
            )
            if (
                config.early_stopping_patience > 0
                and stale_generations >= config.early_stopping_patience
            ):
                _atomic_text(
                    completion_path,
                    json.dumps(
                        {
                            "schema_version": 1,
                            "seed": seed,
                            "reason": "early_stopping",
                            "last_generation": generation,
                            "best_hypervolume": best_hypervolume,
                            "stale_generations": stale_generations,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                )
                break
        else:
            _atomic_text(
                completion_path,
                json.dumps(
                    {
                        "schema_version": 1,
                        "seed": seed,
                        "reason": "max_generations",
                        "last_generation": config.generations,
                        "best_hypervolume": best_hypervolume,
                        "stale_generations": stale_generations,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
