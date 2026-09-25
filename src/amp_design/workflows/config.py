"""Configuration parsing and validation for sampling and optimization."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


def _strict_section(payload: dict[str, Any], name: str, cls: type) -> dict[str, Any]:
    section = payload.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Configuration must contain a mapping named {name!r}")
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(section).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown {name} configuration keys: {unknown}")
    return section


def _path(value: str | Path) -> Path:
    raw = str(value)
    expanded = os.path.expandvars(raw)
    if "$" in expanded:
        raise ValueError(f"Path contains an unresolved environment variable: {raw}")
    return Path(expanded).expanduser().resolve()


@dataclass(frozen=True)
class ObjectiveSpec:
    """One optimizer objective and whether it is minimized or maximized."""

    column: str
    direction: str = "minimize"

    @classmethod
    def from_value(cls, value: str | dict[str, Any]) -> ObjectiveSpec:
        if isinstance(value, str):
            return cls(column=value)
        if not isinstance(value, dict):
            raise ValueError("Each objective must be a column name or a mapping")
        unknown = set(value).difference({"column", "direction"})
        if unknown:
            raise ValueError(f"Unknown objective keys: {sorted(unknown)}")
        if "column" not in value:
            raise ValueError("Objective mappings require a 'column' key")
        return cls(column=str(value["column"]), direction=str(value.get("direction", "minimize")))

    def validate(self) -> None:
        if not self.column or not self.column.strip():
            raise ValueError("Objective column names must be non-empty")
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("Objective direction must be 'minimize' or 'maximize'")


@dataclass(frozen=True)
class SamplingConfig:
    checkpoint: Path
    output_root: Path
    training_seed: int
    lengths: tuple[int, ...]
    samples_per_length: int = 1000
    shard_size: int = 1000
    steps: int = 256
    batch_size: int = 512
    device: str = "auto"

    @classmethod
    def from_file(cls, path: Path) -> SamplingConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values = _strict_section(payload, "sampling", cls)
        return cls.from_mapping(values)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> SamplingConfig:
        values = dict(values)
        allowed = {field.name for field in fields(cls)}
        if unknown := sorted(set(values).difference(allowed)):
            raise ValueError(f"Unknown sampling configuration keys: {unknown}")
        values["checkpoint"] = _path(values["checkpoint"])
        values["output_root"] = _path(values["output_root"])
        values["lengths"] = tuple(int(value) for value in values["lengths"])
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Sampling checkpoint not found: {self.checkpoint}")
        if not self.lengths or len(set(self.lengths)) != len(self.lengths):
            raise ValueError("Sampling lengths must be non-empty and unique")
        if any(not 5 <= value <= 64 for value in self.lengths):
            raise ValueError("Sampling lengths must be within [5, 64]")
        if min(self.samples_per_length, self.shard_size, self.steps, self.batch_size) < 1:
            raise ValueError("Sampling count, shard, step, and batch values must be positive")
        if self.samples_per_length % self.shard_size:
            raise ValueError("samples_per_length must be divisible by shard_size")


@dataclass(frozen=True)
class SamplingRunsConfig:
    """One or more independently trained Flow Matching sampling runs."""

    runs: tuple[SamplingConfig, ...]

    @classmethod
    def from_file(cls, path: Path) -> SamplingRunsConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw = payload.get("sampling_runs")
        if not isinstance(raw, list) or not raw:
            raise ValueError("Configuration must contain a non-empty 'sampling_runs' list")
        result = cls(tuple(SamplingConfig.from_mapping(value) for value in raw))
        result.validate()
        return result

    @classmethod
    def is_configured(cls, path: Path) -> bool:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return isinstance(payload.get("sampling_runs"), list)

    def validate(self) -> None:
        seeds = [run.training_seed for run in self.runs]
        if len(set(seeds)) != len(seeds):
            raise ValueError("sampling_runs training_seed values must be unique")
        manifests = [
            run.output_root / "manifests" / f"seed_{run.training_seed}.json"
            for run in self.runs
        ]
        if len(set(manifests)) != len(manifests):
            raise ValueError("sampling_runs must produce distinct manifest paths")


@dataclass(frozen=True)
class OptimizationConfig:
    ranked_candidates: Path
    initial_population: Path
    training_sequences: Path
    promotion_gate: Path
    scorer_root: Path
    safety_ad_root: Path
    stability_model: Path
    esm_model: Path
    output_root: Path
    finalization_seed_directories: tuple[Path, ...] = ()
    apex_root: Path | None = None
    strain_groups: Path | None = None
    activity_thresholds: Path | None = None
    activity_scorer: str = "apex"
    activity_targets: tuple[str, ...] = ()
    activity_aggregation: str = "median"
    activity_quantile: float = 0.5
    activity_output_column: str | None = None
    activity_command: tuple[str, ...] = ()
    activity_model_files: tuple[Path, ...] = ()
    objectives: tuple[ObjectiveSpec, ...] = ()
    seeds: tuple[int, ...] = (42, 123, 2025)
    generations: int = 100
    population_size: int = 500
    batch_size_apex: int = 3000
    batch_size_esm: int = 32
    device: str = "auto"
    protocol: str = "three_objective_no_ad_v1.3.release"
    mmseqs_command: tuple[str, ...] = ("mmseqs",)
    mmseqs_threads: int = 16
    cluster_max_members: int = 5
    resume_generation: int = 1
    apex_uncertainty_max: float = 0.3074556134498036
    novelty_max_identity: float = 0.5
    novelty_min_coverage: float = 0.8
    enforce_safety_ad: bool = True
    enforce_apex_uncertainty: bool = True
    enforce_novelty: bool = True
    enforce_initial_edit_distance: bool = False
    max_initial_edit_fraction: float = 0.25
    max_initial_edits: int = 5
    same_anchor_crossover: bool = True
    max_offspring_attempts: int = 20
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.001
    early_stopping_metric: str = "hypervolume"
    novelty_references: tuple[Path, ...] = ()

    @classmethod
    def from_file(cls, path: Path, *, require_search_inputs: bool = True) -> OptimizationConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values = _strict_section(payload, "optimization", cls)
        for key in (
            "ranked_candidates",
            "initial_population",
            "training_sequences",
            "promotion_gate",
            "scorer_root",
            "safety_ad_root",
            "stability_model",
            "esm_model",
            "output_root",
        ):
            values[key] = _path(values[key])
        if values.get("apex_root") is not None:
            values["apex_root"] = _path(values["apex_root"])
        if values.get("strain_groups") is not None:
            values["strain_groups"] = _path(values["strain_groups"])
        if values.get("activity_thresholds") is not None:
            values["activity_thresholds"] = _path(values["activity_thresholds"])
        values["seeds"] = tuple(int(value) for value in values.get("seeds", cls.seeds))
        values["mmseqs_command"] = tuple(
            str(value) for value in values.get("mmseqs_command", cls.mmseqs_command)
        )
        values["novelty_references"] = tuple(
            _path(value) for value in values.get("novelty_references", ())
        )
        values["activity_targets"] = tuple(
            str(value) for value in values.get("activity_targets", ())
        )
        values["activity_command"] = tuple(
            str(value) for value in values.get("activity_command", ())
        )
        values["activity_model_files"] = tuple(
            _path(value) for value in values.get("activity_model_files", ())
        )
        values["finalization_seed_directories"] = tuple(
            _path(value) for value in values.get("finalization_seed_directories", ())
        )
        values["objectives"] = tuple(
            ObjectiveSpec.from_value(value) for value in values.get("objectives", ())
        )
        result = cls(**values)
        result.validate(require_search_inputs=require_search_inputs)
        return result

    def validate(self, *, require_search_inputs: bool = True) -> None:
        required_files = (
            self.training_sequences,
            self.promotion_gate,
        )
        if self.activity_scorer == "apex":
            if self.apex_root is None:
                raise ValueError("apex_root is required for the legacy APEX backend")
            if self.strain_groups is None:
                raise ValueError("strain_groups is required for the legacy APEX backend")
            required_files = (*required_files, self.strain_groups)
            if self.activity_targets:
                if self.activity_thresholds is None:
                    raise ValueError(
                        "activity_thresholds is required for selected-target APEX optimization"
                    )
                required_files = (*required_files, self.activity_thresholds)
        elif self.activity_scorer == "apex_pathogen":
            if self.apex_root is None:
                raise ValueError("apex_root is required for the APEX-pathogen backend")
            if self.activity_thresholds is None:
                raise ValueError("activity_thresholds is required for the APEX-pathogen backend")
            required_files = (*required_files, self.activity_thresholds)
        if require_search_inputs:
            required_files = (
                self.ranked_candidates,
                self.initial_population,
                *required_files,
            )
        required_files = (
            *required_files,
            self.stability_model,
            *self.novelty_references,
            *self.activity_model_files,
        )
        missing_files = [str(path) for path in required_files if not path.is_file()]
        required_dirs = (self.scorer_root, self.safety_ad_root, self.esm_model)
        if self.apex_root is not None and self.activity_scorer in {"apex", "apex_pathogen"}:
            required_dirs = (*required_dirs, self.apex_root)
        missing_dirs = [str(path) for path in required_dirs if not path.is_dir()]
        if missing_files or missing_dirs:
            raise FileNotFoundError(
                f"Missing optimization assets: files={missing_files}, directories={missing_dirs}"
            )
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Optimization seeds must be non-empty and unique")
        if self.finalization_seed_directories:
            if len(self.finalization_seed_directories) != len(self.seeds):
                raise ValueError(
                    "finalization_seed_directories must contain exactly one directory per seed"
                )
            for seed, directory in zip(
                self.seeds, self.finalization_seed_directories, strict=True
            ):
                if directory.name != f"seed_{seed}":
                    raise ValueError(
                        "finalization_seed_directories must follow configured seed order and "
                        f"use seed_<seed> names ({directory.name!r} != 'seed_{seed}')"
                    )
                if not directory.is_dir():
                    raise FileNotFoundError(
                        f"Finalization seed directory not found: {directory}"
                    )
        if self.activity_scorer not in {"apex", "apex_pathogen", "command"}:
            raise ValueError("activity_scorer must be 'apex', 'apex_pathogen', or 'command'")
        if self.activity_scorer == "command":
            if not self.activity_command:
                raise ValueError("activity_command is required for the command scorer")
            if not self.activity_model_files:
                raise ValueError(
                    "activity_model_files must include the command wrapper and model assets"
                )
            if self.activity_targets:
                raise ValueError(
                    "activity_targets are only aggregated by built-in APEX scorers; "
                    "pass target arguments directly to activity_command"
                )
            placeholders = {
                token for token in self.activity_command if token in {"{input}", "{output}"}
            }
            if placeholders != {"{input}", "{output}"}:
                raise ValueError(
                    "activity_command must contain separate {input} and {output} tokens"
                )
        elif self.activity_command:
            raise ValueError("activity_command is only valid with activity_scorer: command")
        if len(set(self.activity_targets)) != len(self.activity_targets):
            raise ValueError("activity_targets must be unique")
        if any(not value.strip() for value in self.activity_targets):
            raise ValueError("activity_targets must not contain empty names")
        if self.activity_targets and self.activity_scorer in {"apex", "apex_pathogen"}:
            from amp_design.predictors.activity import _target_columns

            _target_columns(self.activity_scorer, self.activity_targets)
        if self.activity_aggregation not in {"median", "mean", "max", "min", "quantile"}:
            raise ValueError("activity_aggregation must be median, mean, max, min, or quantile")
        if not 0 <= self.activity_quantile <= 1:
            raise ValueError("activity_quantile must lie within [0, 1]")
        if self.activity_output_column is not None and not self.activity_output_column.strip():
            raise ValueError("activity_output_column must be non-empty when provided")
        for objective in self.effective_objectives:
            objective.validate()
        columns = self.objective_columns
        if len(set(columns)) != len(columns):
            raise ValueError("Objective columns must be unique")
        if (
            min(
                self.generations,
                self.population_size,
                self.batch_size_apex,
                self.batch_size_esm,
                self.mmseqs_threads,
                self.cluster_max_members,
                self.max_offspring_attempts,
            )
            < 1
        ):
            raise ValueError("Optimization numeric parameters must be positive")
        if not 1 <= self.resume_generation <= self.generations:
            raise ValueError("resume_generation must lie within [1, generations]")
        if not self.mmseqs_command:
            raise ValueError("mmseqs_command must contain at least one token")
        if not 0 < self.novelty_max_identity <= 1:
            raise ValueError("novelty_max_identity must lie within (0, 1]")
        if not 0 < self.novelty_min_coverage <= 1:
            raise ValueError("novelty_min_coverage must lie within (0, 1]")
        if self.apex_uncertainty_max <= 0:
            raise ValueError("apex_uncertainty_max must be positive")
        if not 0 < self.max_initial_edit_fraction <= 1:
            raise ValueError("max_initial_edit_fraction must lie within (0, 1]")
        if self.max_initial_edits < 1:
            raise ValueError("max_initial_edits must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if self.early_stopping_metric != "hypervolume":
            raise ValueError("early_stopping_metric must be 'hypervolume'")
        if self.activity_scorer == "apex_pathogen":
            assert self.activity_thresholds is not None
            assert self.apex_root is not None
            thresholds = json.loads(self.activity_thresholds.read_text(encoding="utf-8"))
            if thresholds.get("activity_scorer") != "apex_pathogen":
                raise ValueError("Activity-threshold contract uses the wrong scorer")
            if expected_commit := thresholds.get("repository_commit"):
                completed = subprocess.run(
                    ["git", "-C", str(self.apex_root), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                observed_commit = completed.stdout.strip()
                if completed.returncode or observed_commit != expected_commit:
                    raise ValueError(
                        "APEX-pathogen checkout does not match the activity-threshold "
                        f"contract ({observed_commit or 'unavailable'} != {expected_commit})"
                    )
            calibrated = float(thresholds["uncertainty_threshold_q95_log10_mic_sd"])
            if abs(calibrated - self.apex_uncertainty_max) > 1e-12:
                raise ValueError(
                    "apex_uncertainty_max does not match the frozen activity-threshold "
                    f"contract ({self.apex_uncertainty_max} != {calibrated})"
                )
        elif self.activity_scorer == "apex" and self.activity_targets:
            assert self.activity_thresholds is not None
            assert self.apex_root is not None
            thresholds = json.loads(self.activity_thresholds.read_text(encoding="utf-8"))
            expected = {
                "activity_scorer": "apex",
                "activity_targets": list(self.activity_targets),
                "activity_aggregation": self.activity_aggregation,
                "activity_output_column": self.effective_activity_output_column,
            }
            observed = {key: thresholds.get(key) for key in expected}
            if observed != expected:
                raise ValueError(
                    f"Selected-target APEX threshold contract mismatch: {observed} != {expected}"
                )
            calibrated = float(thresholds["uncertainty_threshold_q95_log10_mic_sd"])
            if abs(calibrated - self.apex_uncertainty_max) > 1e-12:
                raise ValueError(
                    "apex_uncertainty_max does not match the selected-target APEX "
                    f"contract ({self.apex_uncertainty_max} != {calibrated})"
                )
            completed = subprocess.run(
                ["git", "-C", str(self.apex_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            observed_commit = completed.stdout.strip()
            expected_commit = str(thresholds.get("repository_commit", ""))
            if completed.returncode or not expected_commit or observed_commit != expected_commit:
                raise ValueError(
                    "Legacy APEX checkout does not match the selected-target threshold "
                    f"contract ({observed_commit or 'unavailable'} != "
                    f"{expected_commit or 'missing'})"
                )
            from amp_design.predictors.apex import discover_models, sha256

            model_hashes = {path.name: sha256(path) for path in discover_models(self.apex_root)}
            model_payload = json.dumps(model_hashes, sort_keys=True, separators=(",", ":"))
            observed_model_set = hashlib.sha256(model_payload.encode()).hexdigest()
            expected_model_set = str(thresholds.get("model_set_sha256", ""))
            if not expected_model_set or observed_model_set != expected_model_set:
                raise ValueError(
                    "Legacy APEX model set does not match the selected-target threshold contract"
                )

    @property
    def effective_activity_output_column(self) -> str:
        if self.activity_output_column is not None:
            return self.activity_output_column
        if self.activity_targets or self.activity_scorer == "command":
            return "activity_score"
        return "apex_pathogen_median_log10_mic"

    @property
    def effective_objectives(self) -> tuple[ObjectiveSpec, ...]:
        if self.objectives:
            return self.objectives
        return (
            ObjectiveSpec(self.effective_activity_output_column),
            ObjectiveSpec("toxicity_score"),
            ObjectiveSpec("hemolysis_score"),
        )

    @property
    def objective_columns(self) -> tuple[str, ...]:
        return tuple(objective.column for objective in self.effective_objectives)

    @property
    def objective_directions(self) -> tuple[str, ...]:
        return tuple(objective.direction for objective in self.effective_objectives)

    @property
    def effective_seed_directories(self) -> tuple[Path, ...]:
        """Return explicit finalization inputs or conventional output-root paths."""

        if self.finalization_seed_directories:
            return self.finalization_seed_directories
        return tuple(self.output_root / f"seed_{seed}" for seed in self.seeds)


@dataclass(frozen=True)
class PipelineConfig:
    """Paths and hard gates for the public sampling-to-final-candidate workflow."""

    sampling_manifests: tuple[Path, ...]
    prepared_candidates: Path
    scored_candidates: Path
    finalized_dir: Path
    hard_filter_dir: Path
    run_sampling: bool = True
    scoring_chunk_size: int = 3000
    mic_max_um: float = 128.0
    activity_hard_threshold: float | None = None
    activity_hard_direction: str = "minimize"
    require_safety_ad: bool = True
    require_apex_uncertainty: bool = True
    require_novelty: bool = True
    require_stability: bool = True
    stability_min_hours: float = 1.0
    require_known_amp_novelty: bool = False
    known_amp_reference: Path | None = None
    known_amp_max_identity: float = 0.8
    known_amp_min_coverage: float = 0.8
    known_amp_mmseqs_command: tuple[str, ...] = ("mmseqs",)
    known_amp_mmseqs_threads: int = 16

    @classmethod
    def from_file(cls, path: Path) -> PipelineConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values = _strict_section(payload, "pipeline", cls)
        values["sampling_manifests"] = tuple(_path(value) for value in values["sampling_manifests"])
        for key in (
            "prepared_candidates",
            "scored_candidates",
            "finalized_dir",
            "hard_filter_dir",
        ):
            values[key] = _path(values[key])
        if values.get("known_amp_reference") is not None:
            values["known_amp_reference"] = _path(values["known_amp_reference"])
        values["known_amp_mmseqs_command"] = tuple(
            str(value)
            for value in values.get("known_amp_mmseqs_command", cls.known_amp_mmseqs_command)
        )
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        if not self.sampling_manifests:
            raise ValueError("sampling_manifests must be non-empty")
        if self.scoring_chunk_size < 1:
            raise ValueError("scoring_chunk_size must be positive")
        if self.mic_max_um <= 0 or self.stability_min_hours <= 0:
            raise ValueError("Hard-gate thresholds must be positive")
        if self.activity_hard_threshold is not None and not math.isfinite(
            self.activity_hard_threshold
        ):
            raise ValueError("activity_hard_threshold must be finite when provided")
        if self.activity_hard_direction not in {"minimize", "maximize"}:
            raise ValueError("activity_hard_direction must be 'minimize' or 'maximize'")
        outputs = {
            self.prepared_candidates.resolve(),
            self.scored_candidates.resolve(),
            self.finalized_dir.resolve(),
            self.hard_filter_dir.resolve(),
        }
        if len(outputs) != 4:
            raise ValueError("Pipeline output paths must be distinct")
        if self.require_known_amp_novelty:
            if self.known_amp_reference is None or not self.known_amp_reference.is_file():
                raise FileNotFoundError(
                    f"Known-AMP reference is missing: {self.known_amp_reference}"
                )
            if not 0 < self.known_amp_max_identity <= 1:
                raise ValueError("known_amp_max_identity must be in (0, 1]")
            if not 0 < self.known_amp_min_coverage <= 1:
                raise ValueError("known_amp_min_coverage must be in (0, 1]")
            if not self.known_amp_mmseqs_command or self.known_amp_mmseqs_threads < 1:
                raise ValueError("Known-AMP MMseqs2 command and threads must be configured")


@dataclass(frozen=True)
class PublicationConfig:
    """All-rank, dual-activity, external-safety publication finalization."""

    output_dir: Path
    apex_pathogen_root: Path
    hemopi2_root: Path
    hemopi2_wrapper: Path
    toxinpred3_root: Path
    toxinpred3_runner: Path
    toxinpred3_python: Path
    apex_pathogen_mic_max_um: float = 128.0
    toxinpred3_threshold: float = 0.38
    hemopi2_threshold: float = 0.58
    apex_pathogen_batch_size: int = 3000
    cluster_identity: float = 0.8
    cluster_coverage: float = 0.8
    max_representatives_per_cluster: int = 1
    retain_all_ranks: bool = True

    @classmethod
    def from_file(cls, path: Path) -> PublicationConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values = _strict_section(payload, "publication", cls)
        for key in (
            "output_dir",
            "apex_pathogen_root",
            "hemopi2_root",
            "hemopi2_wrapper",
            "toxinpred3_root",
            "toxinpred3_runner",
            "toxinpred3_python",
        ):
            values[key] = _path(values[key])
        result = cls(**values)
        result.validate()
        return result

    @classmethod
    def is_configured(cls, path: Path) -> bool:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return isinstance(payload.get("publication"), dict)

    def validate(self) -> None:
        required_files = (
            self.hemopi2_wrapper,
            self.toxinpred3_runner,
            self.toxinpred3_python,
        )
        required_dirs = (
            self.apex_pathogen_root,
            self.hemopi2_root,
            self.toxinpred3_root,
        )
        missing_files = [str(path) for path in required_files if not path.is_file()]
        missing_dirs = [str(path) for path in required_dirs if not path.is_dir()]
        if missing_files or missing_dirs:
            raise FileNotFoundError(
                f"Missing publication assets: files={missing_files}, directories={missing_dirs}"
            )
        positive = (
            self.apex_pathogen_mic_max_um,
            self.toxinpred3_threshold,
            self.hemopi2_threshold,
            self.apex_pathogen_batch_size,
            self.cluster_identity,
            self.cluster_coverage,
            self.max_representatives_per_cluster,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Publication numeric parameters must be positive")
        if self.cluster_identity != 0.8 or self.cluster_coverage != 0.8:
            raise ValueError("The publication protocol requires MMseqs2 80/80 clustering")
        if self.max_representatives_per_cluster != 1:
            raise ValueError("The publication protocol retains exactly one member per cluster")
        if not self.retain_all_ranks:
            raise ValueError("The publication protocol must retain all ranks before hard filters")
