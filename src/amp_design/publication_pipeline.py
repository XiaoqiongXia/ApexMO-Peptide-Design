"""Resumable all-rank publication finalization for optimized AMP candidates."""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .apex import sha256
from .apex_pathogen import APEX_PATHOGEN_STRAINS, score_apex_pathogen_ensemble
from .external_safety import score_hemopi2, score_toxinpred3
from .genetic_pareto import assign_rank_and_crowding
from .release_config import OptimizationConfig, PipelineConfig, PublicationConfig
from .release_optimization import _atomic_csv, _atomic_parquet, _atomic_text
from .release_pipeline import _cluster_assignments


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _existing_manifest(path: Path) -> dict[str, object] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _write_manifest(path: Path, payload: dict[str, object]) -> dict[str, object]:
    payload = {"schema_version": 1, "created_at": _now(), **payload}
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype("string").str.lower().isin({"true", "1"})


def _final_generation_path(directory: Path, configured_generation: int) -> Path:
    completion = directory / "optimization_complete.json"
    generation = configured_generation
    if completion.is_file():
        generation = int(json.loads(completion.read_text(encoding="utf-8"))["last_generation"])
    path = directory / f"generation_{generation:03d}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Final seed population is missing: {path}")
    return path


def merge_and_filter_internal(
    pipeline: PipelineConfig,
    optimization: OptimizationConfig,
    publication: PublicationConfig,
) -> dict[str, object]:
    """Merge all final populations and apply predeclared internal hard gates."""

    output = publication.output_dir / "01_internal_filters"
    manifest_path = output / "manifest.json"
    if existing := _existing_manifest(manifest_path):
        return existing
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        _final_generation_path(directory, optimization.generations)
        for directory in optimization.effective_seed_directories
    ]
    parts = []
    for seed, path in zip(optimization.seeds, paths, strict=True):
        frame = pd.read_csv(path, low_memory=False)
        frame["source_seed"] = seed
        frame["seed_pareto_rank"] = frame["pareto_rank"]
        frame["seed_crowding_distance"] = frame["crowding_distance"]
        parts.append(frame)
    pooled = pd.concat(parts, ignore_index=True, sort=False)
    source_seeds = pooled.groupby("sequence", sort=False)["source_seed"].agg(
        lambda values: json.dumps(sorted({int(value) for value in values}))
    )
    unique = (
        pooled.sort_values(["sequence_sha256", "source_seed"], kind="mergesort")
        .drop_duplicates("sequence", keep="first")
        .reset_index(drop=True)
    )
    unique["source_seeds_json"] = unique["sequence"].map(source_seeds)

    gate = json.loads(optimization.promotion_gate.read_text(encoding="utf-8"))
    toxicity_max = float(gate["tasks"]["toxicity"]["dehomologized_metrics"]["threshold"])
    hemolysis_max = float(gate["tasks"]["hemolysis"]["dehomologized_metrics"]["threshold"])
    activity = optimization.effective_activity_output_column
    unique["legacy_apex_selected11_mic_pass"] = unique[activity].lt(
        math.log10(pipeline.mic_max_um)
    )
    unique["toxicity_hard_threshold_pass"] = unique["toxicity_score"].lt(toxicity_max)
    unique["hemolysis_hard_threshold_pass"] = unique["hemolysis_score"].lt(hemolysis_max)
    unique["safety_ad_hard_threshold_pass"] = (
        _bool(unique["formal_safety_ad_pass"]) if pipeline.require_safety_ad else True
    )
    unique["apex_uncertainty_hard_threshold_pass"] = (
        unique["apex_median_log10_mic_sd"].le(optimization.apex_uncertainty_max)
        if pipeline.require_apex_uncertainty
        else True
    )
    novelty = (
        _bool(unique["novelty_pass"])
        if "novelty_pass" in unique
        else ~_bool(unique["has_ge_50_identity_match"])
    )
    unique["novelty_hard_threshold_pass"] = novelty if pipeline.require_novelty else True
    unique["stability_hard_threshold_pass"] = (
        unique["stability_half_life_hours"].ge(pipeline.stability_min_hours)
        if pipeline.require_stability
        else True
    )
    unique["initial_edit_distance_hard_threshold_pass"] = (
        _bool(unique["initial_edit_distance_pass"])
        & unique["initial_edit_distance"].le(unique["max_allowed_initial_edits"])
        if optimization.enforce_initial_edit_distance
        else True
    )
    pass_columns = [
        "legacy_apex_selected11_mic_pass",
        "toxicity_hard_threshold_pass",
        "hemolysis_hard_threshold_pass",
        "safety_ad_hard_threshold_pass",
        "apex_uncertainty_hard_threshold_pass",
        "novelty_hard_threshold_pass",
        "stability_hard_threshold_pass",
        "initial_edit_distance_hard_threshold_pass",
    ]
    unique["all_internal_hard_filters_pass"] = unique[pass_columns].all(axis=1)
    eligible = unique.loc[unique["all_internal_hard_filters_pass"]].copy()
    audit_path = output / "all_final_population_audit.parquet"
    eligible_path = output / "internal_eligible.parquet"
    _atomic_parquet(unique, audit_path)
    _atomic_parquet(eligible, eligible_path)
    return _write_manifest(
        manifest_path,
        {
            "stage": "merge_all_ranks_and_internal_hard_filters",
            "inputs": {str(path.resolve()): sha256(path) for path in paths},
            "thresholds": {
                "legacy_apex_selected11_median_mic_um_strict": pipeline.mic_max_um,
                "toxicity_score_strict": toxicity_max,
                "hemolysis_score_strict": hemolysis_max,
            },
            "rows": {"pooled": len(pooled), "unique": len(unique), "eligible": len(eligible)},
            "outputs": {path.name: sha256(path) for path in (audit_path, eligible_path)},
        },
    )


def score_and_filter_apex_pathogen(
    publication: PublicationConfig, *, device: torch.device
) -> dict[str, object]:
    """Apply the independent APEX-pathogen median-MIC hard gate."""

    source = publication.output_dir / "01_internal_filters/internal_eligible.parquet"
    output = publication.output_dir / "02_apex_pathogen"
    manifest_path = output / "manifest.json"
    if existing := _existing_manifest(manifest_path):
        return existing
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source)
    legacy_renames = {
        "apex_pathogen_median_log10_mic": "legacy_apex_selected11_median_log10_mic",
        "apex_median_log10_mic_sd": "legacy_apex_selected11_median_log10_mic_sd",
        "apex_max_log10_mic_sd": "legacy_apex_selected11_max_log10_mic_sd",
        "activity_scorer": "optimization_activity_scorer",
    }
    frame = frame.rename(
        columns={key: value for key, value in legacy_renames.items() if key in frame}
    )
    state_path = output / "state.npz"
    scores, models = score_apex_pathogen_ensemble(
        frame["sequence"].tolist(),
        publication.apex_pathogen_root,
        state_path,
        device=device,
        batch_size=publication.apex_pathogen_batch_size,
    )
    if collisions := set(scores.columns).intersection(frame.columns):
        raise RuntimeError(f"APEX-pathogen output column collision: {sorted(collisions)}")
    scored = pd.concat([frame.reset_index(drop=True), scores], axis=1)
    scored["activity_scorer"] = "apex_pathogen"
    scored["apex_pathogen_median_mic_um"] = np.power(
        10.0, scored["apex_pathogen_median_log10_mic"]
    )
    scored["apex_pathogen_mic_hard_threshold_pass"] = scored[
        "apex_pathogen_median_mic_um"
    ].lt(publication.apex_pathogen_mic_max_um)
    eligible = scored.loc[scored["apex_pathogen_mic_hard_threshold_pass"]].copy()
    scored_path = output / "scored.parquet"
    eligible_path = output / "eligible.parquet"
    _atomic_parquet(scored, scored_path)
    _atomic_parquet(eligible, eligible_path)
    return _write_manifest(
        manifest_path,
        {
            "stage": "true_apex_pathogen_hard_filter",
            "input": str(source.resolve()),
            "input_sha256": sha256(source),
            "model_sha256": {str(path.resolve()): sha256(path) for path in models},
            "pathogens": list(APEX_PATHOGEN_STRAINS),
            "threshold_mic_um_strict": publication.apex_pathogen_mic_max_um,
            "rows": {"input": len(frame), "eligible": len(eligible)},
            "outputs": {
                path.name: sha256(path) for path in (state_path, scored_path, eligible_path)
            },
        },
    )


def apply_external_safety(publication: PublicationConfig) -> dict[str, object]:
    """Apply ToxinPred3 and HemoPI2 after both activity hard gates."""

    source = publication.output_dir / "02_apex_pathogen/eligible.parquet"
    output = publication.output_dir / "03_external_safety"
    manifest_path = output / "manifest.json"
    if existing := _existing_manifest(manifest_path):
        return existing
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source).reset_index(drop=True)
    frame.insert(0, "external_screen_id", [f"AMPSCREEN{i:05d}" for i in range(1, len(frame) + 1)])
    with tempfile.TemporaryDirectory(prefix="amp_publication_safety_", dir="/tmp") as temporary:
        work = Path(temporary)
        hemopi2 = score_hemopi2(
            frame,
            work,
            id_column="external_screen_id",
            root=publication.hemopi2_root,
            wrapper=publication.hemopi2_wrapper,
            threshold=publication.hemopi2_threshold,
        )
        toxinpred3 = score_toxinpred3(
            frame,
            work,
            id_column="external_screen_id",
            root=publication.toxinpred3_root,
            runner=publication.toxinpred3_runner,
            python=publication.toxinpred3_python,
            threshold=publication.toxinpred3_threshold,
        )
    audit = frame.merge(hemopi2, on="external_screen_id", validate="one_to_one").merge(
        toxinpred3, on="external_screen_id", validate="one_to_one"
    )
    if not audit["sequence"].eq(audit["hemopi2_sequence"]).all():
        raise RuntimeError("HemoPI2 sequence merge mismatch")
    audit["toxinpred3_non_toxin_pass"] = audit["toxinpred3_prediction"].eq("Non-Toxin")
    audit["hemopi2_non_hemolytic_pass"] = audit["hemopi2_prediction"].eq("Non-Hemolytic")
    audit["all_external_safety_filters_pass"] = (
        audit["toxinpred3_non_toxin_pass"] & audit["hemopi2_non_hemolytic_pass"]
    )
    eligible = audit.loc[audit["all_external_safety_filters_pass"]].copy()
    audit_path = output / "audit.parquet"
    eligible_path = output / "eligible.parquet"
    _atomic_parquet(audit, audit_path)
    _atomic_parquet(eligible, eligible_path)
    return _write_manifest(
        manifest_path,
        {
            "stage": "external_toxicity_and_hemolysis_hard_filters",
            "input": str(source.resolve()),
            "input_sha256": sha256(source),
            "thresholds": {
                "toxinpred3": publication.toxinpred3_threshold,
                "hemopi2": publication.hemopi2_threshold,
            },
            "rows": {
                "input": len(frame),
                "toxinpred3_non_toxin": int(audit["toxinpred3_non_toxin_pass"].sum()),
                "hemopi2_non_hemolytic": int(audit["hemopi2_non_hemolytic_pass"].sum()),
                "eligible": len(eligible),
            },
            "outputs": {path.name: sha256(path) for path in (audit_path, eligible_path)},
        },
    )


def rank_and_cluster_final(
    optimization: OptimizationConfig, publication: PublicationConfig
) -> dict[str, object]:
    """Globally re-rank final survivors and retain one rank-prioritized 80/80 representative."""

    source = publication.output_dir / "03_external_safety/eligible.parquet"
    output = publication.output_dir / "04_final"
    manifest_path = output / "manifest.json"
    if existing := _existing_manifest(manifest_path):
        return existing
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source)
    objectives = ("apex_pathogen_median_log10_mic", "toxicity_score", "hemolysis_score")
    ranked = assign_rank_and_crowding(
        frame,
        objectives,
        feasible=np.ones(len(frame), dtype=bool),
        directions=("minimize", "minimize", "minimize"),
    ).sort_values(
        ["pareto_rank", "crowding_distance", "sequence_sha256"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "candidate_id", [f"AMPFINAL{i:05d}" for i in range(1, len(ranked) + 1)])
    cluster_path = output / "mmseqs80_cluster.tsv"
    assignments = _cluster_assignments(ranked, optimization, cluster_path)
    ranked["mmseqs80_cluster_id"] = ranked["sequence_sha256"].map(assignments)
    ranked["mmseqs80_cluster_size"] = ranked.groupby("mmseqs80_cluster_id", sort=False)[
        "sequence"
    ].transform("size")
    representatives = ranked.groupby("mmseqs80_cluster_id", sort=False).head(1).copy()
    representatives.insert(
        0,
        "representative_id",
        [f"AMPREP{i:05d}" for i in range(1, len(representatives) + 1)],
    )
    ranked_csv = output / "all_eligible_ranked.csv"
    ranked_parquet = output / "all_eligible_ranked.parquet"
    representatives_csv = output / "final_representatives.csv"
    representatives_parquet = output / "final_representatives.parquet"
    fasta_path = output / "final_representatives.fasta"
    _atomic_csv(ranked, ranked_csv)
    _atomic_parquet(ranked, ranked_parquet)
    _atomic_csv(representatives, representatives_csv)
    _atomic_parquet(representatives, representatives_parquet)
    temporary = fasta_path.with_suffix(".fasta.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in representatives.itertuples(index=False):
            handle.write(f">{row.representative_id}\n{row.sequence}\n")
    os.replace(temporary, fasta_path)
    outputs = (
        ranked_csv,
        ranked_parquet,
        representatives_csv,
        representatives_parquet,
        fasta_path,
        cluster_path,
    )
    return _write_manifest(
        manifest_path,
        {
            "stage": "final_global_ranking_and_mmseqs80_representatives",
            "input": str(source.resolve()),
            "input_sha256": sha256(source),
            "objectives": list(objectives),
            "representative_priority": [
                "pareto_rank ascending",
                "crowding_distance descending",
                "sequence_sha256 ascending",
            ],
            "rows": {
                "all_eligible": len(ranked),
                "rank0": int(ranked["pareto_rank"].eq(0).sum()),
                "mmseqs80_clusters": int(ranked["mmseqs80_cluster_id"].nunique()),
                "representatives": len(representatives),
            },
            "outputs": {path.name: sha256(path) for path in outputs},
        },
    )


def run_publication_pipeline(
    pipeline: PipelineConfig,
    optimization: OptimizationConfig,
    publication: PublicationConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Run or resume every publication finalization stage in protocol order."""

    publication.output_dir.mkdir(parents=True, exist_ok=True)
    stages = {
        "internal": merge_and_filter_internal(pipeline, optimization, publication),
        "apex_pathogen": score_and_filter_apex_pathogen(publication, device=device),
        "external_safety": apply_external_safety(publication),
        "final": rank_and_cluster_final(optimization, publication),
    }
    upstream_paths = [
        *pipeline.sampling_manifests,
        pipeline.prepared_candidates,
        pipeline.scored_candidates,
        optimization.ranked_candidates,
        optimization.initial_population,
    ]
    upstream_paths.extend(
        directory / "optimization_complete.json"
        for directory in optimization.effective_seed_directories
    )
    for path in tuple(upstream_paths):
        upstream_paths.extend(
            (
                path.with_name(path.stem + "_metadata.json"),
                path.with_name(path.stem + ".metadata.json"),
            )
        )
    upstream = {
        str(path.resolve()): sha256(path)
        for path in upstream_paths
        if path.is_file()
    }
    manifest_path = publication.output_dir / "pipeline_manifest.json"
    return _write_manifest(
        manifest_path,
        {
            "protocol": "all_rank_dual_apex_external_safety_mmseqs80_v1",
            "upstream_artifacts": upstream,
            "stage_manifests": {
                name: str((publication.output_dir / directory / "manifest.json").resolve())
                for name, directory in (
                    ("internal", "01_internal_filters"),
                    ("apex_pathogen", "02_apex_pathogen"),
                    ("external_safety", "03_external_safety"),
                    ("final", "04_final"),
                )
            },
            "rows": {name: stage.get("rows", {}) for name, stage in stages.items()},
        },
    )
