"""Pipeline stages for sampling, scoring, optimization and final filtering."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from amp_design.optimization.genetic import (
    AMINO_ACIDS,
    MAX_LENGTH,
    MIN_LENGTH,
    assign_rank_and_crowding,
    hash_sorted_fasta_records,
    survivor_order,
)
from amp_design.optimization.search import (
    ObjectiveEvaluator,
    _atomic_csv,
    _atomic_parquet,
    _atomic_text,
    _cluster_command,
    _feasible,
    _objective_columns,
    _objective_directions,
    _objective_tolerance,
    _read_complete_cluster_assignments,
    _sha256_file,
    read_table,
)
from amp_design.screening.finalization import apply_cluster_cap, finalize_combined_archive
from amp_design.utils.paths import PACKAGE_ROOT
from amp_design.workflows.config import OptimizationConfig, PipelineConfig


def _metadata_path(path: Path) -> Path:
    return path.with_name(path.stem + "_metadata.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite completed output: {path}")


def _filter_known_amp_candidates(
    frame: pd.DataFrame,
    config: PipelineConfig,
    output: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Remove candidates with an MMseqs2 80/80 match to the complete known-AMP set."""

    if not config.require_known_amp_novelty:
        return frame, {"enabled": False, "removed": 0, "retained": len(frame)}
    hits_path = output.with_name(output.stem + "_known_amp_mmseqs_hits.tsv")
    audit_path = output.with_name(output.stem + "_known_amp_novelty_audit.parquet")
    with tempfile.TemporaryDirectory(prefix="amp_known_novelty_") as temporary:
        work = Path(temporary)
        fasta = work / "candidates.fasta"
        tmp = work / "tmp"
        with fasta.open("w", encoding="utf-8") as handle:
            for identifier, sequence in hash_sorted_fasta_records(frame):
                handle.write(f">{identifier}\n{sequence}\n")
        command = [
            *config.known_amp_mmseqs_command,
            "easy-search",
            str(fasta),
            str(config.known_amp_reference),
            str(hits_path),
            str(tmp),
            "--min-seq-id",
            str(config.known_amp_max_identity),
            "-c",
            str(config.known_amp_min_coverage),
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
            str(config.known_amp_mmseqs_threads),
            "--format-output",
            "query,target,fident,qcov,tcov,alnlen",
            "--remove-tmp-files",
            "1",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode:
            raise RuntimeError(
                f"Known-AMP MMseqs2 search failed: {(completed.stderr or completed.stdout)[-2000:]}"
            )
    # Downstream readers expect a result file even when there are no hits.
    hits_path.touch(exist_ok=True)
    matched: set[str] = set()
    if hits_path.is_file() and hits_path.stat().st_size:
        hits = pd.read_csv(hits_path, sep="\t", header=None, usecols=[0])
        matched = set(hits.iloc[:, 0].astype(str))
    audit = frame.copy()
    audit["known_amp_identity_ge_threshold"] = audit["sequence_sha256"].isin(matched)
    audit["known_amp_novelty_pass"] = ~audit["known_amp_identity_ge_threshold"]
    _atomic_parquet(audit, audit_path)
    retained = audit.loc[audit["known_amp_novelty_pass"]].copy().reset_index(drop=True)
    return retained, {
        "enabled": True,
        "reference": str(config.known_amp_reference.resolve()),
        "reference_sha256": _sha256_file(config.known_amp_reference),
        "identity_strictly_below": config.known_amp_max_identity,
        "minimum_coverage": config.known_amp_min_coverage,
        "removed": len(audit) - len(retained),
        "retained": len(retained),
        "hits": str(hits_path.resolve()),
        "hits_sha256": _sha256_file(hits_path),
        "audit": str(audit_path.resolve()),
        "audit_sha256": _sha256_file(audit_path),
    }


def prepare_candidates(config: PipelineConfig) -> dict[str, object]:
    """Stream, validate, and exactly de-duplicate sampling shards into Parquet."""

    output = config.prepared_candidates
    _require_new(output)
    if output.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("prepared_candidates must be a Parquet path")
    output.parent.mkdir(parents=True, exist_ok=True)
    database_handle, database_name = tempfile.mkstemp(prefix="amp_candidates_", suffix=".sqlite")
    os.close(database_handle)
    database = Path(database_name)
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE candidates (
            sequence TEXT PRIMARY KEY,
            sequence_sha256 TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            source_candidate_id TEXT,
            length INTEGER NOT NULL,
            training_seed INTEGER,
            generation_seed INTEGER,
            shard_id INTEGER,
            checkpoint_sha256 TEXT,
            sampling_steps INTEGER,
            use_ema INTEGER,
            source_manifest TEXT NOT NULL,
            source_shard TEXT NOT NULL
        )
        """
    )
    input_rows = invalid_rows = duplicate_rows = 0
    shard_records: list[dict[str, object]] = []
    try:
        for manifest_path in config.sampling_manifests:
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Sampling manifest not found: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checkpoint_hash = str(manifest["checkpoint_sha256"])
            for task in manifest.get("tasks", []):
                shard = Path(task["output"])
                if not shard.is_file():
                    raise FileNotFoundError(f"Sampling shard not found: {shard}")
                frame = pd.read_csv(shard)
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
                if missing := required.difference(frame.columns):
                    raise ValueError(f"Sampling shard {shard} lacks columns: {sorted(missing)}")
                if len(frame) != int(task["count"]):
                    raise ValueError(f"Sampling shard row-count mismatch: {shard}")
                if not frame["checkpoint_sha256"].astype(str).eq(checkpoint_hash).all():
                    raise ValueError(f"Sampling shard checkpoint mismatch: {shard}")
                for row in frame.itertuples(index=False):
                    input_rows += 1
                    sequence = str(row.sequence).strip().upper()
                    length = len(sequence)
                    valid = (
                        MIN_LENGTH <= length <= MAX_LENGTH
                        and set(sequence).issubset(set(AMINO_ACIDS))
                        and int(row.length) == length
                    )
                    if not valid:
                        invalid_rows += 1
                        continue
                    sequence_hash = hashlib.sha256(sequence.encode()).hexdigest()
                    values = (
                        sequence,
                        sequence_hash,
                        f"cand_{sequence_hash}",
                        str(row.candidate_id),
                        length,
                        int(row.training_seed),
                        int(row.generation_seed),
                        int(row.shard_id),
                        str(row.checkpoint_sha256),
                        int(row.sampling_steps),
                        int(bool(row.use_ema)),
                        str(manifest_path.resolve()),
                        str(shard.resolve()),
                    )
                    cursor = connection.execute(
                        "INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        values,
                    )
                    duplicate_rows += int(cursor.rowcount == 0)
                connection.commit()
                shard_records.append(
                    {
                        "path": str(shard.resolve()),
                        "sha256": _sha256_file(shard),
                        "rows": len(frame),
                    }
                )
        unique_rows = int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
        if unique_rows < 1:
            raise ValueError("No valid unique 10-30 aa candidates were prepared")
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError(
                "Candidate preparation requires PyArrow; install amp-design[optimization]"
            ) from error

        temporary = output.with_suffix(output.suffix + ".tmp")
        writer = None
        query = "SELECT * FROM candidates ORDER BY sequence_sha256"
        try:
            for chunk in pd.read_sql_query(query, connection, chunksize=100_000):
                chunk["use_ema"] = chunk["use_ema"].astype(bool)
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise RuntimeError("Prepared candidate Parquet writer received no rows")
    finally:
        connection.close()
        database.unlink(missing_ok=True)
    try:
        prepared = pd.read_parquet(temporary)
        prepared, known_amp_novelty = _filter_known_amp_candidates(prepared, config, output)
        if config.require_known_amp_novelty:
            _atomic_parquet(prepared, output)
        else:
            os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "sampling_manifests": [str(path.resolve()) for path in config.sampling_manifests],
        "input_rows": input_rows,
        "invalid_rows": invalid_rows,
        "duplicate_rows": duplicate_rows,
        "exact_unique_rows": unique_rows,
        "output_rows": len(prepared),
        "known_amp_novelty": known_amp_novelty,
        "length_domain": [MIN_LENGTH, MAX_LENGTH],
        "shards": shard_records,
        "output": str(output.resolve()),
        "output_sha256": _sha256_file(output),
    }
    _atomic_text(_metadata_path(output), json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def score_candidates(
    pipeline: PipelineConfig,
    optimization: OptimizationConfig,
) -> dict[str, object]:
    """Score prepared candidates with activity, safety, AD, novelty, and stability."""

    output = pipeline.scored_candidates
    _require_new(output)
    prepared = read_table(pipeline.prepared_candidates)
    objectives = _objective_columns(optimization)
    evaluator = ObjectiveEvaluator(pd.DataFrame(columns=["sequence", *objectives]), optimization)
    parts = []
    state_root = output.parent / (output.stem + "_state")
    for start in range(0, len(prepared), pipeline.scoring_chunk_size):
        source = prepared.iloc[start : start + pipeline.scoring_chunk_size].copy()
        scored = evaluator.evaluate(source["sequence"].tolist(), state_root)
        provenance_columns = [
            column
            for column in source.columns
            if column == "sequence" or column not in scored.columns
        ]
        provenance = source[provenance_columns].set_index("sequence")
        scored = scored.join(provenance, on="sequence", how="left", validate="one_to_one")
        parts.append(scored)
        print(
            json.dumps(
                {
                    "stage": "score",
                    "completed": min(start + len(source), len(prepared)),
                    "total": len(prepared),
                }
            ),
            flush=True,
        )
    result = pd.concat(parts, ignore_index=True)
    result["formal_pareto_eligible"] = _feasible(result, optimization)
    _atomic_parquet(result, output)
    metadata = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "input": str(pipeline.prepared_candidates.resolve()),
        "input_sha256": _sha256_file(pipeline.prepared_candidates),
        "output": str(output.resolve()),
        "output_sha256": _sha256_file(output),
        "rows": len(result),
        "formal_pareto_eligible": int(result["formal_pareto_eligible"].sum()),
        "safety_ad_pass": int(result["formal_safety_ad_pass"].sum()),
        "novelty_pass": int(result["novelty_pass"].sum()),
        "apex_uncertainty_pass": (
            int(result["apex_uncertainty_feasible"].sum())
            if "apex_uncertainty_feasible" in result
            else None
        ),
        "stable": int(result["stability_prediction"].eq("Stable").sum()),
    }
    _atomic_text(_metadata_path(output), json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def _cluster_assignments(
    frame: pd.DataFrame,
    optimization: OptimizationConfig,
    destination: Path,
) -> dict[str, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="amp_release_cluster_") as temporary:
        work = Path(temporary)
        fasta, prefix, tmp = work / "input.fasta", work / "clusters", work / "tmp"
        with fasta.open("w", encoding="utf-8") as handle:
            for identifier, sequence in hash_sorted_fasta_records(frame):
                handle.write(f">{identifier}\n{sequence}\n")
        command = _cluster_command(optimization, fasta, prefix, tmp)
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode:
            raise RuntimeError(f"MMseqs clustering failed: {completed.stderr[-2000:]}")
        cluster_file = Path(f"{prefix}_cluster.tsv")
        if not cluster_file.is_file():
            raise RuntimeError("MMseqs cluster output is missing")
        shutil.copyfile(cluster_file, destination)
    expected_members = set(frame["sequence_sha256"].astype(str))
    return _read_complete_cluster_assignments(destination, expected_members)


def initialize_pareto(
    pipeline: PipelineConfig,
    optimization: OptimizationConfig,
) -> dict[str, object]:
    """Rank newly scored candidates and build a cluster-diverse initial population."""

    _require_new(optimization.ranked_candidates)
    _require_new(optimization.initial_population)
    scored = read_table(pipeline.scored_candidates)
    feasible = _feasible(scored, optimization)
    scored = scored.copy()
    scored["formal_pareto_eligible"] = feasible
    objectives = _objective_columns(optimization)
    directions = _objective_directions(optimization)
    ranked = assign_rank_and_crowding(
        scored,
        objectives,
        feasible=feasible,
        directions=directions,
    )
    ranked = ranked.sort_values(
        ["formal_pareto_eligible", "pareto_rank", "crowding_distance", "sequence_sha256"],
        ascending=[False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    eligible = ranked.loc[ranked["formal_pareto_eligible"]].copy()
    if len(eligible) < optimization.population_size:
        raise RuntimeError(
            f"Only {len(eligible)} candidates pass initialization constraints; "
            f"need {optimization.population_size}"
        )
    cluster_file = optimization.initial_population.with_name(
        optimization.initial_population.stem + "_mmseqs_cluster.tsv"
    )
    assignments = _cluster_assignments(eligible, optimization, cluster_file)
    counts: dict[str, int] = {}
    selected = []
    for index in survivor_order(eligible):
        row = eligible.iloc[index]
        cluster = assignments[str(row.sequence_sha256)]
        if counts.get(cluster, 0) >= optimization.cluster_max_members:
            continue
        counts[cluster] = counts.get(cluster, 0) + 1
        selected.append(index)
        if len(selected) == optimization.population_size:
            break
    if len(selected) != optimization.population_size:
        raise RuntimeError(
            f"Diversity cap retained {len(selected)} initial candidates; "
            f"need {optimization.population_size}"
        )
    initial = eligible.iloc[selected].copy().reset_index(drop=True)
    _atomic_parquet(ranked, optimization.ranked_candidates)
    _atomic_csv(initial, optimization.initial_population)
    metadata = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "input": str(pipeline.scored_candidates.resolve()),
        "input_sha256": _sha256_file(pipeline.scored_candidates),
        "ranked_candidates": str(optimization.ranked_candidates.resolve()),
        "ranked_sha256": _sha256_file(optimization.ranked_candidates),
        "initial_population": str(optimization.initial_population.resolve()),
        "initial_sha256": _sha256_file(optimization.initial_population),
        "rows": len(ranked),
        "eligible": int(feasible.sum()),
        "population_size": len(initial),
        "clusters_in_initial": len(counts),
    }
    _atomic_text(
        _metadata_path(optimization.initial_population),
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )
    return metadata


def finalize_optimization(
    pipeline: PipelineConfig,
    optimization: OptimizationConfig,
) -> dict[str, object]:
    """Pool complete seed caches, re-rank, and apply the final cluster cap."""

    completion = pipeline.finalized_dir / "post_run_manifest.json"
    _require_new(completion)
    pipeline.finalized_dir.mkdir(parents=True, exist_ok=True)
    evolved = []
    cache_paths = []
    for seed, seed_directory in zip(
        optimization.seeds, optimization.effective_seed_directories, strict=True
    ):
        cache = seed_directory / "evaluation_cache.parquet"
        if not cache.is_file():
            raise FileNotFoundError(f"Optimization cache not found: {cache}")
        frame = pd.read_parquet(cache)
        if "candidate_origin" in frame:
            frame = frame.loc[frame["candidate_origin"].eq("evolved")].copy()
        frame["genetic_seed"] = seed
        evolved.append(frame)
        cache_paths.append(cache)
    original = read_table(optimization.ranked_candidates)
    objectives = _objective_columns(optimization)
    directions = _objective_directions(optimization)
    score_tolerances = {
        objective: _objective_tolerance(objective, optimization) for objective in objectives
    }
    ranked = finalize_combined_archive(
        original,
        evolved,
        objectives,
        original_eligibility_column="formal_pareto_eligible",
        evolved_eligibility_column="formal_genetic_feasible",
        directions=directions,
        score_tolerance=score_tolerances,
        front_only=True,
    )
    ranked_path = pipeline.finalized_dir / "combined_archive_ranked.parquet"
    front_path = pipeline.finalized_dir / "combined_rank0_front.csv"
    diverse_path = pipeline.finalized_dir / "combined_rank0_diverse.csv"
    _atomic_parquet(ranked, ranked_path)
    front = ranked.loc[ranked["combined_archive_front"]].copy()
    _atomic_csv(front, front_path)
    clusters = _cluster_assignments(
        front,
        optimization,
        pipeline.finalized_dir / "combined_rank0_mmseqs_cluster.tsv",
    )
    diverse = apply_cluster_cap(front, clusters, max_per_cluster=optimization.cluster_max_members)
    _atomic_csv(diverse, diverse_path)
    metadata = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "seeds": list(optimization.seeds),
        "original": str(optimization.ranked_candidates.resolve()),
        "original_sha256": _sha256_file(optimization.ranked_candidates),
        "evaluation_caches": [str(path.resolve()) for path in cache_paths],
        "evaluation_cache_sha256": {
            str(path.resolve()): _sha256_file(path) for path in cache_paths
        },
        "activity_scorer": optimization.activity_scorer,
        "objectives": [
            {"column": column, "direction": direction}
            for column, direction in zip(objectives, directions, strict=True)
        ],
        "score_tolerances": score_tolerances,
        "code_sha256": {
            str(Path(__file__).resolve()): _sha256_file(Path(__file__)),
            str((PACKAGE_ROOT / "screening/finalization.py").resolve()): (
                _sha256_file(PACKAGE_ROOT / "screening/finalization.py")
            ),
        },
        "combined_unique": len(ranked),
        "combined_rank0": len(front),
        "diverse_rank0": len(diverse),
        "outputs": {
            path.name: _sha256_file(path) for path in (ranked_path, front_path, diverse_path)
        },
    }
    _atomic_text(completion, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def hard_filter_candidates(
    pipeline: PipelineConfig,
    optimization: OptimizationConfig,
) -> dict[str, object]:
    """Apply the configured filters and export CSV and FASTA files."""

    completion = pipeline.hard_filter_dir / "hard_filter_manifest.json"
    _require_new(completion)
    source = pipeline.finalized_dir / "combined_rank0_diverse.csv"
    data = pd.read_csv(source, low_memory=False)
    if "activity_scorer" not in data or set(data["activity_scorer"].dropna().astype(str)) != {
        optimization.activity_scorer
    }:
        raise ValueError("Final candidate table has incompatible activity-scorer provenance")
    if optimization.activity_scorer == "apex_pathogen":
        if "safety_ad_protocol" not in data or set(
            data["safety_ad_protocol"].dropna().astype(str)
        ) != {"portable_esm_length_v1"}:
            raise ValueError("Final candidate table has incompatible safety-AD provenance")
    gate = json.loads(optimization.promotion_gate.read_text(encoding="utf-8"))
    toxicity_max = float(gate["tasks"]["toxicity"]["dehomologized_metrics"]["threshold"])
    hemolysis_max = float(gate["tasks"]["hemolysis"]["dehomologized_metrics"]["threshold"])
    activity_column = getattr(
        optimization,
        "effective_activity_output_column",
        "apex_pathogen_median_log10_mic",
    )
    if activity_column not in data:
        raise ValueError(f"Final candidate table lacks activity column {activity_column!r}")
    if optimization.activity_scorer in {"apex", "apex_pathogen"}:
        activity_threshold = math.log10(pipeline.mic_max_um)
        activity_direction = "minimize"
    else:
        activity_threshold = pipeline.activity_hard_threshold
        activity_direction = pipeline.activity_hard_direction
    if activity_threshold is None:
        data["activity_hard_threshold_pass"] = True
    elif activity_direction == "minimize":
        data["activity_hard_threshold_pass"] = data[activity_column].lt(activity_threshold)
    else:
        data["activity_hard_threshold_pass"] = data[activity_column].gt(activity_threshold)
    data["mic_hard_threshold_pass"] = data["activity_hard_threshold_pass"]
    data["toxicity_hard_threshold_pass"] = data["toxicity_score"].lt(toxicity_max)
    data["hemolysis_hard_threshold_pass"] = data["hemolysis_score"].lt(hemolysis_max)
    data["safety_ad_hard_threshold_pass"] = (
        data["formal_safety_ad_pass"].fillna(False).astype(bool)
        if pipeline.require_safety_ad
        else True
    )
    data["apex_uncertainty_hard_threshold_pass"] = (
        data["apex_median_log10_mic_sd"].le(optimization.apex_uncertainty_max)
        if pipeline.require_apex_uncertainty
        else True
    )
    novelty = (
        data["novelty_pass"].fillna(False).astype(bool)
        if "novelty_pass" in data
        else ~data["has_ge_50_identity_match"].fillna(True).astype(bool)
    )
    data["novelty_hard_threshold_pass"] = novelty if pipeline.require_novelty else True
    data["stability_hard_threshold_pass"] = (
        data["stability_half_life_hours"].ge(pipeline.stability_min_hours)
        if pipeline.require_stability
        else True
    )
    pass_columns = [
        "activity_hard_threshold_pass",
        "toxicity_hard_threshold_pass",
        "hemolysis_hard_threshold_pass",
        "safety_ad_hard_threshold_pass",
        "apex_uncertainty_hard_threshold_pass",
        "novelty_hard_threshold_pass",
        "stability_hard_threshold_pass",
    ]
    data["all_hard_thresholds_pass"] = data[pass_columns].all(axis=1)
    selected = data.loc[data["all_hard_thresholds_pass"]].copy()
    objective_columns = list(_objective_columns(optimization))
    objective_directions = _objective_directions(optimization)
    if missing := set(objective_columns).difference(selected.columns):
        raise ValueError(f"Final candidate table lacks objective columns: {sorted(missing)}")
    selected = selected.sort_values(
        [*objective_columns, "sequence"],
        ascending=[
            *[direction == "minimize" for direction in objective_directions],
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    selected.insert(
        0,
        "final_candidate_id",
        [f"AMPFINAL{index:05d}" for index in range(1, len(selected) + 1)],
    )
    pipeline.hard_filter_dir.mkdir(parents=True, exist_ok=True)
    audit_path = pipeline.hard_filter_dir / "hard_filter_audit.csv"
    csv_path = pipeline.hard_filter_dir / "final_candidates.csv"
    fasta_path = pipeline.hard_filter_dir / "final_candidates.fasta"
    _atomic_csv(data, audit_path)
    _atomic_csv(selected, csv_path)
    temporary_fasta = fasta_path.with_suffix(".fasta.tmp")
    with temporary_fasta.open("w", encoding="utf-8") as handle:
        for row in selected.itertuples(index=False):
            handle.write(f">{row.final_candidate_id}\n{row.sequence}\n")
    os.replace(temporary_fasta, fasta_path)
    metadata = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "input": str(source.resolve()),
        "input_sha256": _sha256_file(source),
        "input_rows": len(data),
        "selected_rows": len(selected),
        "thresholds": {
            "activity_column": activity_column,
            "activity_threshold_strict": activity_threshold,
            "activity_direction": activity_direction,
            "mic_max_um_strict": (
                pipeline.mic_max_um
                if optimization.activity_scorer in {"apex", "apex_pathogen"}
                else None
            ),
            "toxicity_max_strict": toxicity_max,
            "hemolysis_max_strict": hemolysis_max,
            "apex_uncertainty_max": optimization.apex_uncertainty_max,
            "novelty_max_identity": optimization.novelty_max_identity,
            "stability_min_hours": pipeline.stability_min_hours,
        },
        "required": {
            "safety_ad": pipeline.require_safety_ad,
            "apex_uncertainty": pipeline.require_apex_uncertainty,
            "novelty": pipeline.require_novelty,
            "stability": pipeline.require_stability,
        },
        "pass_counts": {column: int(data[column].sum()) for column in pass_columns},
        "outputs": {path.name: _sha256_file(path) for path in (audit_path, csv_path, fasta_path)},
        "warning": "All gates are computational predictions; experimental validation is required.",
    }
    _atomic_text(completion, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata
