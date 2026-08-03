"""Public, configuration-driven AMP sampling and Pareto optimization commands."""

from __future__ import annotations

import argparse
import importlib.resources
import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from .release_config import (
    OptimizationConfig,
    PipelineConfig,
    PublicationConfig,
    SamplingConfig,
    SamplingRunsConfig,
)


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise a portable CPU fallback."""

    value = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if value == "auto":
        value = "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _jsonable(
    config: SamplingConfig | OptimizationConfig | PipelineConfig,
) -> dict[str, object]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = [str(item) if isinstance(item, Path) else item for item in value]
    return payload


def check_sampling(config: SamplingConfig) -> dict[str, object]:
    device = resolve_device(config.device)
    return {
        "command": "sample",
        "passed": True,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "config": _jsonable(config),
    }


def check_optimization(config: OptimizationConfig) -> dict[str, object]:
    from .apex import discover_models, load_strain_groups
    from .apex_pathogen import discover_pathogen_models

    device = resolve_device(config.device)
    gate = json.loads(config.promotion_gate.read_text(encoding="utf-8"))
    if gate.get("schema_version") != 1 or not gate.get("all_scorers_promoted", False):
        raise RuntimeError(f"Safety scorer promotion gate did not pass: {config.promotion_gate}")
    model_files = [config.scorer_root / f"{task}_model.pkl" for task in ("toxicity", "hemolysis")]
    missing = [str(path) for path in model_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing safety scorer models: {missing}")
    if config.activity_scorer == "apex_pathogen":
        apex_models = discover_pathogen_models(config.apex_root)
    elif config.activity_scorer == "apex":
        if config.strain_groups is None:
            raise ValueError("strain_groups is required for the legacy APEX backend")
        load_strain_groups(config.strain_groups)
        apex_models = discover_models(config.apex_root)
    else:
        apex_models = []
        activity_executable = shutil.which(config.activity_command[0])
        if activity_executable is None:
            raise FileNotFoundError(
                f"Activity scorer command is not executable: {config.activity_command[0]!r}"
            )
    executable = shutil.which(config.mmseqs_command[0])
    if executable is None:
        raise FileNotFoundError(
            f"MMseqs command is not executable on PATH: {config.mmseqs_command[0]!r}"
        )
    version = subprocess.run(
        [*config.mmseqs_command, "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if version.returncode:
        raise RuntimeError(
            f"Configured MMseqs command failed: {(version.stderr or version.stdout)[-1000:]}"
        )
    version_lines = (version.stdout or version.stderr).strip().splitlines()
    return {
        "command": "optimize",
        "passed": True,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "n_apex_models": len(apex_models),
        "activity_scorer": config.activity_scorer,
        "mmseqs_executable": executable,
        "mmseqs_version": version_lines[-1] if version_lines else "unknown",
        "config": _jsonable(config),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("init-config")
    initialize.add_argument("--output-dir", type=Path, default=Path("configs"))
    for command in (
        "sample",
        "optimize",
        "check-sample",
        "check-optimize",
        "prepare-candidates",
        "score-candidates",
        "initialize-pareto",
        "finalize",
        "hard-filter",
        "pipeline",
        "finalize-all-ranks",
        "score-apex-pathogen",
        "external-safety",
        "rank-final",
        "publication-pipeline",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
    optimize = subparsers.choices["optimize"]
    optimize.add_argument(
        "--preflight-only",
        action="store_true",
        help="Load and score probe sequences, write the frozen preflight, then stop.",
    )
    optimize.add_argument(
        "--only-seed",
        type=int,
        help="Run only one configured genetic seed (for multi-GPU worker sharding).",
    )
    optimize.add_argument(
        "--resume-generation",
        type=int,
        help="Override the resume cursor for this worker without changing the frozen contract.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "init-config":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        templates = importlib.resources.files("amp_design").joinpath("templates")
        for name in (
            "release_sampling_v1_3.yaml",
            "release_optimization_v1_3.yaml",
            "release_pipeline_v1_3.yaml",
            "publication_pipeline_apex_dual_v1.yaml",
        ):
            destination = args.output_dir / name
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite configuration: {destination}")
            destination.write_text(templates.joinpath(name).read_text(encoding="utf-8"))
            print(f"Wrote {destination}")
        return
    if args.command in {"sample", "check-sample"}:
        configs = (
            SamplingRunsConfig.from_file(args.config).runs
            if SamplingRunsConfig.is_configured(args.config)
            else (SamplingConfig.from_file(args.config),)
        )
        reports = [check_sampling(config) for config in configs]
        if args.command == "check-sample":
            print(json.dumps({"runs": reports}, indent=2, sort_keys=True))
            return
        try:
            from .production_sampling import sample_tasks
        except ImportError as error:
            raise RuntimeError(
                "Sampling dependencies are missing; install the pinned Flow Matching package"
            ) from error
        for config in configs:
            sample_tasks(
                config.checkpoint,
                config.output_root,
                training_seed=config.training_seed,
                lengths=config.lengths,
                samples_per_length=config.samples_per_length,
                shard_size=config.shard_size,
                steps=config.steps,
                batch_size=config.batch_size,
                device=resolve_device(config.device),
            )
        return

    publication_commands = {
        "finalize-all-ranks",
        "score-apex-pathogen",
        "external-safety",
        "rank-final",
        "publication-pipeline",
    }
    if args.command in publication_commands:
        from .publication_pipeline import (
            apply_external_safety,
            merge_and_filter_internal,
            rank_and_cluster_final,
            run_publication_pipeline,
            score_and_filter_apex_pathogen,
        )

        pipeline = PipelineConfig.from_file(args.config)
        optimization = OptimizationConfig.from_file(args.config)
        publication = PublicationConfig.from_file(args.config)
        if args.command == "finalize-all-ranks":
            result = merge_and_filter_internal(pipeline, optimization, publication)
        elif args.command == "score-apex-pathogen":
            result = score_and_filter_apex_pathogen(
                publication, device=resolve_device(optimization.device)
            )
        elif args.command == "external-safety":
            result = apply_external_safety(publication)
        elif args.command == "rank-final":
            result = rank_and_cluster_final(optimization, publication)
        else:
            result = run_publication_pipeline(
                pipeline,
                optimization,
                publication,
                device=resolve_device(optimization.device),
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command in {
        "prepare-candidates",
        "score-candidates",
        "initialize-pareto",
        "finalize",
        "hard-filter",
        "pipeline",
    }:
        from .release_pipeline import (
            finalize_optimization,
            hard_filter_candidates,
            initialize_pareto,
            prepare_candidates,
            score_candidates,
        )

        pipeline = PipelineConfig.from_file(args.config)
        optimization = OptimizationConfig.from_file(
            args.config,
            require_search_inputs=args.command in {"finalize", "hard-filter"},
        )
        if args.command == "prepare-candidates":
            print(json.dumps(prepare_candidates(pipeline), indent=2, sort_keys=True))
            return
        if args.command == "score-candidates":
            print(json.dumps(score_candidates(pipeline, optimization), indent=2, sort_keys=True))
            return
        if args.command == "initialize-pareto":
            print(json.dumps(initialize_pareto(pipeline, optimization), indent=2, sort_keys=True))
            return
        if args.command == "finalize":
            print(
                json.dumps(finalize_optimization(pipeline, optimization), indent=2, sort_keys=True)
            )
            return
        if args.command == "hard-filter":
            print(
                json.dumps(hard_filter_candidates(pipeline, optimization), indent=2, sort_keys=True)
            )
            return

        if pipeline.run_sampling:
            from .production_sampling import sample_tasks

            samplings = (
                SamplingRunsConfig.from_file(args.config).runs
                if SamplingRunsConfig.is_configured(args.config)
                else (SamplingConfig.from_file(args.config),)
            )
            expected_manifests = {
                run.output_root / "manifests" / f"seed_{run.training_seed}.json"
                for run in samplings
            }
            if expected_manifests != set(pipeline.sampling_manifests):
                raise ValueError(
                    "pipeline.sampling_manifests must exactly match sampling_runs outputs"
                )
            for sampling in samplings:
                sample_tasks(
                    sampling.checkpoint,
                    sampling.output_root,
                    training_seed=sampling.training_seed,
                    lengths=sampling.lengths,
                    samples_per_length=sampling.samples_per_length,
                    shard_size=sampling.shard_size,
                    steps=sampling.steps,
                    batch_size=sampling.batch_size,
                    device=resolve_device(sampling.device),
                )
        search_inputs_ready = (
            optimization.ranked_candidates.exists()
            and optimization.initial_population.exists()
        )
        if not search_inputs_ready:
            if not pipeline.prepared_candidates.exists():
                prepare_candidates(pipeline)
            if not pipeline.scored_candidates.exists():
                score_candidates(pipeline, optimization)
            initialize_pareto(pipeline, optimization)
        optimization = OptimizationConfig.from_file(args.config)
        from .release_optimization import run_optimization

        search_complete = all(
            (seed_directory / f"generation_{optimization.generations:03d}.csv").is_file()
            for seed_directory in optimization.effective_seed_directories
        )
        if not search_complete:
            run_optimization(optimization)
        if PublicationConfig.is_configured(args.config):
            from .publication_pipeline import run_publication_pipeline

            publication = PublicationConfig.from_file(args.config)
            result = run_publication_pipeline(
                pipeline,
                optimization,
                publication,
                device=resolve_device(optimization.device),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        if not (pipeline.finalized_dir / "post_run_manifest.json").exists():
            finalize_optimization(pipeline, optimization)
        if not (pipeline.hard_filter_dir / "hard_filter_manifest.json").exists():
            hard_filter_candidates(pipeline, optimization)
        return

    config = OptimizationConfig.from_file(args.config)
    report = check_optimization(config)
    if args.command == "check-optimize":
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    from .release_optimization import run_optimization

    run_optimization(
        config,
        preflight_only=args.preflight_only,
        only_seed=args.only_seed,
        resume_generation=args.resume_generation,
    )


if __name__ == "__main__":
    main()
