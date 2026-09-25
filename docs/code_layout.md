# Functional code layout

Implementation code is organized by responsibility. Existing console-command names,
Python import paths and top-level script commands remain available. Old source
modules are small aliases to the canonical module object, so imports, monkeypatches
and historical pickle class references resolve to the same implementation.

```text
src/amp_design/
  datasets/       preparation.py
  generation/     model.py, tokenizer.py, flow_matching.py, samplers.py,
                  training.py, sampling.py, sharded_sampling.py
  predictors/     activity.py, apex.py, apex_pathogen.py, safety.py
  optimization/   objectives.py, pareto.py, genetic.py, search.py
  screening/      finalization.py
  evaluation/     sequence_metrics.py
  workflows/      config.py, cli.py, release.py, publication.py
  utils/          paths.py, tracking.py
  templates/      packaged YAML configuration templates
  *.py            legacy import/command compatibility aliases

scripts/
  setup/          environment bootstrap and external asset setup
  pipeline/       full-workflow launcher
  scoring/        ToxinPred3 and HemoPI2 subprocess adapters
  training/       server scorer-training sources and provenance records
  validation/     asset verification and installation/smoke checks
  *.py, *.sh      legacy command forwarders
```

## Where changes belong

- Put reusable logic in the appropriate `src/amp_design/` functional package.
- Put environment setup, process launchers and checks in the corresponding `scripts/` directory.
- Edit the canonical implementation, not its compatibility alias. There is only one copy of each implementation.
- `generation/training.py` trains the generative prior. It does not train the toxicity, hemolysis or stability predictors.
- `predictors/` contains scorer inference and aggregation. Scorer-training sources are now in [`scripts/training/`](../scripts/training/README.md). Only their project-root lookup was adjusted for the new folder depth. The server safety script differs from the hash recorded with the released models, so exact weight reproduction remains unverified.

## Commands

Recommended paths:

```bash
bash scripts/setup/bootstrap.sh
bash scripts/pipeline/run_full_pipeline.sh
bash scripts/validation/run_smoke_test.sh
python scripts/validation/verify_release_assets.py
amp-train --cfg job
amp-design --help
```

Legacy commands such as `bash scripts/bootstrap.sh`, `python scripts/verify_release_assets.py`
and `python -m amp_design.release_cli --help` still forward to the organized code.
The exact old-to-new mapping is in [module_moves.json](module_moves.json).

## Scope and reproducibility

This change reorganizes existing implementations and updates imports, command entry
points, resource paths, code-hash provenance paths, configuration runner paths and
documentation. It does not change model parameters, objective calculations, search
operators, screening thresholds or saved scientific results. Tests continue to cover
the legacy imports and add checks of canonical entry points.

Source locations and source-file hashes necessarily change. Existing runs whose
resume manifests bind old code hashes should continue to use the original code
revision; do not overwrite their recorded hashes to force a resume. No server
results or previously sealed submission artifacts are changed by this refactor.

Git LFS weights remain LFS-managed. This local refactor does not download or replace
them; use the existing setup instructions to obtain model assets.
