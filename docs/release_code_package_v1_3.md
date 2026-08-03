# AMP Design v1.3 research code package

## Release status

This package is a research-code release for reproducible AMP sampling and
three-objective genetic Pareto optimization. It does not claim experimental,
clinical, or therapeutic validation of generated peptides.

The Python wheel contains the AMP Design implementation. Model checkpoints,
training data, ESM weights, APEX weights, MMseqs2, and generated outputs are
external runtime assets and are not bundled into the wheel.

## Installation

Python 3.10 or newer is required. Install the optimization extras when genetic
optimization will be run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[optimization]'
```

The sampler uses the pinned `flow-matching==1.0.10` API. That dependency is
CC BY-NC 4.0 and therefore constrains this release workflow to non-commercial
research unless separate permission is obtained.

The optimization extra pins the ESM/scorer runtime (`transformers==5.12.1`,
`scikit-learn==1.7.1`). The full preflight records these and all other core
package versions together with model and input hashes.

MMseqs2 must be executable using the command listed in the optimization YAML.
The checked project environment uses `conda run -n bg mmseqs`.

An installed wheel can write both editable example configurations without
requiring the source checkout:

```bash
amp-design init-config --output-dir configs/release_v1_3
```

## Sampling

Copy and edit `configs/release_sampling_v1_3.example.yaml`, then validate all
paths and the selected compute device without writing output:

```bash
amp-design check-sample --config configs/release_sampling_v1_3.example.yaml
```

Run exact-length, deterministic, resumable sharded sampling:

```bash
amp-design sample --config configs/release_sampling_v1_3.example.yaml
```

Each completed shard is validated before it is accepted. Existing valid shards
are skipped. A manifest records the checkpoint SHA-256, seeds, lengths, device,
sampling steps, and every shard path.

## Genetic Pareto optimization

Copy and edit `configs/release_optimization_v1_3.example.yaml`. The required
inputs are the frozen ranked candidate table, a 500-sequence initial
population, AMP training sequences, the passed scorer-promotion gate, two
safety models, a local ESM-2 model, all 40 APEX ensemble models, the APEX strain
grouping, and MMseqs2.

Run the lightweight asset check first:

```bash
amp-design check-optimize --config configs/release_optimization_v1_3.example.yaml
```

Run the full scoring preflight. This loads every scoring stack, verifies the
initial objective values, scores novel probes, and freezes all input and model
hashes:

```bash
amp-design optimize \
  --config configs/release_optimization_v1_3.example.yaml \
  --preflight-only
```

Run all configured seeds and generations:

```bash
amp-design optimize --config configs/release_optimization_v1_3.example.yaml
```

## End-to-end sampling-to-candidate pipeline

The GitHub release also provides a single configuration that connects multiple
Flow Matching checkpoints to exact de-duplication, complete-known-AMP MMseqs2
80% identity/80% coverage filtering, scoring, Pareto initialization, genetic
search, multi-seed all-rank finalization, dual activity gates, external safety,
and representative export:

```bash
amp-design pipeline --config configs/pipeline.yaml
```

The same run can be split into resumable stages:

```bash
amp-design sample --config configs/pipeline.yaml
amp-design prepare-candidates --config configs/pipeline.yaml
amp-design score-candidates --config configs/pipeline.yaml
amp-design initialize-pareto --config configs/pipeline.yaml
amp-design optimize --config configs/pipeline.yaml --preflight-only
amp-design optimize --config configs/pipeline.yaml
amp-design finalize --config configs/pipeline.yaml
amp-design hard-filter --config configs/pipeline.yaml
```

For a multi-checkpoint run, declare each generator under `sampling_runs:` and
list the corresponding manifests under `pipeline.sampling_manifests`. The
one-command runner rejects a mismatch between these two lists. Set
`pipeline.require_known_amp_novelty: true` and provide
`pipeline.known_amp_reference` to ensure that the initial optimization pool is
built only after the declared known-AMP screen. Existing completed manifests
can be reused with `run_sampling: false`; a clean reproduction uses new output
paths and `run_sampling: true`.

For the all-rank dual-APEX publication protocol, add a `publication:` section
and use `publication_pipeline_apex_dual_v1.yaml`. The single `pipeline` command
then replaces the legacy rank-0 finalization with the following resumable
stages:

```bash
amp-design finalize-all-ranks --config configs/publication.yaml
amp-design score-apex-pathogen --config configs/publication.yaml
amp-design external-safety --config configs/publication.yaml
amp-design rank-final --config configs/publication.yaml
```

The final stage globally re-ranks every hard-filter-eligible candidate and
retains one MMseqs2 80% identity/80% coverage representative per cluster,
prioritizing lower Pareto rank and then higher crowding distance.

`pipeline_manifest.json` binds the sampling manifests, prepared and scored
tables, initialized pool, per-seed completion records, and every publication
stage by SHA-256, so the final representatives can be traced to the sampling
checkpoints without relying on a frozen initial pool.

Machine-specific interpreter paths are supplied through the environment, not
committed in YAML. Before loading a publication configuration, set:

```bash
export AMP_TOXINPRED3_PYTHON="$(conda run -n amp_toxinpred3 which python)"
```

The assembled GitHub release automates this setup. `scripts/bootstrap.sh`
creates or updates the `amp_flow` and `amp_toxinpred3` environments from their
versioned YAML files, pulls Git LFS content, downloads pinned external assets,
and writes the resolved interpreter to an ignored `.amp_design_env` file.
`scripts/run_smoke_test.sh` validates the installation without sampling, while
`scripts/run_full_pipeline.sh` starts or resumes the configured workflow.

New offspring are exact-excluded against the training sequences, the complete
ranked pool, and configured additional reference tables. MMseqs2 additionally
enforces the declared identity/coverage novelty rule. Formal feasibility also
requires the portable ESM safety applicability domain and the APEX ensemble
uncertainty limit. Stability remains exploratory and is an optional final hard
gate rather than a Pareto objective.

The optimizer writes every generation, lineage, MMseqs2 cluster assignment,
RNG state, full sequence history, and evaluation cache atomically. To resume
after generation 51, set `resume_generation: 52` in a copied YAML while keeping
the same output root and frozen inputs, then invoke the same optimize command.

## Reproducibility contract

The release optimizer minimizes predicted pathogen log10 MIC, toxicity score,
and hemolysis score. It uses canonical 10–30 aa sequences, deterministic
mutation/crossover, seeded constrained rank-and-crowding selection, and an
80%-identity MMseqs2 cluster cap of five sequences per cluster.

Do not change the inputs, models, promotion gate, strain grouping, or runtime
implementation after preflight. A mismatch is rejected rather than silently
continuing a different experiment.

## External assets and licensing

APEX is marked as non-profit research use in the project notices and must be
obtained and used under its own terms. ESM-2, MMseqs2, PyTorch, Transformers,
and all other third-party components retain their respective licenses. See
`THIRD_PARTY_NOTICES.md`.

The repository currently has no project-level `LICENSE` file. A project owner
must choose and add the intended source-code license before uploading this
package to a public registry or release page.
