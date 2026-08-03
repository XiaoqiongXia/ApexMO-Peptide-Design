# AMP Design v1.3

Research code and frozen model assets for an auditable peptide-design workflow:
exact-length Flow Matching sampling, candidate scoring and filtering,
three-objective genetic Pareto optimization, and final hard-gate export.

> All activity, toxicity, hemolysis, and stability values are model
> predictions. Generated sequences have not been experimentally validated and
> are not intended for clinical or therapeutic use.

## Repository contents

- `src/amp_design/`: installable Python package.
- `models/generator/`: three independently trained Flow Matching checkpoints.
- `models/safety/`: frozen toxicity/hemolysis classifiers and portable ESM safety-AD assets.
- `assets/pareto/`: optional frozen reference pool for legacy result reproduction.
- `assets/training/`: AMP training sequences used only to freeze residue frequencies.
- `assets/known_amp/`: frozen complete known-AMP reference used before pool initialization.
- `results/reference_candidates/`: released computational candidate tables.
- `configs/`: editable sampling, optimization, and end-to-end pipeline configurations.
- `external/`: third-party assets fetched separately under their own licenses.

Large project-owned files are tracked with Git LFS. The supported first-run
entry point creates/updates both Conda environments, pulls LFS assets, downloads
and verifies third-party assets, records the local ToxinPred3 interpreter in an
ignored runtime file, and runs a non-computational installation smoke test:

```bash
bash scripts/bootstrap.sh
```

This step is resumable and may be rerun. Use `--skip-smoke` only when preparing
an offline installation whose assets will be verified later.

## Manual installation

The following commands document what `bootstrap.sh` automates:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[optimization,dev]'
```

Fetch the pinned ESM-2, APEX/APEX-pathogen, ToxinPred3, and official HemoPI2
assets. Third-party assets remain subject to their respective licenses.

```bash
bash scripts/setup_external_assets.sh
```

MMseqs2 is also required. Install it with your package manager and verify that
`mmseqs version` succeeds before starting optimization.

ToxinPred3 has a frozen environment separate from the main package. Create that
environment according to its upstream installation instructions and export its
Python executable before validating or running the publication configuration:

```bash
export AMP_TOXINPRED3_PYTHON="$(conda run -n amp_toxinpred3 which python)"
```

The YAML references `${AMP_TOXINPRED3_PYTHON}` and therefore contains no
machine-specific home directory.

## Verify or run

The smoke test validates hashes, configuration contracts, MMseqs2, both Conda
environments, model discovery, two-sequence Flow Matching inference, real
ToxinPred3/HemoPI2 inference, and the release pipeline tests without starting a
formal optimization run:

```bash
bash scripts/run_smoke_test.sh
```

Run the complete pipeline on a deliberately tiny sample, including real Flow
Matching sampling, APEX34 scoring, one-generation genetic optimization,
independent APEX-pathogen and external safety scoring, and final MMseqs2
representative selection:

```bash
bash scripts/run_full_pipeline.sh configs/smoke_e2e.yaml
```

`configs/smoke_e2e.yaml` uses relaxed hard thresholds solely to test pipeline
mechanics. Its final sequences are not scientifically screened candidates.

Start or resume the complete publication workflow with:

```bash
bash scripts/run_full_pipeline.sh
```

An alternative configuration can be passed as the first argument to either
runner. Machine-specific values are read from the ignored `.amp_design_env`
created by bootstrap, never committed to Git.

## End-to-end workflow

The default publication pipeline samples peptides from three independently
trained Flow Matching checkpoints, validates and exactly de-duplicates all
shards, and removes candidates matching the bundled complete known-AMP
reference at 80% identity/80% coverage. It then scores APEX34 activity over the
declared 11-pathogen subset, internal safety, stability, applicability domain,
and training-reference novelty; builds a new cluster-diverse 500-sequence
initial population; runs the genetic search; and pools all final ranks across
seeds. Final candidates must also pass independent APEX-pathogen, ToxinPred3,
and HemoPI2 gates before global Pareto re-ranking and one rank-prioritized
MMseqs2 80/80 representative per cluster.

```bash
amp-design pipeline --config configs/pipeline.yaml
```

Large runs may execute the same stages separately:

```bash
amp-design sample --config configs/pipeline.yaml
amp-design prepare-candidates --config configs/pipeline.yaml
amp-design score-candidates --config configs/pipeline.yaml
amp-design initialize-pareto --config configs/pipeline.yaml
amp-design optimize --config configs/pipeline.yaml --preflight-only
amp-design optimize --config configs/pipeline.yaml
amp-design finalize-all-ranks --config configs/pipeline.yaml
amp-design score-apex-pathogen --config configs/pipeline.yaml
amp-design external-safety --config configs/pipeline.yaml
amp-design rank-final --config configs/pipeline.yaml
```

Final outputs are written to `outputs/pipeline/final/04_final/`; the root
`pipeline_manifest.json` binds upstream sampling manifests, scored and initial
pool tables, seed completion records, stage outputs, and final representatives
by SHA-256. “Novel” is relative to the bundled reference and configured
training/additional references; it does not imply absence from every external
peptide database.

## Sampling

The default pipeline configuration uses generator checkpoints trained with
seeds 42, 123, and 2025 and automatically selects CUDA when available, with a
CPU fallback. Each run writes its own resumable manifest under the shared
sampling output root.

```bash
amp-design check-sample --config configs/sampling.yaml
amp-design sample --config configs/sampling.yaml
```

Sampling is deterministic, exact-length, sharded, validated, and resumable.
The output manifest records the checkpoint hash and all sampling parameters.
The sampler no longer has the former 100-shard ceiling; practical scale remains
limited by storage and compute resources.

## Genetic Pareto optimization

```bash
amp-design check-optimize --config configs/optimization.yaml
amp-design optimize --config configs/optimization.yaml --preflight-only
amp-design optimize --config configs/optimization.yaml
```

The optimizer minimizes predicted pathogen log10 MIC, toxicity, and hemolysis.
It runs independent configured seeds, applies an 80%-identity MMseqs2 diversity
cap, freezes code/input/model/config hashes, and saves complete lineage, RNG,
history, clustering, and evaluation-cache state for resumption.
New offspring must also pass the configured ESM safety applicability domain,
APEX-pathogen ensemble-uncertainty limit, and MMseqs2 novelty rule. Stability remains an
exploratory prediction and is applied only as an optional final hard gate.

To resume after generation 51, copy `configs/optimization.yaml`, set
`resume_generation: 52`, keep every other frozen setting unchanged, and run the
same optimize command.

## Reproducibility and model scope

See [`MODEL_CARD.md`](MODEL_CARD.md), [`models/MODEL_MANIFEST.json`](models/MODEL_MANIFEST.json),
and [`release_manifest.json`](release_manifest.json). Third-party terms are
summarized in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License status

The project owner has not yet selected a project-level source-code license.
Until a `LICENSE` file is added, the repository is source-available with no
permission grant beyond applicable law. See [`LICENSE_PENDING.md`](LICENSE_PENDING.md).
Flow Matching remains CC BY-NC; APEX-pathogen is distributed under the MIT license.
