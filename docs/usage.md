# Usage guide

Run commands from the repository root. The main entry point is
`bash scripts/pipeline/run_full_pipeline.sh`; it loads the environment configured during
installation and uses `configs/pipeline.yaml` by default.

## Installation and checks

`bash scripts/setup/bootstrap.sh` creates two Conda environments: `amp_flow` for the
main pipeline and `amp_toxinpred3` for ToxinPred3's older scikit-learn runtime.
It retrieves Git LFS files and external model weights, then saves local
interpreter paths in the ignored `.amp_design_env` file.

To rerun the installation checks:

```bash
bash scripts/validation/run_smoke_test.sh
```

These checks include file hashes, configuration loading, a two-sequence
sampling test and ToxinPred3/HemoPI2 inference. They do not run the full
optimization. `bash scripts/setup/bootstrap.sh --skip-smoke` skips these checks
during setup; run them once all assets are available.

For manual installation, create the environments and fetch the assets:

```bash
git lfs install --local
git lfs pull
conda env create -f environment.yml
conda env create -f environment-toxinpred3.yml
conda activate amp_flow
bash scripts/setup/setup_external_assets.sh
export AMP_TOXINPRED3_PYTHON="$(conda run -n amp_toxinpred3 which python)"
```

Manual installation supports the `amp-design` commands below. The shell
runners additionally require `.amp_design_env`, which bootstrap creates.
MMseqs2 is included in `environment.yml`; check it with `mmseqs version`.

## Sampling

After bootstrap, activate the main environment before using the CLI directly:

```bash
source .amp_design_env
conda activate "$AMP_DESIGN_ENV_NAME"
amp-design check-sample --config configs/sampling.yaml
amp-design sample --config configs/sampling.yaml
```

Edit `configs/sampling.yaml` to choose the checkpoint, peptide lengths, number
of samples and batch size. `device: auto` selects CUDA when available and CPU
otherwise. Completed shards are validated and skipped on subsequent runs;
manifests record sampling parameters and checkpoint hashes.

The standalone sampling configuration uses one generator. The complete
pipeline lists all three under `sampling_runs` in `configs/pipeline.yaml`.

## Full workflow

```bash
bash scripts/pipeline/run_full_pipeline.sh configs/pipeline.yaml
```

The default configuration samples 1,000 sequences at each length from 10 to
30 residues for each of three checkpoints (training seeds 42, 123 and 2025).
After deduplication and reference filtering, it scores the candidates and
builds a diverse initial population for optimization.

The search minimizes median pathogen log10 MIC, toxicity score and hemolysis
score. It uses the 11-strain subset of the 34-strain APEX ensemble,
a population of 500, and five search seeds: 42, 123, 2025, 271828 and 314159.
Each search runs for up to 25 generations, with early stopping configured
after five generations without sufficient hypervolume improvement.

Candidates must satisfy the configured applicability-domain, ensemble
uncertainty and novelty criteria. The default search also limits edits from
the initial sequence and retains at most five members per MMseqs2 cluster.
Stability is an additional predicted filter, not a search objective.

After the search, candidates from all ranks and seeds are pooled, screened
with APEX-pathogen, ToxinPred3 and HemoPI2, and ranked again. Final selection
retains one representative per MMseqs2 cluster at 80% identity and 80% coverage.
Novelty is measured against the supplied reference sequences.

## Running individual stages

With the main environment active and `AMP_TOXINPRED3_PYTHON` set, the complete
workflow can also be run as separate stages:

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

The preflight checks the scoring models and initial population before the
search. Resuming requires the same input files, models and search settings.
To continue an interrupted optimization after generation 10, set
`optimization.resume_generation: 11` in the same configuration and rerun the
optimization command with the same output root.

## Outputs

Under `outputs/pipeline/`, sampling manifests, prepared and scored candidate
tables, initial populations and search histories are retained. The final
outputs in `final/04_final/` include:

- `final_representatives.csv`
- `final_representatives.parquet`
- `final_representatives.fasta`

`pipeline_manifest.json` records the input and output hashes for each stage.
To verify the files shipped with this repository, run:

```bash
python scripts/validation/verify_release_assets.py
```

Add `--include-external` to check downloaded assets as well.

## Reference optimization

`configs/optimization.yaml` is a separate configuration for the bundled
reference pool. It uses APEX-pathogen directly, three search seeds and
100 generations. Its candidate tables are separate from the default pipeline
outputs.

```bash
amp-design check-optimize --config configs/optimization.yaml
amp-design optimize --config configs/optimization.yaml --preflight-only
amp-design optimize --config configs/optimization.yaml
```

This configuration defaults to `cuda:0`; set `device: cpu` for a CPU run.
The accompanying candidate tables are in `results/reference_candidates/`.

## Development

The bootstrap environment includes the development dependencies. Run the
checks from the repository root:

```bash
pytest -q
ruff check src tests
```

Model architecture, scoring metrics and limitations are described in
[the model notes](../MODEL_CARD.md). External models and dependencies have
separate [license terms](../THIRD_PARTY_NOTICES.md).
