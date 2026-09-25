# ApexMO

ApexMO generates antimicrobial peptide candidates with a length-conditioned
flow-matching model, then searches for sequences with lower predicted MIC,
toxicity and hemolysis. This repository contains three trained generators,
scoring models, and the code for training, sampling and candidate selection.

![ApexMO workflow and sequence analysis](docs/images/Fig_1.png)

*Model training, peptide generation, Pareto optimization and sequence analysis.*

## Installation

On Linux, install Conda, Git LFS, curl and unzip, then run:

```bash
git clone https://github.com/XiaoqiongXia/ApexMO-Peptide-Design.git
cd ApexMO-Peptide-Design
bash scripts/setup/bootstrap.sh
```

Setup creates the `amp_flow` and `amp_toxinpred3` environments, downloads model
weights and checks that the models load and run. Rerun the same command if a
download is interrupted. The default pipeline uses CUDA when available and
otherwise runs on CPU.

## Run

Check the installation with a small example:

```bash
bash scripts/pipeline/run_full_pipeline.sh configs/smoke_e2e.yaml
```

The example uses relaxed thresholds. For candidate selection, review
[`configs/pipeline.yaml`](configs/pipeline.yaml) and run:

```bash
bash scripts/pipeline/run_full_pipeline.sh
```

The default workflow samples peptides of 10–30 residues and optimizes their
predicted activity, toxicity and hemolysis. APEX scores activity during the
search; APEX-pathogen, ToxinPred3 and HemoPI2 screen the resulting candidates.
Final CSV, Parquet and FASTA files are saved in `outputs/pipeline/final/04_final/`.

See the [usage guide](docs/usage.md) for individual stages, output files and
resuming interrupted runs. The [model notes](MODEL_CARD.md) describe the search
settings, evaluation results and prediction limits. Candidate peptides require
experimental validation.

## Training

Generator training is in
[`src/amp_design/generation/training.py`](src/amp_design/generation/training.py).
Toxicity, hemolysis and stability training scripts are in
[`scripts/training/`](scripts/training/README.md). That directory lists the
required data, known source issues and unresolved differences between the
training scripts and the versions recorded with the released weights.

## Repository

| Directory | Contents |
| --- | --- |
| [`src/amp_design/`](src/amp_design/) | Generation, scoring, optimization and screening |
| [`scripts/`](scripts/) | Setup, workflow runners, predictor training and installation checks |
| [`configs/`](configs/) | Training, sampling and optimization settings |
| [`models/`](models/) | Generator checkpoints and scoring models |
| [`assets/`](assets/) | Generator training sequences and reference sets |
| [`results/reference_candidates/`](results/reference_candidates/) | Candidate tables from the reference optimization run |
| [`tests/`](tests/) | Unit and integration tests |

The [code layout](docs/code_layout.md) lists the modules by function and maps
the older import and script paths to their current locations.

## License

A project license has not yet been selected; see [license status](LICENSE_PENDING.md).
External code and models have their own [license terms](THIRD_PARTY_NOTICES.md).
