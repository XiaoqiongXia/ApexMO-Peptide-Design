# ApexMO

Antimicrobial peptide generation with discrete flow matching and multi-objective
optimization of predicted activity, toxicity and hemolysis.

![ApexMO workflow](docs/images/Fig_1.png)

## Installation

Requires Linux, Conda, Git LFS, curl and unzip.

```bash
git clone https://github.com/XiaoqiongXia/ApexMO-Peptide-Design.git
cd ApexMO-Peptide-Design
bash scripts/setup/bootstrap.sh
```

## Run

Installation check with relaxed thresholds:

```bash
bash scripts/pipeline/run_full_pipeline.sh configs/smoke_e2e.yaml
```

Full workflow using [`configs/pipeline.yaml`](configs/pipeline.yaml):

```bash
bash scripts/pipeline/run_full_pipeline.sh
```

Results: `outputs/pipeline/final/04_final/` (CSV, Parquet and FASTA).

## Training

- [Sequence generator](src/amp_design/generation/training.py)
- [Toxicity, hemolysis and stability predictors](scripts/training/README.md)

## Documentation

- [Usage and configuration](docs/usage.md)
- [Models and evaluation results](MODEL_CARD.md)
- [Code layout](docs/code_layout.md)

## License

Project license pending; see [license status](LICENSE_PENDING.md) and
[third-party terms](THIRD_PARTY_NOTICES.md).
