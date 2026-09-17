# Model notes

## Sequence generator

ApexMO uses a length-conditioned discrete flow-matching transformer to generate
peptide sequences. The model has a hidden size of 256, six transformer blocks,
eight attention heads and a 4× MLP ratio. It supports sequences up to 64
residues; the default design workflow samples lengths of 10–30 residues.

Three checkpoints trained with seeds 42, 123 and 2025 are included in
`models/generator/`. Sampling uses exponential moving average (EMA) weights.
Checkpoint hashes, selected training steps and validation losses are recorded
in [`models/MODEL_MANIFEST.json`](models/MODEL_MANIFEST.json).

## Scoring models

The toxicity and hemolysis classifiers use ESM-2 sequence embeddings and
sequence length. The manuscript and Fig. 2 report results on the complete
frozen test sets:

| Classifier | Test sequences | AUROC | AUPRC | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Toxicity | 2,206 | 0.9303 | 0.9384 | 0.9637 | 0.5245 |
| Hemolysis | 386 | 0.8301 | 0.8298 | 0.9274 | 0.3382 |

A separate evaluation removes test sequences with a development-set MMseqs2
match at ≥50% identity and ≥80% coverage of both sequences. It reuses the
same predictions and classification thresholds, without retraining the models:

| Classifier | Retained sequences | Excluded sequences | AUROC | AUPRC | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Toxicity | 1,161 | 1,045 | 0.8879 | 0.8202 | 0.9217 | 0.5282 |
| Hemolysis | 153 | 233 | 0.7473 | 0.7112 | 0.8657 | 0.3488 |

Both tables report average precision as AUPRC. The evaluation sets also differ
in class balance: the positive fraction changes from 50.0% to 29.7% for toxicity
and from 46.4% to 43.8% for hemolysis. These are evaluations on different samples,
not scores from different model versions.

The exact values are available in
[`frozen_test_metrics.json`](models/safety/frozen_test_metrics.json) and
[`dehomologized_test_metrics.json`](models/safety/dehomologized_test_metrics.json).
Specificity is limited, particularly for hemolysis. These scores support
candidate selection and do not establish experimental safety.

The default pipeline uses the 40-model APEX ensemble during optimization,
aggregating predictions over the 11 pathogen strains listed in
[`configs/pipeline.yaml`](configs/pipeline.yaml). The 8-model APEX-pathogen
ensemble provides a further activity screen after optimization. ToxinPred3
and HemoPI2 provide external toxicity and hemolysis predictions.

The ESM encoder is `facebook/esm2_t30_150M_UR50D`. External weights are
retrieved by `scripts/setup_external_assets.sh`; versions and hashes are
recorded in the release manifests.

## Search settings

The default pipeline minimizes three objectives: median pathogen log10 MIC,
toxicity score and hemolysis score. It uses 500 sequences, five search seeds
and up to 25 generations, with early stopping. Applicability-domain,
ensemble-uncertainty, novelty and edit-distance constraints limit the search.
The exploratory stability predictor is used as an additional filter.

The separate [`configs/optimization.yaml`](configs/optimization.yaml) runs
APEX-pathogen optimization from the bundled reference pool, with three seeds
and 100 generations. Reference candidate tables correspond to that workflow.

## Interpretation

All reported activity and safety values are model predictions. The released
candidates have not been experimentally validated by this package, and
predictions may be unreliable outside the models' training domains.

The activity and safety models are not independent experimental assays.
Synthesis feasibility, aggregation, proteolytic stability, immunogenicity and
off-target effects require separate assessment. The stability score is
exploratory and does not establish blood stability. This code is intended for
computational research and candidate selection, not clinical use.

File hashes are listed in [`release_manifest.json`](release_manifest.json).
See the [usage guide](docs/usage.md) for running the models
and [third-party notices](THIRD_PARTY_NOTICES.md) for dependency terms.
