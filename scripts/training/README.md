# Scorer-training source archive

These scripts were copied from the current research-server working directory.
Their SHA-256 values were checked against the server before inclusion. The only
source edit changes `Path(__file__).resolve().parents[1]` to `parents[2]` so that
the new directory depth still resolves the repository root. Training logic,
parameters, input selection and output filenames are unchanged.

| Script | Role | Relationship to released weights |
| --- | --- | --- |
| `train_custom_scorers_nested_v1_1.py` | Nested group cross-validation and full-data fitting for toxicity and hemolysis | Current server source; its hash differs from the source hash recorded with the released safety weights |
| `train_custom_scorers.py` | Earlier safety classifiers and exploratory stability Ridge regression | Historical reference; exact correspondence to the released exploratory stability weight is unverified |
| `train_stability_classifier.py` | Cross-validation of the one-hour blood-stability classifier | Writes evaluation artifacts; does not export a final fitted classifier and is not the released stability regression model |

## Provenance

[`source_manifest.json`](source_manifest.json) records both the original server
hash and the relocated repository hash for each script, along with the exact
single replacement. Reversing that replacement recovers the original bytes.

The released [`pretrain_manifest.json`](../../models/safety/pretrain_manifest.json)
records safety training-source SHA-256
`b28905cb6c5f7f13890d7abdc2e5f8b3325d13c9a295991c61f32dc9a5b7c0af`.
The current server source is
`5b59939c5a7e65f4a4c5e3d1052cccef14c99cbc8722dbd11b8ad176c29efbef`.
Neither the original manifest nor model weights were changed to reconcile this
difference. A historical training commit has not been established. These sources
are therefore available for inspection, but exact reproduction of published
weights or metrics has not been demonstrated.

## Inputs and environment

Paths are relative to the repository root, regardless of the working directory:

- `outputs/custom_scorers_stage_a_v1/`: normalized toxicity, hemolysis and
  stability CSVs and the `mmseqs/*_cluster.tsv` group assignments used by each
  script. These preprocessing artifacts are not bundled in this source release.
- `external/models/facebook/esm2_t30_150M_UR50D/`: local encoder and tokenizer
  assets required by the two `train_custom_scorers*` scripts.
- `outputs/custom_scorers_stage_b2_nested_v1/`: existing embedding-cache directory
  used by the nested-CV script. It must exist before that script writes a cache.
- `outputs/custom_scorers_stage_b_v1/stab_batch_*.npy`: existing stability
  embeddings required by the classifier script. The earlier training script
  does not create these batch files; their generation step is not included here.

The sources import NumPy, pandas, scikit-learn, SciPy and, for the two
`train_custom_scorers*` scripts, PyTorch and Transformers. The repository's base
and `optimization` dependencies provide these libraries, including SciPy through
scikit-learn. This is not a recovered lockfile of the original training run.

## Execution behavior and known limitations

These are standalone Python sources under `scripts/training/`, not part of the
default candidate-generation pipeline. Read the requirements before executing
them in a separate local checkout containing copies of the required inputs.

- The nested-CV script creates `outputs/custom_scorers_stage_b2_nested_v1_2/`
  and fails if that output directory already exists.
- The earlier script and the stability classifier create their output
  directories at import time and can overwrite files when executed. They are
  preserved as sources, not exposed as importable package APIs.
- In the earlier script, the embedding loop advances by 8 while its slice
  contains up to 32 sequences. The resulting overlap can misalign embedding rows
  and labels on an uncached run. This source issue is preserved and documented;
  the file must not be represented as a validated clean-run reproduction.
- The stability classifier depends on the existing embedding order. Its source
  labels grouped records using the first row's binary label while also computing
  a median half-life; these can disagree for mixed-label records. That behavior
  is unchanged and requires scientific review before a new training release.

Integration checks cover source hashes, Python syntax, the relocated root, and
the existing repository tests. No training was executed, no metrics were
regenerated, and no server results were modified. Completing a reproducible
training release still requires matching the historical source version and
providing the corresponding preprocessing artifacts and environment record.
