# Predictor training reference

Script entry points are listed in [`scripts/training/`](../scripts/training/README.md).
The one-hour stability classifier is separate from the regression model in
`models/stability_exploratory/`.

## Data and dependencies

All paths below are relative to the repository root. The processed training
data and cached embeddings are not included in this source release.

| Path | Required by |
| --- | --- |
| `outputs/custom_scorers_stage_a_v1/` | All three scripts: normalized CSVs and `mmseqs/*_cluster.tsv` group assignments |
| `external/models/facebook/esm2_t30_150M_UR50D/` | Both `train_custom_scorers*` scripts: local encoder and tokenizer |
| `outputs/custom_scorers_stage_b2_nested_v1/` | Nested-CV script: embedding cache; this directory must already exist |
| `outputs/custom_scorers_stage_b_v1/stab_batch_*.npy` | Stability classifier: cached embeddings in the expected sequence order |

The earlier training script does not produce `stab_batch_*.npy`; the script
that generated those files is not included.

The scripts use NumPy, pandas, scikit-learn and SciPy. The two
`train_custom_scorers*` scripts also use PyTorch and Transformers. These are
provided by the repository's base and `optimization` dependencies, but the
original training environment has not been recovered as a lockfile.

## Source versions

Source hashes were checked against the server on 25 September 2026.
[`source_manifest.json`](../scripts/training/source_manifest.json) records the original and
repository hashes. The only edit to each script was the project-root lookup:
`parents[1]` became `parents[2]` after moving the file into this directory.
Reversing that edit recovers the original file.

The safety script's original hash differs from the training-script hash in
[`models/safety/pretrain_manifest.json`](../models/safety/pretrain_manifest.json).
The version used to train the released safety weights has not been located.
The exact source version for the released stability regression weight is also
unverified. Retraining with these files has not been shown to reproduce the
published weights or metrics.

## Known issues

- In `train_custom_scorers.py`, the embedding loop advances by 8 but reads up to
  32 sequences per slice. On an uncached run, overlapping batches can produce
  more embedding rows than labels.
- In `train_stability_classifier.py`, each group's label comes from its first
  row, while its half-life is aggregated by the median. Those two values can
  disagree when a group contains both stable and unstable measurements.

These behaviors are retained in the source snapshots. Review them before
using the scripts for new training runs.

## Output directories

The scripts write to fixed paths under `outputs/`. The nested-CV script creates
`custom_scorers_stage_b2_nested_v1_2/` and stops if that directory already exists.
The earlier script uses `custom_scorers_stage_b_v1/`; the stability classifier
uses `custom_scorers_stage_c_stability_classifier_v1/`. Both create directories
at import time and can overwrite files when run. Run them in a separate
checkout with copies of the required inputs.
