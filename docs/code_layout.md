# Code layout

Python modules are grouped by function. Setup and command-line scripts live
under `scripts/`.

```text
src/amp_design/
  datasets/       data preparation and train/validation splits
  generation/     model, training and sampling
  predictors/     activity and safety scoring
  optimization/   objectives, Pareto ranking and genetic search
  screening/      candidate pooling and cluster selection
  evaluation/     sequence metrics
  workflows/      configuration, CLI and pipeline stages
  utils/          paths, checkpoints and experiment tracking
  templates/      packaged YAML configurations

scripts/
  setup/          environment and model installation
  pipeline/       full workflow
  scoring/        ToxinPred3 and HemoPI2 adapters
  training/       toxicity, hemolysis and stability training sources
  validation/     file checks and installation tests
```

For generator training, see `generation/training.py`. Predictor training has
separate data requirements and version notes in
[`scripts/training/README.md`](../scripts/training/README.md).

## Older paths

Top-level modules such as `amp_design.training` and scripts such as
`scripts/bootstrap.sh` forward to the locations above. Existing imports and
commands still work. Edit the implementation in its subdirectory, rather than
the forwarding file. [module_moves.json](module_moves.json) lists each mapping.

For example:

| Older command | Current command |
| --- | --- |
| `bash scripts/bootstrap.sh` | `bash scripts/setup/bootstrap.sh` |
| `bash scripts/run_full_pipeline.sh` | `bash scripts/pipeline/run_full_pipeline.sh` |
| `python scripts/verify_release_assets.py` | `python scripts/validation/verify_release_assets.py` |

Installed commands such as `amp-train` and `amp-design` keep their names.

## Resuming older runs

Resume records include hashes of the source files. Use the original code
revision when continuing an existing run: moving or editing a file changes its
hash, even when the algorithm is unchanged. Keep the hashes in the run's
manifest so that these checks can detect a changed environment.
