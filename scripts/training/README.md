# Predictor training

| Script | Purpose |
| --- | --- |
| [`train_custom_scorers_nested_v1_1.py`](train_custom_scorers_nested_v1_1.py) | Toxicity and hemolysis classifiers; nested cross-validation and model fitting |
| [`train_custom_scorers.py`](train_custom_scorers.py) | Earlier safety classifiers and exploratory stability regression |
| [`train_stability_classifier.py`](train_stability_classifier.py) | One-hour stability classification; cross-validation only |

Processed training inputs are not bundled, and reproduction of the released
weights is unverified. See the [training reference](../../docs/predictor_training.md)
for input paths, output directories and issues to review before running.
