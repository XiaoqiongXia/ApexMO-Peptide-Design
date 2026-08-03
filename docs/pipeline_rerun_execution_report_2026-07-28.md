# AMP design pipeline correction and rerun report

- Execution date: 2026-07-28
- Scope: safety-scorer gate, offline Pareto, historical correction, formal genetic search,
  pooled finalization, benchmark summary, downstream local predictors, and figures
- Historical result directories were not overwritten.

## Outcome

The corrected workflow was executed through the new formal pooled genetic Pareto result.
Model training and the original 59,422-candidate APEX/safety inference were not repeated,
because their frozen predictions passed the new hash, mapping, and promotion checks. The
offline ranking and genetic search were rerun in new versioned directories.

All values below are computational predictions, not experimental antimicrobial activity or
safety measurements.

## Safety-scorer promotion gate

Existing frozen predictions were evaluated after excluding frozen-test sequences with a
development-set MMseqs match at identity >=0.50 and query/target coverage >=0.80.

| Task | Retained n | AUROC | AUPRC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| Toxicity | 1,161 | 0.8879 | 0.8202 | 0.9217 | 0.5282 |
| Hemolysis | 153 | 0.7473 | 0.7112 | 0.8657 | 0.3488 |

The strict gate at
`outputs/custom_scorers_stage_b2_nested_v1_2/scorer_promotion_gate_v1_3.json`
reports `all_scorers_promoted=true`.

## Offline Pareto v1.2

The non-overwriting offline run is in `outputs/pareto_offline_custom_v1_2`.

- complete audit rows: 59,422;
- formal eligible rows: 28,495;
- offline rank-0 rows: 144;
- safety-priority eligible rows: 453;
- all formal-eligible ranks and front membership match the prior v1.1 computation;
- the new run additionally enforces the successful promotion gate and frozen expected hashes.

## Historical v1.2 correction

The exploratory recovery is in
`outputs/genetic_pareto_nsga2_v1_2_historical_corrected`.

The legacy cache did not contain every lineage sequence. Generation snapshots recovered 855,
796, and 727 scored rows for seeds 42, 123, and 2025. A further 4,167, 3,730, and 3,806
unselected lineage proposals, respectively, had no recoverable score and were not invented or
silently included. The maximally recoverable archive contains 166,765 unique feasible rows,
348 pooled rank-0 rows, and 306 rows after the cluster cap. This output remains exploratory.

## Formal genetic search

GPU preflight used NVIDIA H100 PCIe hardware. It verified:

- zero objective difference for all 500 initial candidates;
- successful new-sequence ESM and 40-model APEX inference;
- hashes for all 40 APEX weights, ESM weights, both safety models, the promotion gate,
  runner, ranked pool, and initial pool;
- exact sequence order, SHA-256 mapping, and finite objectives.

Seeds 123 and 2025 completed uninterrupted under protocol
`three_objective_no_ad_v1.3.1`. Seed 42 exposed a final-cache bug in the first resumable
attempt: the resumed process had not reloaded the prior evaluation cache. That attempt was not
used. The runner was corrected to reload the cache on resume and atomically persist it after
every generation. Seed 42 was rerun from the beginning under
`three_objective_no_ad_v1.3.2`, deliberately stopped after generation 51, and successfully
resumed at generation 52.

For every accepted seed trajectory:

- 100 generation, lineage, RNG state, history, MMseqs cluster, and MMseqs metadata files exist;
- lineage contains 50,000 rows and 50,000 unique offspring;
- evaluation cache contains 50,500 unique rows (500 initial + 50,000 offspring);
- history contains 50,500 rows and has zero cache misses;
- final population contains 500 unique hashes and finite values for all three objectives.

The v1.3.2 change affects cache persistence/resume only; proposal, scoring, ranking, and
survival semantics are unchanged from v1.3.1. Exact protocol and preflight provenance are
recorded per seed in the pooled manifest.

## Formal pooled result

The formal pooled output is in `outputs/genetic_pareto_nsga2_v1_3_finalized`.

| Result | Count |
|---|---:|
| Combined unique feasible archive | 178,486 |
| True pooled rank-0 front | 297 |
| Rank-0 after 80% identity cluster cap (max 5/cluster) | 237 |
| Cluster-diverse candidates with predicted median pathogen MIC <64 µM | 191 |
| Three-hard-threshold candidates (MIC <128 µM plus both custom safety thresholds) | 132 |

The pooled front was computed with an exact incremental non-dominated archive. Regression
tests compare this mode with full non-dominated sorting and obtain identical front membership.

## Released-model and exploratory stability audit

All 132 three-hard-threshold candidates were evaluated locally:

- HemoPI2: 118 non-hemolytic and 14 hemolytic predictions;
- ToxinPred3: 113 non-toxin and 19 toxin predictions;
- released-model safety consensus: 102 candidates;
- exploratory Ridge stability model: 17/132 predicted half-life >=1 hour;
- among the 102 released-model safety-consensus candidates: 7 predicted half-life >=1 hour.

The repository does not contain the DBAASP general AMP classifier used for the historical
318-to-87 intersection. None of the new 132 sequences overlaps the historical 318 sequences,
so old DBAASP labels cannot be reused. Consequently no new DBAASP-intersection count is
reported; obtaining those predictions is the remaining external step.

## Corrected benchmark summary

The corrected output is in
`outputs/generative_model_benchmark_v1/formal/primary_qc_v1_1_corrected`.
Model-level endpoints are unchanged. Cell-local de-duplication changes 27 of 189
seed-by-length cells; the largest increase in a cell's unique-sequence count is 62.

## Figures

Two standalone bundles were generated, each with editable PDF/SVG, 600-dpi PNG, plotted-data
CSV, and metadata JSON:

- `figures/genetic_pareto_rank0_progress_v1_3`;
- `figures/genetic_pareto_pooled_front_v1_3`.

Both use white backgrounds, no grid, restrained fixed colors or `cividis`, standard
sans-serif text, and 89-mm intended width.

## Production interpretation

Use `outputs/genetic_pareto_nsga2_v1_3_finalized` for pooled-front and candidate-count claims.
Do not use the historical 740/318/87/56 tables as pooled rank-0 results. The 102-candidate
released-model consensus table is a computational cross-predictor audit and not an
experimentally validated safety set. DBAASP prediction and experimental testing remain
required before final experimental nomination.
