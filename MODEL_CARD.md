# AMP Design v1.3 model card

## Intended use

This release supports non-commercial computational research on antimicrobial
peptide generation and prioritization. It is intended for hypothesis generation
and experimental candidate selection, not direct medical, diagnostic, clinical,
environmental-release, or therapeutic decisions.

## Generator models

Three independently seeded exact-length discrete Flow Matching models are
included under `models/generator/`. Each checkpoint contains the AMP DDiT model
configuration, model parameters, EMA parameters, optimizer-era metadata, and
the generation configuration required by `amp-design sample`.

The architecture uses hidden size 256, six transformer blocks, eight attention
heads, a 4x MLP ratio, length conditioning, and a maximum modeled length of 64
amino acids. The released sampling workflow uses EMA weights.

Checkpoint hashes, training seeds, selected steps, and validation losses are in
`models/MODEL_MANIFEST.json`.

## Safety models

The toxicity and hemolysis classifiers consume local ESM-2 embeddings plus
sequence length. On the dehomologized frozen evaluation used for promotion:

- toxicity: AUROC 0.8879, AUPRC 0.8202, sensitivity 0.9217, specificity 0.5282;
- hemolysis: AUROC 0.7473, AUPRC 0.7112, sensitivity 0.8657, specificity 0.3488.

These metrics are predictive performance estimates, not guarantees of safety.
Specificity is modest, particularly for hemolysis, and model outputs should be
used as prioritization scores followed by experimental testing.

## Activity oracle and external encoder

Genetic optimization uses the APEX-pathogen 8-model, 11-pathogen MIC ensemble and
`facebook/esm2_t30_150M_UR50D`. These weights are not redistributed in this
repository. `scripts/setup_external_assets.sh` retrieves pinned versions and
the optimization preflight records their hashes.

## Optimization objectives

The formal release optimization minimizes:

1. APEX-pathogen median log10 MIC;
2. toxicity probability;
3. hemolysis probability.

The search domain is canonical peptides of length 10–30 aa. The default
population is 500 sequences for 100 generations and three seeds (42, 123,
2025), with a maximum of five survivors per 80%-identity MMseqs2 cluster.
All new offspring must also pass a portable ESM 5-nearest-neighbour safety
applicability domain, the frozen APEX-pathogen ensemble-uncertainty limit, and the
configured training/reference-set novelty rule.

## Limitations

- No released sequence has been experimentally confirmed by this package.
- Model predictions may be unreliable outside the training applicability domain.
- High cationicity, aggregation, synthesis feasibility, proteolytic stability,
  immunogenicity, and off-target effects require separate assessment.
- APEX-pathogen and safety objectives are model-correlated and should not be interpreted
  as independent experimental measurements.
- The supplementary stability model is exploratory and is not part of the
  three-objective formal genetic search. It may be enabled as a final
  prioritization gate, but does not establish experimental blood stability.

## Provenance

All bundled files are listed with SHA-256 digests in `release_manifest.json`.
Training/evaluation methodology and formal rerun results are documented under
`docs/`.
