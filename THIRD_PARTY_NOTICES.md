# Third-party notices

## Facebook Research Flow Matching

- Source: <https://github.com/facebookresearch/flow_matching>
- Pinned revision: `11568d3`
- License: Creative Commons Attribution-NonCommercial 4.0 International
- Runtime package: `flow-matching==1.0.10`

This project uses the Flow Matching probability path, scheduler, solver, and model-wrapper APIs for non-commercial research. AMP-specific model and training code lives in this repository rather than modifying the external source tree.

## pymoo

- Source: <https://github.com/anyoptimization/pymoo>
- Pinned revision: `23110c1`
- License: Apache License 2.0
- Runtime package: `pymoo>=0.6.2,<0.7`

pymoo provides non-dominated sorting and crowding-distance calculations for the Pareto-ranking phase.

## Penn Software APEX

- Source: <https://gitlab.com/machine-biology-group-public/apex>
- Pinned revision: `2030551`
- License: non-profit research use only; see `external/apex/LICENSE`
- Local path: `external/apex`

APEX supplies a 40-model ensemble that predicts MIC values for 34 bacterial strains.
The AMP design integration preserves all strain-specific values and derives pathogen activity,
commensal activity, selectivity, and ensemble uncertainty. Commercial use and redistribution to
commercial third parties require separate permission from the Penn Center for Innovation.

## APEX-pathogen

- Source: <https://gitlab.com/machine-biology-group-public/apex-pathogen>
- Pinned revision: `417a4441a1e6ef8b10d2352e1c059622d5259f3a`
- License: MIT
- Local path: `external/apex-pathogen`

APEX-pathogen supplies an 8-model ensemble that predicts MIC values for 11 pathogens. The local
adapter preserves all pathogen-specific ensemble mean MIC values and reports disagreement on the
`log10(MIC)` scale together with a median pathogen activity score.
