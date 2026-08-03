"""Frozen promotion-gate evaluation for custom safety scorers."""

from __future__ import annotations

from typing import Any


def evaluate_safety_promotion(
    development: dict[str, Any],
    frozen_test: dict[str, Any],
    *,
    prevalence: float,
    dehomologized: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate every prespecified classification promotion requirement."""

    by_seed = {int(row["seed"]): row for row in development["seeds"]}
    if set(by_seed) != {42, 123, 2025}:
        raise ValueError("Promotion gate requires development seeds 42, 123, and 2025")
    main = by_seed[42]
    checks = {
        "main_auroc_ge_0_75": float(main["auroc"]) >= 0.75,
        "main_auprc_ge_prevalence_plus_0_10": float(main["auprc"]) >= prevalence + 0.10,
        "main_sensitivity_ge_0_95": float(main["sensitivity"]) >= 0.95,
        "main_specificity_ge_0_30": float(main["specificity"]) >= 0.30,
        "frozen_auroc_ge_0_70": float(frozen_test["auroc"]) >= 0.70,
        "frozen_auroc_drop_le_0_10": float(frozen_test["auroc"]) >= float(main["auroc"]) - 0.10,
        "frozen_sensitivity_ge_0_90": float(frozen_test["sensitivity"]) >= 0.90,
        "frozen_specificity_ge_0_20": float(frozen_test["specificity"]) >= 0.20,
        "seed123_auroc_ge_0_70": float(by_seed[123]["auroc"]) >= 0.70,
        "seed123_sensitivity_ge_0_95": float(by_seed[123]["sensitivity"]) >= 0.95,
        "seed123_specificity_ge_0_20": float(by_seed[123]["specificity"]) >= 0.20,
        "seed2025_auroc_ge_0_70": float(by_seed[2025]["auroc"]) >= 0.70,
        "seed2025_sensitivity_ge_0_95": float(by_seed[2025]["sensitivity"]) >= 0.95,
        "seed2025_specificity_ge_0_20": float(by_seed[2025]["specificity"]) >= 0.20,
        "dehomologized_evaluation_present": dehomologized is not None,
    }
    if dehomologized is not None:
        required = {"n", "auroc", "auprc", "sensitivity", "specificity"}
        if missing := required.difference(dehomologized):
            raise ValueError(f"De-homologized evaluation lacks fields: {sorted(missing)}")
    return {
        "prevalence": prevalence,
        "main_seed": 42,
        "checks": checks,
        "all_numeric_gates_pass": all(
            value for name, value in checks.items() if name != "dehomologized_evaluation_present"
        ),
        "formal_gate_pass": all(checks.values()),
        "dehomologized_metrics": dehomologized,
    }
