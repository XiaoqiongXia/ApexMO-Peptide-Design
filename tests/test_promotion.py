from amp_design.promotion import evaluate_safety_promotion


def test_sensitivity_seeds_use_prespecified_point_two_specificity_gate() -> None:
    development = {
        "seeds": [
            {"seed": 42, "auroc": .8, "auprc": .7, "sensitivity": .95, "specificity": .3},
            {"seed": 123, "auroc": .7, "auprc": .7, "sensitivity": .95, "specificity": .21},
            {"seed": 2025, "auroc": .7, "auprc": .7, "sensitivity": .95, "specificity": .2},
        ]
    }
    frozen = {"auroc": .75, "auprc": .7, "sensitivity": .9, "specificity": .2}
    dehomologized = {"n": 10, "auroc": .7, "auprc": .6, "sensitivity": .9, "specificity": .2}
    result = evaluate_safety_promotion(
        development, frozen, prevalence=.5, dehomologized=dehomologized
    )
    assert result["formal_gate_pass"]


def test_missing_dehomologized_evaluation_blocks_formal_promotion() -> None:
    development = {
        "seeds": [
            {"seed": seed, "auroc": .9, "auprc": .8, "sensitivity": .96, "specificity": .4}
            for seed in (42, 123, 2025)
        ]
    }
    frozen = {"auroc": .85, "auprc": .8, "sensitivity": .95, "specificity": .4}
    result = evaluate_safety_promotion(
        development, frozen, prevalence=.5, dehomologized=None
    )
    assert result["all_numeric_gates_pass"]
    assert not result["formal_gate_pass"]
