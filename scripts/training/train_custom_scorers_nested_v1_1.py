#!/usr/bin/env python3
"""Train audit-ready ESM2-t30 toxicity/hemolysis scorers with nested CV."""
from __future__ import annotations

import hashlib, json, pickle, platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs/custom_scorers_stage_a_v1"
OUT = ROOT / "outputs/custom_scorers_stage_b2_nested_v1_2"
CACHE_ROOT = ROOT / "outputs/custom_scorers_stage_b2_nested_v1"
MODEL = ROOT / "external/models/facebook/esm2_t30_150M_UR50D"
SEEDS = (42, 123, 2025)
CS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cluster_map(path: Path) -> dict[str, str]:
    t = pd.read_csv(path, sep="\t", header=None, names=["rep", "member"])
    return dict(zip(t.member.astype(str), t.rep.astype(str)))


def load_task(task: str) -> pd.DataFrame:
    if task == "toxicity":
        d = pd.read_csv(DATA / "toxicity_normalized.csv")
        d = d[(d.split == "train") & d.canonical].copy()
        cmap = cluster_map(DATA / "mmseqs/toxicity_train_cluster.tsv")
    else:
        d = pd.read_csv(DATA / "hemolysis_normalized.csv")
        d = d[(d.split == "cross_val") & d.canonical].copy()
        cmap = cluster_map(DATA / "mmseqs/hemolysis_cross_val_cluster.tsv")
    d["group"] = d.sequence_sha256.map(cmap)
    if d.group.isna().any():
        raise ValueError(f"Missing MMseqs2 groups for {task}")
    return d.reset_index(drop=True)


def embed(sequences: list[str], device_name: str) -> np.ndarray:
    keys = np.array([hashlib.sha256(s.encode()).hexdigest() for s in sequences])
    cache = CACHE_ROOT / ("embeddings_" + hashlib.sha256("|".join(keys).encode()).hexdigest()[:16] + ".npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if np.array_equal(z["keys"], keys):
            return z["X"]
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    enc = EsmModel.from_pretrained(MODEL, local_files_only=True).to(device).eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(sequences), 32):
            ss = sequences[start:start + 32]
            batch = {k: v.to(device) for k, v in tok(ss, return_tensors="pt", padding=True).items()}
            h = enc(**batch).last_hidden_state
            mask = batch["attention_mask"].bool()
            # ESM2 uses BOS at position 0 and EOS immediately after each sequence.
            mask[:, 0] = False
            for j, s in enumerate(ss):
                mask[j, len(s) + 1] = False
                rows.append(h[j][mask[j]].mean(0).cpu().numpy())
    X = np.asarray(rows, dtype=np.float32)
    np.savez_compressed(cache, keys=keys, X=X)
    return X


def fit_pipeline(X: np.ndarray, y: np.ndarray, C: float, class_weight: str | None) -> Pipeline:
    p = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(
        C=C, class_weight=class_weight, max_iter=5000, solver="lbfgs"))])
    p.fit(X, y)
    return p


def select_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float, float]:
    """Choose highest-specificity OOF threshold with sensitivity >= 0.95."""
    candidates = []
    for threshold in np.unique(score[y == 1]):
        sensitivity = float((score[y == 1] >= threshold).mean())
        if sensitivity >= 0.95:
            specificity = float((score[y == 0] < threshold).mean())
            candidates.append((specificity, -float(threshold), float(threshold), sensitivity))
    if not candidates:
        raise RuntimeError("No OOF threshold reaches sensitivity >= 0.95")
    specificity, _, threshold, sensitivity = max(candidates)
    return threshold, sensitivity, specificity


def select_inner(X, y, groups, train_idx, seed, task):
    inner = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
    candidates = [(C, cw) for C in CS for cw in ((None, "balanced") if task == "hemolysis" else (None,))]
    scores = []
    for C, cw in candidates:
        vals = []
        Xi, yi, gi = X[train_idx], y[train_idx], groups[train_idx]
        for it, iv in inner.split(Xi, yi, gi):
            p = fit_pipeline(Xi[it], yi[it], C, cw)
            vals.append(average_precision_score(yi[iv], p.predict_proba(Xi[iv])[:, 1]))
        scores.append((float(np.mean(vals)), -C, 0 if cw is None else -1, C, cw))
    return max(scores)[3:]


def run_task(task: str, X: np.ndarray, d: pd.DataFrame):
    y, groups = d.label.to_numpy(int), d.group.to_numpy(str)
    per_seed = []
    main_choices = []
    for seed in SEEDS:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        oof = np.full(len(d), np.nan)
        choices = []
        for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
            C, cw = select_inner(X, y, groups, tr, seed + 1000 + fold, task)
            p = fit_pipeline(X[tr], y[tr], C, cw)
            oof[va] = p.predict_proba(X[va])[:, 1]
            choices.append({"fold": fold, "C": C, "class_weight": cw})
        threshold, sens, spec = select_threshold(y, oof)
        auc = float(roc_auc_score(y, oof)); auprc = float(average_precision_score(y, oof))
        per_seed.append({"seed": seed, "n": len(d), "auroc": auc, "auprc": auprc,
                         "threshold": threshold, "sensitivity": sens, "specificity": spec,
                         "fold_choices": choices})
        if seed == 42:
            main_choices = choices
            np.savez_compressed(OUT / f"{task}_oof_seed{seed}.npz", y=y, score=oof)
        else:
            np.savez_compressed(OUT / f"{task}_oof_seed{seed}.npz", y=y, score=oof)
    counts = {}
    for c in main_choices:
        counts[(c["C"], c["class_weight"])] = counts.get((c["C"], c["class_weight"]), 0) + 1
    modal = sorted(counts.items(), key=lambda x: (-x[1], x[0][0], str(x[0][1])))[0][0]
    full = fit_pipeline(X, y, modal[0], modal[1])
    with (OUT / f"{task}_model.pkl").open("wb") as f: pickle.dump(full, f)
    by_seed = {row["seed"]: row for row in per_seed}
    prevalence = float(y.mean())
    main = by_seed[42]
    development_gate = (
        main["auroc"] >= .75
        and main["auprc"] >= prevalence + .10
        and main["sensitivity"] >= .95
        and main["specificity"] >= .30
        and all(
            by_seed[seed]["auroc"] >= .70
            and by_seed[seed]["sensitivity"] >= .95
            and by_seed[seed]["specificity"] >= .20
            for seed in (123, 2025)
        )
    )
    return {"task": task, "n": len(d), "seeds": per_seed,
            "modal_main_setting": {"C": modal[0], "class_weight": modal[1]},
            "positive_prevalence": prevalence,
            "development_gate_candidate": development_gate,
            "formal_gate_candidate": None,
            "formal_gate_note": "Requires frozen-test and de-homologized evaluation artifact"}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = {"script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)),
                "seeds": list(SEEDS), "C_grid": list(CS), "model": str(MODEL),
                "model_config_sha256": sha256(MODEL / "config.json"), "device": "cuda" if torch.cuda.is_available() else "cpu",
                "input_hashes": {f: sha256(DATA / f) for f in ["toxicity_normalized.csv", "hemolysis_normalized.csv"]}}
    (OUT / "pretrain_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tox, he = load_task("toxicity"), load_task("hemolysis")
    all_seq = pd.unique(pd.concat([tox.sequence, he.sequence], ignore_index=True)).tolist()
    Xall = embed(all_seq, manifest["device"]); lookup = {s: Xall[i] for i, s in enumerate(all_seq)}
    results = {}
    for task, d in (("toxicity", tox), ("hemolysis", he)):
        X = np.vstack([lookup[s] for s in d.sequence])
        X = np.column_stack([X, d.sequence.str.len().to_numpy(float)])
        results[task] = run_task(task, X, d)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
