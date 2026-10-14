"""
12_compare_models.py

Model-selection step 3 of 3: compare the trained candidates and select one.

Reads every  <runs_root>/<model>/seed<k>/  produced by 11_train_model.py and
writes, to --out_dir:
    model_comparison.md / .csv   headline table + per-split metrics
    selected_model.json          the choice, the rule, and the numbers behind it
    roc_test.png                 ROC curves on the test split
    confusion_matrices_test.png  confusion matrices on the test split
    training_curves.png          validation AUC per epoch

How the choice is made (fixed BEFORE looking at test results):
  1. The decision uses VALIDATION data only: mean validation AUC over seeds.
  2. Any model within --tie_margin (default 0.005 AUC) of the best is
     treated as tied; among tied models the one with FEWEST PARAMETERS wins
     (simpler, faster, cheaper to explain - and a difference that small is
     smaller than normal seed-to-seed noise).
  3. The test split is reported for every model but never used to choose.
     Confidence intervals come from a GROUP bootstrap (whole near-duplicate
     groups are resampled together), so they are not artificially narrow.
  4. Classification threshold: chosen on validation (maximise F1, or fixed
     0.5 with --threshold_mode fixed) and then applied unchanged to test.

Sanity checks: refuses to compare runs trained on different splits.csv files
(identical md5 required) unless --allow_mixed_splits is given.

Usage
    python 12_compare_models.py --runs_root ../runs --out_dir ../reports/model_selection
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

SPLITS = ["val", "test", "external"]
METRICS = ["auc", "accuracy", "precision", "recall", "specificity", "f1"]


# ------------------------------------------------------------------ metrics
def safe_div(a, b):
    return float(a) / float(b) if b else float("nan")


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec, rec = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec) if not (np.isnan(prec) or np.isnan(rec)) else float("nan")
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return {"auc": auc, "accuracy": safe_div(tp + tn, len(y)), "precision": prec, "recall": rec,
            "specificity": safe_div(tn, tn + fp), "f1": f1,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "threshold": float(thr)}


def pick_threshold(y, p, mode):
    if mode == "fixed":
        return 0.5
    grid = np.linspace(0.05, 0.95, 91)
    scores = []
    for t in grid:
        f1 = metrics_at(y, p, t)["f1"]
        scores.append(-1.0 if np.isnan(f1) else f1)
    scores = np.array(scores)
    best = np.where(scores >= scores.max() - 1e-12)[0]
    return float(grid[best[np.argmin(np.abs(grid[best] - 0.5))]])  # ties -> closest to 0.5


# ------------------------------------------------------------------ loading
def load_runs(root):
    runs = []
    for cfg_path in sorted(Path(root).glob("*/seed*/config.json")):
        d = cfg_path.parent
        cfg = json.loads(cfg_path.read_text())
        preds = {s: pd.read_csv(d / f"preds_{s}.csv") for s in SPLITS if (d / f"preds_{s}.csv").exists()}
        if "val" in preds and "test" in preds:
            runs.append({"model": cfg["model"], "seed": cfg["seed"], "cfg": cfg, "preds": preds, "dir": d})
        else:
            print(f"skipping incomplete run {d}")
    return runs


def seed_ensemble(model_runs, split):
    """Average P(PCOS) over seeds, aligned by image path."""
    frames = [r["preds"][split].set_index("path")["prob_pcos"].rename(f"s{r['seed']}") for r in model_runs
              if split in r["preds"]]
    if not frames:
        return None
    base = model_runs[0]["preds"][split].set_index("path")[["label", "group_id"]]
    probs = pd.concat(frames, axis=1, join="inner")
    out = base.loc[probs.index].copy()
    out["prob"] = probs.mean(axis=1)
    return out.reset_index()


# --------------------------------------------------------- group bootstrap
def boot_samples(groups, n_boot, rng):
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[k] for k in pick])


def boot_auc_ci(y, p, groups, n_boot, rng):
    vals = []
    for idx in boot_samples(groups, n_boot, rng):
        if len(np.unique(y[idx])) > 1:
            vals.append(roc_auc_score(y[idx], p[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def boot_auc_diff(y, pa, pb, groups, n_boot, rng):
    diffs = []
    for idx in boot_samples(groups, n_boot, rng):
        if len(np.unique(y[idx])) > 1:
            diffs.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    if not diffs:
        return (float("nan"),) * 3
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--out_dir", default="reports/model_selection")
    ap.add_argument("--threshold_mode", choices=["val_f1", "fixed"], default="val_f1")
    ap.add_argument("--tie_margin", type=float, default=0.005)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow_mixed_splits", action="store_true")
    ap.add_argument("--shortcut_auc", type=float, default=None,
                    help="Test AUC of the trivial-feature baseline from 13_shortcut_check.py (e.g. the "
                         "gradient-boosting test AUC). If given, it is printed as a warning banner in the "
                         "report so CNN scores are never read without it.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    runs = load_runs(args.runs_root)
    if not runs:
        raise SystemExit(f"No completed runs found under {args.runs_root}/<model>/seed*/")
    md5s = {r["cfg"].get("splits_md5") for r in runs}
    if len(md5s) > 1 and not args.allow_mixed_splits:
        raise SystemExit("ERROR: runs were trained on DIFFERENT splits.csv files "
                         f"({len(md5s)} versions) - results are not comparable. Re-train, or pass --allow_mixed_splits.")
    models = sorted({r["model"] for r in runs})
    by_model = {m: [r for r in runs if r["model"] == m] for m in models}
    print({m: [r["seed"] for r in rs] for m, rs in by_model.items()})
    if len({len(v) for v in by_model.values()}) > 1:
        print("WARNING: models have different numbers of seeds - comparison is less fair.")

    # ---- per-run metrics (threshold from that run's own validation predictions)
    rows = []
    for r in runs:
        yv, pv = r["preds"]["val"]["label"].to_numpy(), r["preds"]["val"]["prob_pcos"].to_numpy()
        thr = pick_threshold(yv, pv, args.threshold_mode)
        for s, df in r["preds"].items():
            met = metrics_at(df["label"].to_numpy(), df["prob_pcos"].to_numpy(), thr)
            rows.append({"model": r["model"], "seed": r["seed"], "split": s, **met,
                         "n_params": r["cfg"]["n_params"], "best_epoch": r["cfg"]["best_epoch"],
                         "train_min": r["cfg"]["train_seconds"] / 60})
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / "per_run_metrics.csv", index=False)

    agg = (per_run.groupby(["model", "split"])[METRICS + ["threshold"]]
           .agg(["mean", "std"]).round(4))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out / "model_comparison.csv", index=False)

    # ---- selection on validation only
    val = per_run[per_run.split == "val"].groupby("model")["auc"].agg(["mean", "std"]).fillna(0.0)
    params = {m: int(by_model[m][0]["cfg"]["n_params"]) for m in models}
    best_auc = float(val["mean"].max())
    tied = [m for m in models if val.loc[m, "mean"] >= best_auc - args.tie_margin]
    selected = sorted(tied, key=lambda m: (params[m], -val.loc[m, "mean"]))[0]
    top = val["mean"].idxmax()
    noise = float(max(val.loc[m, "std"] for m in tied)) if len(tied) > 1 else 0.0
    reason = (f"'{selected}' has the highest validation AUC." if len(tied) == 1 else
              f"{len(tied)} models are within {args.tie_margin} validation AUC of the best ('{top}', "
              f"{best_auc:.4f}); '{selected}' has the fewest parameters among them.")
    if len(tied) > 1 and noise >= args.tie_margin:
        reason += (" Seed-to-seed std of the tied models "
                   f"({noise:.4f}) is as large as the margin, so they are statistically indistinguishable "
                   "on validation - say so in the thesis rather than claiming a winner.")

    # ---- test-set reporting (seed-ensembled probabilities, group bootstrap)
    ens = {m: seed_ensemble(by_model[m], "test") for m in models}
    yt = ens[models[0]]["label"].to_numpy()
    gt = ens[models[0]]["group_id"].to_numpy()
    ci, diffs, ens_auc = {}, {}, {}
    for m in models:
        e = ens[m].set_index("path").loc[ens[models[0]]["path"]]
        ci[m] = boot_auc_ci(yt, e["prob"].to_numpy(), gt, args.n_boot, rng)
        ens_auc[m] = float(roc_auc_score(yt, e["prob"].to_numpy()))
    pa = ens[selected].set_index("path").loc[ens[models[0]]["path"]]["prob"].to_numpy()
    for m in models:
        if m == selected:
            continue
        pb = ens[m].set_index("path").loc[ens[models[0]]["path"]]["prob"].to_numpy()
        diffs[m] = boot_auc_diff(yt, pa, pb, gt, args.n_boot, rng)

    sel = {"selected_model": selected, "rule": reason, "selection_split": "validation",
           "threshold_mode": args.threshold_mode, "tie_margin": args.tie_margin,
           "validation_auc_mean_std": {m: [round(float(val.loc[m, "mean"]), 4), round(float(val.loc[m, "std"]), 4)]
                                       for m in models},
           "n_params": params, "n_seeds": {m: len(by_model[m]) for m in models},
           "test_auc_seed_avg": {m: round(ens_auc[m], 4) for m in models},
           "test_auc_ci95_group_bootstrap": {m: [round(ci[m][0], 4), round(ci[m][1], 4)] for m in models},
           "test_auc_difference_selected_minus_other": {m: {"mean": round(d[0], 4), "ci95": [round(d[1], 4), round(d[2], 4)],
                                                            "ci_excludes_zero": bool(d[1] > 0 or d[2] < 0)}
                                                        for m, d in diffs.items()}}
    caution = None
    if args.shortcut_auc is not None:
        best_cnn = max(ens_auc.values())
        gap = best_cnn - args.shortcut_auc
        caution = (f"Trivial-feature baseline (brightness, black margins, tissue extent, file size; no "
                   f"anatomy) reaches test AUC {args.shortcut_auc:.3f}; the best CNN reaches {best_cnn:.3f} "
                   f"(difference {gap:+.3f}). ")
        if args.shortcut_auc >= 0.85:
            caution += ("The baseline is already very high, so these CNN scores do NOT show that the models "
                        "learned ovarian pathology: the classes are separable by image characteristics alone "
                        "(different sources/framing). Do not interpret differences between models as clinical "
                        "performance.")
        elif gap < 0.05:
            caution += "The CNNs barely beat a baseline that uses no anatomy; interpret with caution."
        else:
            caution += "The CNNs clearly beat the no-anatomy baseline."
        sel["shortcut_baseline_test_auc"] = round(args.shortcut_auc, 4)
        sel["caution"] = caution
    (out / "selected_model.json").write_text(json.dumps(sel, indent=2))

    # ---- markdown report
    def ms(m, s, k):
        a = agg[(agg.model == m) & (agg.split == s)]
        if a.empty:
            return "-"
        return f"{a[k + '_mean'].iloc[0]:.3f} ± {np.nan_to_num(a[k + '_std'].iloc[0]):.3f}"

    lines = ["# Model comparison", ""]
    if caution:
        lines += [f"> **WARNING - read before using these numbers.** {caution}", ""]
    lines += [f"Seeds per model: {', '.join(f'{m}={len(by_model[m])}' for m in models)}  ",
             f"Threshold: {args.threshold_mode} (chosen on validation, applied unchanged to test).  ",
             "Per-seed columns are mean ± std over seeds (threshold-dependent metrics use each seed's own validation-chosen threshold). "
             "'Seed-avg' = AUC of the probabilities averaged over seeds, with a 95% group-bootstrap CI.", "",
             "| Model | Params (M) | Val AUC | Test AUC per seed | Test AUC seed-avg [95% CI] | Test acc | Test precision | Test recall | Test specificity | Test F1 | Best epoch | Train min |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        pr = per_run[(per_run.model == m) & (per_run.split == "test")]
        lines.append(f"| {m}{' **(selected)**' if m == selected else ''} | {params[m] / 1e6:.1f} | {ms(m, 'val', 'auc')} | "
                     f"{ms(m, 'test', 'auc')} | {ens_auc[m]:.3f} [{ci[m][0]:.3f}, {ci[m][1]:.3f}] | {ms(m, 'test', 'accuracy')} | "
                     f"{ms(m, 'test', 'precision')} | {ms(m, 'test', 'recall')} | {ms(m, 'test', 'specificity')} | "
                     f"{ms(m, 'test', 'f1')} | {pr['best_epoch'].mean():.0f} | {pr['train_min'].mean():.1f} |")
    if (per_run.split == "external").any():
        lines += ["", "## External (independent dataset)", "", "| Model | AUC | Accuracy | Precision | Recall | Specificity | F1 |",
                  "|---|---|---|---|---|---|---|"]
        for m in models:
            lines.append(f"| {m} | " + " | ".join(ms(m, "external", k) for k in METRICS) + " |")
    lines += ["", "## Selection", "", reason, "",
              "Paired group-bootstrap difference in test AUC (selected minus other):", ""]
    for m, d in diffs.items():
        flag = "CI excludes 0" if (d[1] > 0 or d[2] < 0) else "CI includes 0 (no demonstrated difference)"
        lines.append(f"- vs {m}: {d[0]:+.4f}  [{d[1]:+.4f}, {d[2]:+.4f}]  - {flag}")
    (out / "model_comparison.md").write_text("\n".join(lines) + "\n")

    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for m in models:
        e = ens[m]
        fpr, tpr, _ = roc_curve(e["label"], e["prob"])
        ax.plot(fpr, tpr, label=f"{m} (AUC {roc_auc_score(e['label'], e['prob']):.3f})", lw=2 if m == selected else 1.3)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set(xlabel="1 - specificity", ylabel="sensitivity (recall)", title="Test ROC (seed-averaged)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "roc_test.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.4))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        thr = float(per_run[(per_run.model == m) & (per_run.split == "val")]["threshold"].mean())
        e = ens[m]
        cm = confusion_matrix(e["label"], (e["prob"] >= thr).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Healthy", "PCOS"], yticklabels=["Healthy", "PCOS"],
               xlabel="predicted", ylabel="true", title=f"{m}\nthr={thr:.2f}")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrices_test.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    for m in models:
        hs = [pd.read_csv(r["dir"] / "history.csv").set_index("epoch")["val_auc"] for r in by_model[m]
              if (r["dir"] / "history.csv").exists()]
        if hs:
            ax.plot(pd.concat(hs, axis=1).mean(axis=1), label=m)
    ax.set(xlabel="epoch", ylabel="validation AUC", title="Training curves (mean over seeds)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "training_curves.png", dpi=150)
    plt.close()

    print("\n".join(lines))
    print(f"\nSELECTED: {selected}\nFiles written to {out}")


if __name__ == "__main__":
    main()
