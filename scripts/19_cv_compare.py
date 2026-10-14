"""
19_cv_compare.py - compare the three models on POOLED OUT-OF-FOLD predictions from the group K-fold
runs (10e_make_cv_splits.py + 11_train_model.py once per fold and model), next to a trivial-feature
baseline evaluated on exactly the same folds.

Every image is a test image exactly once, so the pooled AUC covers every scan family. Uncertainty is a
bootstrap over GROUPS (scan families), not over images. Group-level numbers (how many Healthy / PCOS
FAMILIES are called correctly) are the honest sample size: about 25 Healthy families, not 600 images.

  python .\\scripts\\19_cv_compare.py --runs_root .\\runs_cv --splits_root .\\reports\\cv --folds 5 ^
         --data_dir .\\dataset\\roi_center --out_dir .\\reports\\model_selection_cv

Layout it expects:  runs_cv\\fold<k>\\<model>\\seed<s>\\{preds_val.csv, preds_test.csv, config.json}
(i.e. 11_train_model.py run with --out_root .\\runs_cv\\fold<k> --splits .\\reports\\cv\\fold<k>\\splits.csv).

Selection rule (fixed in advance): highest pooled out-of-fold AUC; models within --tie (0.005) of the best
are tied and the one with the fewest parameters wins. The paired group-bootstrap difference is printed so
you can see whether the choice is supported.  --data_dir enables the trivial-feature baseline.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = ["resnet50", "efficientnet_b0", "densenet121"]


def load_module(fname, alias):
    p = Path(__file__).with_name(fname)
    if not p.exists():
        sys.exit(f"ERROR: {p} not found (keep the scripts together in one folder)")
    spec = importlib.util.spec_from_file_location(alias, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def resolve(data_dir, rel):
    q = Path(data_dir) / rel
    if q.exists():
        return str(q)
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG"):
        r = q.with_suffix(ext)
        if r.exists():
            return str(r)
    return str(q)


def group_level(df, prob_col, thr):
    """per-group mean probability -> is the family called correctly?"""
    g = df.groupby("group_id").agg(label=("label", "first"), p=(prob_col, "mean"), n=("label", "size"))
    g["correct"] = np.where(g.label == 1, g.p >= thr, g.p < thr)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--splits_root", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="training seed to read (seed<k> folder)")
    ap.add_argument("--data_dir", default=None, help="image folder for the trivial-feature baseline")
    ap.add_argument("--out_dir", default="reports/model_selection_cv")
    ap.add_argument("--tie", type=float, default=0.005)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--boot_seed", type=int, default=0)
    a = ap.parse_args()
    K = a.folds
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cm = load_module("12_compare_models.py", "cmpmod")
    rng = np.random.default_rng(a.boot_seed)

    # ------------------------------------------------------------ collect out-of-fold predictions
    oof, per_fold, params, missing = {}, [], {}, []
    for m in MODELS:
        frames = []
        for k in range(K):
            d = Path(a.runs_root) / f"fold{k}" / m / f"seed{a.seed}"
            if not ((d / "preds_test.csv").exists() and (d / "preds_val.csv").exists()):
                missing.append(str(d))
                continue
            te, va = pd.read_csv(d / "preds_test.csv"), pd.read_csv(d / "preds_val.csv")
            thr = cm.pick_threshold(va.label.values, va.prob_pcos.values, "val_f1")
            te = te.assign(fold=k, model=m, thr=thr)
            frames.append(te)
            row = {"model": m, "fold": k, "n_test": len(te), "thr": thr}
            if te.label.nunique() == 2:
                row["test_auc"] = roc_auc_score(te.label, te.prob_pcos)
            if va.label.nunique() == 2:
                row["val_auc"] = roc_auc_score(va.label, va.prob_pcos)
            per_fold.append(row)
            cfgp = d / "config.json"
            if cfgp.exists():
                params[m] = json.loads(cfgp.read_text()).get("n_params", np.nan)
        if frames:
            oof[m] = pd.concat(frames, ignore_index=True)
    if missing:
        print(f"WARNING: {len(missing)} run(s) missing, e.g. {missing[0]}")
    if not oof:
        sys.exit("ERROR: no runs found. Check --runs_root / --seed.")
    full = [m for m in oof if oof[m].fold.nunique() == K]
    if len(full) < len(oof):
        print(f"NOTE: models without all {K} folds are excluded from the comparison: "
              f"{[m for m in oof if m not in full]}")
    if not full:
        sys.exit("ERROR: no model has predictions for all folds yet.")
    oof = {m: oof[m] for m in full}

    # all models must cover the same images
    base = oof[full[0]].sort_values("path").reset_index(drop=True)
    y, groups = base.label.values.astype(int), base.group_id.values
    P = {m: oof[m].set_index("path").loc[base.path, "prob_pcos"].values for m in full}
    thr_img = {m: oof[m].set_index("path").loc[base.path, "thr"].values for m in full}
    nH, nP = int(((base.label == 0)).sum()), int((base.label == 1).sum())
    gH = base[base.label == 0].group_id.nunique()
    gP = base[base.label == 1].group_id.nunique()

    # ------------------------------------------------------------ trivial-feature baseline, same folds
    baseline = None
    if a.data_dir:
        from sklearn.ensemble import HistGradientBoostingClassifier
        sc = load_module("13_shortcut_check.py", "shortmod")
        feats_cache = {}
        pbase = np.full(len(base), np.nan)
        for k in range(K):
            sp = pd.read_csv(Path(a.splits_root) / f"fold{k}" / "splits.csv")
            sp = sp[sp.split.isin(["train", "test"])]
            for p in sp.path:
                if p not in feats_cache:
                    f = sc.feats(resolve(a.data_dir, p))
                    feats_cache[p] = f
            ok = sp[[feats_cache[p] is not None for p in sp.path]]
            X = pd.DataFrame([feats_cache[p] for p in ok.path])
            tr, te = (ok.split == "train").values, (ok.split == "test").values
            gb = HistGradientBoostingClassifier(max_depth=3, max_iter=150, random_state=0)
            gb.fit(X[tr], ok.label.values[tr])
            pr = pd.Series(gb.predict_proba(X[te])[:, 1], index=ok.path[te].values)
            idx = base.index[base.path.isin(pr.index)]
            pbase[idx] = pr.loc[base.path[idx]].values
        if np.isnan(pbase).any():
            print("WARNING: baseline missing for some images; they are dropped from the baseline numbers.")
        baseline = pbase

    # ------------------------------------------------------------ metrics
    def boot_ci(p):
        ok = ~np.isnan(p)
        return cm.boot_auc_ci(y[ok], p[ok], groups[ok], a.n_boot, rng)

    rows = []
    for m in full:
        p = P[m]
        pred = (p >= thr_img[m]).astype(int)
        tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tp = int(((pred == 1) & (y == 1)).sum())
        lo, hi = boot_ci(p)
        tmp = base.assign(p=p)
        gl = group_level(tmp, "p", 0.5)
        pf = pd.DataFrame([r for r in per_fold if r["model"] == m])
        rows.append({
            "model": m, "params_M": params.get(m, np.nan) / 1e6,
            "oof_auc": roc_auc_score(y, p), "ci_lo": lo, "ci_hi": hi,
            "fold_auc_mean": pf.test_auc.mean(), "fold_auc_min": pf.test_auc.min(),
            "val_auc_mean": pf.val_auc.mean() if "val_auc" in pf else np.nan,
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "balanced_acc": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
            "healthy_groups_ok": f"{int(gl[gl.label == 0].correct.sum())}/{gH}",
            "pcos_groups_ok": f"{int(gl[gl.label == 1].correct.sum())}/{gP}",
        })
    R = pd.DataFrame(rows).sort_values("oof_auc", ascending=False).reset_index(drop=True)
    R.to_csv(out / "model_comparison_cv.csv", index=False)
    pd.DataFrame(per_fold).to_csv(out / "per_fold_metrics.csv", index=False)
    extra = {f"p_{m}": P[m] for m in full}
    if baseline is not None:
        extra["p_trivial_baseline"] = baseline
    base.assign(**extra).to_csv(out / "oof_predictions.csv", index=False)

    best_auc = R.oof_auc.max()
    tied = R[R.oof_auc >= best_auc - a.tie]
    sel = tied.sort_values(["params_M", "oof_auc"], ascending=[True, False]).iloc[0].model
    diffs = {}
    for m in full:
        if m != sel:
            diffs[m] = cm.boot_auc_diff(y, P[sel], P[m], groups, a.n_boot, rng)

    vs_base = None
    if baseline is not None:
        okb = ~np.isnan(baseline)
        vs_base = cm.boot_auc_diff(y[okb], P[sel][okb], baseline[okb], groups[okb], a.n_boot, rng)

    # ------------------------------------------------------------ report
    L = ["# Model comparison - group K-fold cross-validation", ""]
    L.append(f"{K} folds over {len(base)} images in {gH + gP} scan families "
             f"(Healthy {nH} images / **{gH} families**, PCOS {nP} images / {gP} families). "
             f"Every image is tested once; AUC is pooled over all out-of-fold predictions; "
             f"CI = 95% bootstrap over families.")
    L.append("")
    if baseline is not None:
        ok = ~np.isnan(baseline)
        blo, bhi = boot_ci(baseline)
        bauc = roc_auc_score(y[ok], baseline[ok])
        best_cnn = R.oof_auc.max()
        L.append(f"> **Shortcut baseline.** A gradient-boosting model on 23 trivial image statistics "
                 f"(brightness, contrast, black margins, tissue extent, file size; no anatomy), evaluated on "
                 f"the SAME folds, reaches pooled AUC **{bauc:.3f}** [{blo:.3f}, {bhi:.3f}]; the best CNN "
                 f"reaches {best_cnn:.3f} (difference {best_cnn - bauc:+.3f}). "
                 f"{'The baseline is high, so CNN scores do not show that pathology was learned.' if bauc >= 0.8 else 'The baseline is moderate; the CNN gain over it is the better indicator of learning.'}")
        L.append("")
    L.append("| Model | Params (M) | Pooled OOF AUC [95% CI] | Per-fold AUC mean (min) | Sens. | Spec. | Bal. acc. | Healthy families correct | PCOS families correct |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in R.iterrows():
        mark = " **(selected)**" if r.model == sel else ""
        L.append(f"| {r.model}{mark} | {r.params_M:.1f} | {r.oof_auc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}] | "
                 f"{r.fold_auc_mean:.3f} ({r.fold_auc_min:.3f}) | {r.sensitivity:.3f} | {r.specificity:.3f} | "
                 f"{r.balanced_acc:.3f} | {r.healthy_groups_ok} | {r.pcos_groups_ok} |")
    if baseline is not None:
        L.append(f"| *trivial-feature baseline* | - | {bauc:.3f} [{blo:.3f}, {bhi:.3f}] | - | - | - | - | - | - |")
    L += ["", "Sens./Spec./Bal. acc. use, for each fold, the threshold that maximised F1 on that fold's validation "
          "split. 'Families correct' averages P(PCOS) within a scan family and thresholds at 0.5.", "",
          "## Selection", "",
          f"Rule: highest pooled out-of-fold AUC; models within {a.tie} of the best are tied and the smallest wins.",
          f"Tied with the best: {', '.join(tied.model)}.  **Selected: {sel}**", ""]
    for m, (d, lo, hi) in diffs.items():
        verdict = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0 (no demonstrated difference)"
        L.append(f"- paired group-bootstrap AUC difference, {sel} minus {m}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] - {verdict}")
    if vs_base is not None:
        d, lo, hi = vs_base
        verdict = ("CI excludes 0: the CNN beats the trivial-feature baseline" if (lo > 0 or hi < 0)
                   else "CI includes 0: the CNN is NOT shown to beat the trivial-feature baseline")
        L.append(f"- paired group-bootstrap AUC difference, {sel} minus trivial-feature baseline: "
                 f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}] - {verdict}")
    L += ["", "Out-of-fold selection among three candidates is slightly optimistic for the winner; no separate "
          "outer test set remains. A different data source is still needed to show the models generalise."]
    (out / "model_comparison_cv.md").write_text("\n".join(L), encoding="utf-8")
    sel_json = {"selected_model": sel, "rule": "highest pooled OOF AUC; ties within tie -> fewest params",
                "pooled_oof_auc": {r.model: round(r.oof_auc, 4) for _, r in R.iterrows()},
                "n_healthy_families": gH, "n_pcos_families": gP, "folds": K,
                "shortcut_baseline_oof_auc": None if baseline is None else round(float(bauc), 4),
                "selected_minus_baseline_auc": None if vs_base is None else
                {"diff": round(vs_base[0], 4), "ci_lo": round(vs_base[1], 4), "ci_hi": round(vs_base[2], 4)}}
    (out / "selected_model_cv.json").write_text(json.dumps(sel_json, indent=2))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for m in full:
        fpr, tpr, _ = roc_curve(y, P[m])
        ax.plot(fpr, tpr, label=f"{m} ({roc_auc_score(y, P[m]):.3f})")
    if baseline is not None:
        ok = ~np.isnan(baseline)
        fpr, tpr, _ = roc_curve(y[ok], baseline[ok])
        ax.plot(fpr, tpr, "k--", label=f"trivial features ({bauc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity"); ax.set_title("Pooled out-of-fold ROC")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "roc_oof.png", dpi=130); plt.close(fig)

    print("\n".join(L))
    print(f"\nFiles written to {out}")


if __name__ == "__main__":
    main()
