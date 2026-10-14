"""
10e_make_cv_splits.py - group-aware K-fold splits, so EVERY scan family is a test image exactly once.

Why: with ~25 independent Healthy scan families in total, any single train/val/test split leaves only
1-3 Healthy families in the test set, which is too few to say anything. Cross-validation over groups
uses all of them: each fold tests on ~1/K of the groups, and the pooled out-of-fold predictions cover
every group.

Input: a splits.csv from 10_make_splits.py (its group_id column and the kept images are reused; the
old train/val/test column is ignored). Whole groups go to one fold; Healthy and PCOS groups are spread
over the folds separately so every fold has Healthy groups.

For fold k:  test = fold k,  val = fold (k+1) mod K,  train = the other folds.

  python .\\scripts\\10e_make_cv_splits.py --splits .\\reports\\splits_v3\\splits.csv --folds 5 ^
         --out_dir .\\reports\\cv

Writes reports\\cv\\fold0\\splits.csv ... foldK-1\\splits.csv and folds.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out_dir", default="reports/cv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    K = a.folds
    if K < 3:
        sys.exit("ERROR: --folds must be >= 3 (train, val and test are different folds)")

    df = pd.read_csv(a.splits)
    df = df[df.split.isin(["train", "val", "test"])].reset_index(drop=True)
    if df.groupby("group_id")["label"].nunique().max() > 1:
        sys.exit("ERROR: a group contains both classes; rebuild splits with 10_make_splits.py (class-aware grouping)")
    G = df.groupby("group_id").agg(label=("label", "first"), n=("path", "count")).reset_index()
    rng = np.random.default_rng(a.seed)

    fold_of = {}
    for lab, g in G.groupby("label"):
        if len(g) < K:
            print(f"WARNING: class {lab} has only {len(g)} groups for {K} folds; some folds will have none of it")
        g = g.assign(tie=rng.random(len(g))).sort_values(["n", "tie"], ascending=[False, True])
        load = np.zeros((K, 2))  # images, groups per fold for this class
        for gid, n in zip(g.group_id, g.n):
            k = min(range(K), key=lambda f: (load[f, 0], load[f, 1], f))
            fold_of[gid] = k
            load[k] += (n, 1)
    df["fold"] = df.group_id.map(fold_of)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df[["path", "label", "group_id", "fold"]].to_csv(out / "folds.csv", index=False)

    rows, warn = [], False
    for k in range(K):
        d = df.copy()
        d["split"] = np.where(d.fold == k, "test", np.where(d.fold == (k + 1) % K, "val", "train"))
        (out / f"fold{k}").mkdir(exist_ok=True)
        d.drop(columns=["fold"]).to_csv(out / f"fold{k}" / "splits.csv", index=False)
        for s in ("train", "val", "test"):
            for lab, name in ((0, "Healthy"), (1, "PCOS")):
                m = d[(d.split == s) & (d.label == lab)]
                rows.append({"fold": k, "split": s, "class": name, "images": len(m), "groups": m.group_id.nunique()})
    t = pd.DataFrame(rows)
    print(f"{len(df)} images, {df.group_id.nunique()} groups -> {K} folds\n")
    print("Test split of each fold (images / groups):")
    piv = t[t.split == "test"].pivot(index="fold", columns="class", values=["images", "groups"])
    print(piv.to_string())
    print("\nValidation split of each fold (images / groups):")
    print(t[t.split == "val"].pivot(index="fold", columns="class", values=["images", "groups"]).to_string())
    low = t[(t.split.isin(["val", "test"])) & (t["class"] == "Healthy") & (t.groups < 2)]
    if len(low):
        print("\nWARNING: some val/test splits have fewer than 2 Healthy groups; try fewer --folds.")
    print(f"\nWrote {out}\\fold0..fold{K - 1}\\splits.csv")


if __name__ == "__main__":
    main()
