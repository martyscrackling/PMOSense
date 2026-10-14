"""
10d_design_sweep.py - tabulate what each (grouping threshold, conflict threshold, per-group cap)
would leave you with, using the saved pair_scores.csv. Takes seconds, changes nothing.

Same rules as 10_make_splits.py:
  * links between different classes are never used for grouping
  * images touching a cross-class pair >= conflict threshold are excluded
  * at most CAP random images are kept per group

  python .\\scripts\\10d_design_sweep.py --data_dir .\\dataset\\enhanced_extended --classes Healthy PCOS ^
         --pairs .\\reports\\splits\\pair_scores.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def labels_of(root, classes):
    y = []
    for k, cls in enumerate(classes):
        d = Path(root) / cls
        if not d.is_dir():
            sys.exit(f"ERROR: class folder not found: {d}")
        y += [k for p in sorted(d.rglob("*")) if p.suffix.lower() in EXTS]
    return np.array(y)


def components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return np.array([find(i) for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--classes", nargs=2, required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--min_inliers", nargs="+", type=int, default=[25, 30, 35, 40, 50, 60, 80])
    ap.add_argument("--conflict_inliers", nargs="+", type=int, default=[60, 80, 100])
    ap.add_argument("--caps", nargs="+", type=int, default=[20, 40, 0])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    y = labels_of(a.data_dir, a.classes)
    c = pd.read_csv(a.pairs)
    if c[["i", "j"]].max().max() >= len(y):
        sys.exit("ERROR: pair_scores.csv does not match this image folder")
    I, J, S = c.i.values, c.j.values, c.inliers.values
    cross_all = y[I] != y[J]
    n = len(y)
    print(f"{n} images: {a.classes[0]} {int((y == 0).sum())}, {a.classes[1]} {int((y == 1).sum())}\n")
    print(f"{'group>=':>7} {'confl>=':>7} {'cap':>4} | {'kept':>5} {'healthy':>7} {'pcos':>5} | "
          f"{'groups':>6} {'largest':>7} | {'conflict-excl':>13}")
    for mi in a.min_inliers:
        same = (~cross_all) & (S >= mi)
        comp = components(n, list(zip(I[same], J[same])))
        for ci in a.conflict_inliers:
            if ci < mi:
                continue
            sc = cross_all & (S >= ci)
            conflict = np.zeros(n, bool)
            conflict[I[sc]] = True
            conflict[J[sc]] = True
            for cap in a.caps:
                rng = np.random.default_rng(a.seed)
                keep = ~conflict
                if cap > 0:
                    df = pd.DataFrame({"g": comp, "i": np.arange(n)})[keep]
                    for _, idx in df.groupby("g")["i"]:
                        idx = idx.values
                        if len(idx) > cap:
                            keep[rng.choice(idx, size=len(idx) - cap, replace=False)] = False
                ng = len(set(comp[keep]))
                big = pd.Series(comp[keep]).value_counts().iloc[0] if keep.any() else 0
                print(f"{mi:>7} {ci:>7} {cap:>4} | {keep.sum():>5} {int((keep & (y == 0)).sum()):>7} "
                      f"{int((keep & (y == 1)).sum()):>5} | {ng:>6} {big:>7} | {int(conflict.sum()):>13}")
        print()
    print("Pick a row that keeps enough HEALTHY images (aim for >= ~150, so val/test each get >= ~25) "
          "while group>= is as low as your borderline_pairs.png inspection allows. cap=0 means no cap.")


if __name__ == "__main__":
    main()
