"""
10b_threshold_sweep.py - see how the duplicate-GROUP structure changes with the linking rule,
using the pair scores that 10_make_splits.py already saved. Takes seconds. Changes nothing.

Why: single-linkage grouping (link A-B if they match, merge everything connected) can chain
unrelated scans into one giant group through a few weak links. Two rules are compared:

  plain    : link every pair with inliers >= T
  support  : keep a link A-B only if A and B ALSO share >= M common linked neighbours.
             Real families of re-saved copies are tightly interconnected (A~B, A~C, B~C ...);
             an accidental chain A~B~C where A and C are unrelated has no such support.

Usage (from the pcosense folder):
  python .\\scripts\\10b_threshold_sweep.py --data_dir .\\dataset\\enhanced_extended --classes Healthy PCOS ^
         --pairs .\\reports\\splits\\pair_scores.csv
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_labels(root, classes):
    rows = []
    for k, cls in enumerate(classes):
        d = Path(root) / cls
        if not d.is_dir():
            sys.exit(f"ERROR: class folder not found: {d}")
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in EXTS:
                rows.append(k)
    return np.array(rows)


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


def supported(edges, m):
    nb = defaultdict(set)
    for i, j in edges:
        nb[i].add(j)
        nb[j].add(i)
    return [(i, j) for i, j in edges if len(nb[i] & nb[j]) >= m]


def summarize(labels, edges):
    comp = components(len(labels), edges)
    s = pd.Series(comp).value_counts()
    mixed = int((pd.DataFrame({"g": comp, "y": labels}).groupby("g")["y"].nunique() > 1).sum())
    top = s.values[:5].tolist()
    return len(s), int(s.iloc[0]), top, mixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--classes", nargs=2, required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--thresholds", nargs="+", type=int, default=[25, 40, 60, 80, 100, 130, 160, 200])
    ap.add_argument("--min_common", nargs="+", type=int, default=[0, 1, 2, 3])
    a = ap.parse_args()

    labels = list_labels(a.data_dir, a.classes)
    c = pd.read_csv(a.pairs)
    if c[["i", "j"]].max().max() >= len(labels):
        sys.exit("ERROR: pair_scores.csv does not match this image folder")
    print(f"{len(labels)} images, {len(c):,} scored pairs (inliers: median {c.inliers.median():.0f}, "
          f"90th pct {c.inliers.quantile(.9):.0f}, max {c.inliers.max()})\n")
    print(f"{'thr':>5} {'M':>2} {'edges':>8} {'groups':>7} {'largest':>8} {'mixed-class':>11}  top-5 sizes")
    for t in a.thresholds:
        e0 = list(zip(c.i[c.inliers >= t], c.j[c.inliers >= t]))
        for m in a.min_common:
            e = e0 if m == 0 else supported(e0, m)
            g, big, top, mixed = summarize(labels, e)
            print(f"{t:>5} {m:>2} {len(e):>8} {g:>7} {big:>8} {mixed:>11}  {top}")
        print()
    print("M=0 is the plain rule used by 10_make_splits.py. Look for a row where the largest group is "
          "small (tens, not hundreds), mixed-class is ~0, and most images are still in non-singleton groups.")


if __name__ == "__main__":
    main()
