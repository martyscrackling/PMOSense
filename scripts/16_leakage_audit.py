"""
16_leakage_audit.py - after the split is made, how many validation / test images still have a
near-duplicate in TRAIN (or in each other)? Uses the saved pair_scores.csv, so it is instant.

Pairs scoring >= --min_inliers were grouped and cannot cross splits. This script looks at the pairs
scoring BELOW that threshold (weaker, but often still copies - e.g. the 31-34 inlier pairs seen in
borderline_pairs.png) and counts how many cross a split boundary.

If validation has far more such cross-split partners than test, that explains why validation AUC
is ~1.0 while test AUC is much lower: validation is partly 'seen' scans.

  python .\\scripts\\16_leakage_audit.py --data_dir .\\dataset\\enhanced_extended --classes Healthy PCOS ^
         --splits .\\reports\\splits\\splits.csv --pairs .\\reports\\splits\\pair_scores.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
BINS = [(10, 20), (20, 25), (25, 30), (30, 35), (35, 45), (45, 10**6)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="the SAME folder you gave 10_make_splits.py")
    ap.add_argument("--classes", nargs=2, required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--pairs", required=True)
    a = ap.parse_args()

    paths = []
    for cls in a.classes:
        d = Path(a.data_dir) / cls
        if not d.is_dir():
            sys.exit(f"ERROR: class folder not found: {d}")
        paths += [str(p.relative_to(a.data_dir)).replace("\\", "/")
                  for p in sorted(d.rglob("*")) if p.suffix.lower() in EXTS]
    sp = pd.read_csv(a.splits)
    sp = sp[sp.split.isin(["train", "val", "test"])]
    split_of = dict(zip(sp.path, sp.split))
    label_of = dict(zip(sp.path, sp.label))
    c = pd.read_csv(a.pairs)
    if c[["i", "j"]].max().max() >= len(paths):
        sys.exit("ERROR: pair_scores.csv does not match this image folder")
    arr = np.array(paths, dtype=object)
    si = np.array([split_of.get(p, "") for p in arr[c.i.values]])
    sj = np.array([split_of.get(p, "") for p in arr[c.j.values]])
    ok = (si != "") & (sj != "") & (si != sj)
    li = np.array([label_of.get(p, -1) for p in arr[c.i.values]])
    lj = np.array([label_of.get(p, -1) for p in arr[c.j.values]])
    c = c[ok].copy()
    c["si"], c["sj"] = si[ok], sj[ok]
    c["same_label"] = li[ok] == lj[ok]
    n_img = sp.groupby("split").size().to_dict()
    print("Images per split:", n_img, "\n")

    def partners(target, other, same_label):
        """images in `target` having >=1 partner in `other`, per score bin (cumulative columns too)."""
        m = ((c.si == target) & (c.sj == other)) | ((c.sj == target) & (c.si == other))
        sub = c[m & (c.same_label == same_label)]
        who = np.where(sub.si == target, sub.i, sub.j)
        sub = sub.assign(img=who)
        out = {}
        for lo, hi in BINS:
            s = sub[(sub.inliers >= lo) & (sub.inliers < hi)]
            out[f"{lo}-{hi if hi < 10**6 else '+'}"] = s.img.nunique()
        out[">=25 (any)"] = sub[sub.inliers >= 25].img.nunique()
        out[">=30 (any)"] = sub[sub.inliers >= 30].img.nunique()
        return out

    for same, title in [(True, "SAME-label partners across a split boundary (classic leakage: inflates scores)"),
                        (False, "OPPOSITE-label partners across a split boundary (conflicting labels: deflates scores)")]:
        rows = []
        for target, other in [("val", "train"), ("test", "train"), ("test", "val")]:
            r = partners(target, other, same)
            r = {k: f"{v} ({100 * v / n_img[target]:.0f}%)" for k, v in r.items()}
            rows.append({"images in": target, "with a partner in": other, **r})
        print(f"{title}\n(number and share of images; best pair's ORB-inlier score bin)\n")
        print(pd.DataFrame(rows).to_string(index=False))
        print()
    print("Reading it:\n"
          " - SAME-label table, bins >= your grouping threshold should be ~0 (those pairs were grouped). "
          "Counts in the bins just below it are residual leakage that INFLATES scores.\n"
          " - OPPOSITE-label table: an image whose near-identical twin has the other label is "
          "unlearnable; if test has far more of these than validation, test AUC is dragged down by label "
          "conflicts, not by the model.")


if __name__ == "__main__":
    main()
