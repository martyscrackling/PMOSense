"""
17_show_conflict_pairs.py - LOOK at pairs of images that have OPPOSITE labels but match each other
(test image vs train image, or val vs train). Decide with your own eyes: same scan labelled both ways,
or just similar-looking different scans?

  python .\\scripts\\17_show_conflict_pairs.py --data_dir .\\dataset\\enhanced_extended --classes Healthy PCOS ^
         --splits .\\reports\\splits\\splits.csv --pairs .\\reports\\splits\\pair_scores.csv --min_inliers 45

Writes reports\\inspect\\conflict_pairs.png (left = the held-out image, right = its training twin).
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
import cv2
import numpy as np
import pandas as pd

cv2.ocl.setUseOpenCL(False)
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def tile(path, text, s=200):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (s, s)) if im is not None else np.zeros((s, s), np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    im = cv2.copyMakeBorder(im, 20, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(im, text, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--classes", nargs=2, required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--min_inliers", type=int, default=45)
    ap.add_argument("--max_inliers", type=int, default=10**6)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--heldout", choices=["val", "test"], default="test")
    ap.add_argument("--out_dir", default="reports/inspect")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    paths = []
    for cls in a.classes:
        d = Path(a.data_dir) / cls
        if not d.is_dir():
            sys.exit(f"ERROR: class folder not found: {d}")
        paths += [str(p.relative_to(a.data_dir)).replace("\\", "/")
                  for p in sorted(d.rglob("*")) if p.suffix.lower() in EXTS]
    sp = pd.read_csv(a.splits)
    split_of, label_of = dict(zip(sp.path, sp.split)), dict(zip(sp.path, sp.label))
    c = pd.read_csv(a.pairs)
    if c[["i", "j"]].max().max() >= len(paths):
        sys.exit("ERROR: pair_scores.csv does not match this image folder")
    c = c[(c.inliers >= a.min_inliers) & (c.inliers < a.max_inliers)]
    arr = np.array(paths, dtype=object)
    rows = []
    for i, j, s in zip(c.i.values, c.j.values, c.inliers.values):
        pi, pj = arr[i], arr[j]
        si, sj = split_of.get(pi), split_of.get(pj)
        if si is None or sj is None or label_of[pi] == label_of[pj]:
            continue
        if si == a.heldout and sj == "train":
            rows.append((pi, pj, s))
        elif sj == a.heldout and si == "train":
            rows.append((pj, pi, s))
    print(f"{len(rows)} opposite-label pairs, {a.heldout} vs train, with {a.min_inliers}+ inliers")
    if not rows:
        return
    rng = np.random.default_rng(a.seed)
    pick = [rows[k] for k in rng.choice(len(rows), size=min(a.n, len(rows)), replace=False)]
    name = {0: a.classes[0], 1: a.classes[1]}
    panels = []
    for h, t, s in pick:
        left = tile(str(Path(a.data_dir) / h), f"{a.heldout}: {name[label_of[h]]}")
        right = tile(str(Path(a.data_dir) / t), f"train: {name[label_of[t]]}  ({s} inl.)")
        panels.append(np.hstack([left, right]))
    per_row = 2
    while len(panels) % per_row:
        panels.append(np.full_like(panels[0], 255))
    grid = np.vstack([np.hstack(panels[k:k + per_row]) for k in range(0, len(panels), per_row)])
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "conflict_pairs.png"), grid)
    print(f"wrote {out / 'conflict_pairs.png'}  (left = held-out image, right = its training twin)")


if __name__ == "__main__":
    main()
