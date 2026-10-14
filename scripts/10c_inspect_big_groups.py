"""
10c_inspect_big_groups.py - look INSIDE the biggest groups to decide whether they are real
families of re-saved copies, or unrelated scans glued together by static features
(scanner text, scale marks, fan outline) that sit at the same pixel position in every image.

Reads pair_scores.csv from 10_make_splits.py. Changes nothing except writing PNGs/CSV.

  python .\\scripts\\10c_inspect_big_groups.py --data_dir .\\dataset\\enhanced_extended ^
         --classes Healthy PCOS --pairs .\\reports\\splits\\pair_scores.csv --thr 60 --out_dir .\\reports\\inspect

Outputs:
  big_group_members.png   : random members of the 3 largest groups (one group per block of rows)
  linked_pairs.png        : 8 random linked pairs from the largest group, drawn with their RANSAC
                            inlier matches (green). Lines that are all short and horizontal-parallel
                            at the SAME position = static features; lines on tissue that move = real copy.
  static_fraction.csv     : for 300 linked pairs, the share of inlier points that did not move (<2 px).
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


def list_files(root, classes):
    out = []
    for cls in classes:
        d = Path(root) / cls
        if not d.is_dir():
            sys.exit(f"ERROR: class folder not found: {d}")
        out += [(str(p), cls) for p in sorted(d.rglob("*")) if p.suffix.lower() in EXTS]
    return out


def feats(img, n=200):
    orb = cv2.ORB_create(nfeatures=n, fastThreshold=10, edgeThreshold=15)
    kps, des = orb.detectAndCompute(img, None)
    pts = np.float32([k.pt for k in kps]) if kps else np.zeros((0, 2), np.float32)
    return pts, des


def match(fa, fb):
    """Return (n_inliers, src_pts_inliers, dst_pts_inliers) for direct matching."""
    (pa, da), (pb, db) = fa, fb
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return 0, None, None
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 10:
        return 0, None, None
    src = pa[[g.queryIdx for g in good]]
    dst = pb[[g.trainIdx for g in good]]
    _, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if inl is None:
        return 0, None, None
    m = inl.ravel().astype(bool)
    return int(m.sum()), src[m], dst[m]


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


def thumb(path, s=112):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, (s, s)) if im is not None else np.zeros((s, s), np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--classes", nargs=2, required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--thr", type=int, default=60)
    ap.add_argument("--out_dir", default="reports/inspect")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    files = list_files(a.data_dir, a.classes)
    c = pd.read_csv(a.pairs)
    if c[["i", "j"]].max().max() >= len(files):
        sys.exit("ERROR: pair_scores.csv does not match this image folder")
    e = c[c.inliers >= a.thr]
    comp = components(len(files), list(zip(e.i, e.j)))
    sizes = pd.Series(comp).value_counts()
    print(f"threshold {a.thr}: largest groups = {sizes.values[:5].tolist()}")

    # ---- 1. members of the 3 biggest groups
    per_row, n_show = 12, 24
    blocks = []
    for gid in sizes.index[:3]:
        members = np.flatnonzero(comp == gid)
        pick = rng.choice(members, size=min(n_show, len(members)), replace=False)
        cls = pd.Series([files[m][1] for m in members]).value_counts().to_dict()
        print(f"  group of {len(members)} images, classes {cls}")
        tiles = [thumb(files[m][0]) for m in pick]
        while len(tiles) % per_row:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[k:k + per_row]) for k in range(0, len(tiles), per_row)]
        blk = np.vstack(rows)
        blk = cv2.copyMakeBorder(blk, 22, 4, 0, 0, cv2.BORDER_CONSTANT, value=255)
        cv2.putText(blk, f"group of {len(members)} images  {cls}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        blocks.append(blk)
    cv2.imwrite(str(out / "big_group_members.png"), np.vstack(blocks))

    # ---- 2. linked pairs drawn with inlier matches (largest group)
    big = e[(comp[e.i.values] == sizes.index[0])]
    sel = big.sample(min(8, len(big)), random_state=a.seed)
    panels = []
    for _, r in sel.iterrows():
        ia, ib = int(r.i), int(r.j)
        A = cv2.imread(files[ia][0], cv2.IMREAD_GRAYSCALE)
        B = cv2.imread(files[ib][0], cv2.IMREAD_GRAYSCALE)
        n, s, d = match(feats(A), feats(B))
        canvas = cv2.cvtColor(np.hstack([A, B]), cv2.COLOR_GRAY2BGR)
        if n:
            for p, q in zip(s, d):
                cv2.line(canvas, (int(p[0]), int(p[1])), (int(q[0] + A.shape[1]), int(q[1])), (0, 220, 0), 1)
        cv2.putText(canvas, f"{n} inliers (saved score {int(r.inliers)})", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        panels.append(canvas)
    if panels:
        grid = [np.hstack(panels[k:k + 2]) if k + 1 < len(panels) else panels[k] for k in range(0, len(panels), 2)]
        w = max(g.shape[1] for g in grid)
        grid = [cv2.copyMakeBorder(g, 0, 0, 0, w - g.shape[1], cv2.BORDER_CONSTANT) for g in grid]
        cv2.imwrite(str(out / "linked_pairs.png"), np.vstack(grid))

    # ---- 3. static-feature test over 300 linked pairs (direct matches only)
    rows = []
    cache = {}

    def get(i):
        if i not in cache:
            cache[i] = feats(cv2.imread(files[i][0], cv2.IMREAD_GRAYSCALE))
        return cache[i]

    for _, r in e.sample(min(300, len(e)), random_state=a.seed).iterrows():
        n, s, d = match(get(int(r.i)), get(int(r.j)))
        if n >= 10:
            disp = np.linalg.norm(d - s, axis=1)
            rows.append({"i": int(r.i), "j": int(r.j), "saved_score": int(r.inliers), "inliers_now": n,
                         "static_frac": float((disp < 2).mean()), "median_shift_px": float(np.median(disp)),
                         "in_big_group": bool(comp[int(r.i)] == sizes.index[0])})
    sf = pd.DataFrame(rows)
    sf.to_csv(out / "static_fraction.csv", index=False)
    if len(sf):
        print(f"\nOf {len(sf)} sampled linked pairs that re-matched directly:")
        print(f"  median share of inlier points that did NOT move (<2 px): {sf.static_frac.median():.2f}")
        print(f"  pairs where >=70% of inliers did not move: {(sf.static_frac >= .7).mean() * 100:.0f}%")
        print(f"  median shift of inlier points: {sf.median_shift_px.median():.1f} px")
        print("Reading it: static share near 1 => matches are on fixed overlay features (false links). "
              "Static share near 0 with large shifts => matches follow the tissue (real re-saved copies).")
    print(f"\nWrote {out}\\big_group_members.png, linked_pairs.png, static_fraction.csv")


if __name__ == "__main__":
    main()
