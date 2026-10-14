"""
10_make_splits.py

Model-selection step 1 of 3: build a LEAKAGE-SAFE train / val / test split.

Why this script exists
----------------------
In your gallery screenshot, runs of neighbouring files (e.g. Image_419 ..
Image_426) are the SAME ovary re-saved with different shifts, rotations,
gray bars and white wedges. If those variants land in different splits,
the test set contains images the model has effectively already seen, and
every metric is inflated. A plain random (even stratified) split - which
is what Chapter III currently describes - cannot prevent that.

What it does
------------
  1. Extracts ORB keypoints from every image (and from its mirror image).
  2. Compares every pair: ratio-test descriptor matching, then RANSAC
     fits a rotation+shift+scale transform and counts INLIERS. Two
     variants of one scan share dozens to hundreds of inliers; two
     different scans share almost none. Unlike a perceptual hash, this is
     unaffected by shifts, rotations, mirroring, bars and wedges.
  3. Links pairs with >= --min_inliers and merges links into GROUPS
     (connected components, union-find).
  4. Assigns WHOLE GROUPS to train / val / test, keeping the PCOS/Healthy
     ratio near the target in every split.
  5. Asserts that no group appears in more than one split.
  6. Optionally registers a second, independent dataset as an EXTERNAL
     test set, dropping any external image that duplicates a main image.

Why not pHash here: tested on shifted / rotated / wedge-padded copies,
64-bit pHash distances for TRUE copies (median 12-20 bits) overlapped the
distances between DIFFERENT scans (median 20, 1st percentile 8), so no
threshold separated them. ORB+RANSAC separated them cleanly. Both tests
used synthetic images - confirm on yours with the audit images.

Cost: all-pairs matching. Measured ~1 ms per pair per core, i.e. ~50
CPU-minutes for 2,500 images (divided by --workers). Much larger datasets
need a candidate-retrieval stage first; the script prints an estimate.

Usage
  python 10_make_splits.py --data_dir ../dataset/resized \
      --classes Healthy PCOS --out_dir ../reports/splits --seed 42
  # optional independent dataset (already preprocessed the same way):
  #   --external_dir ../dataset_external/resized
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")  # must be set before importing cv2
import cv2
import numpy as np
import pandas as pd

cv2.ocl.setUseOpenCL(False)  # CPU only: OpenCL init crashes worker processes on some Windows machines
cv2.setNumThreads(1)         # parallelism comes from the process pool, not from OpenCV

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
REPORT_FLOOR = 10  # pairs with fewer inliers than this are never recorded


# ---------------------------------------------------------------- file listing
def list_images(root, classes):
    rows = []
    for cls in classes:
        d = Path(root) / cls
        if not d.is_dir():
            sys.exit(f"ERROR: class folder not found: {d}")
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in EXTS:
                rel = str(p.relative_to(root)).replace("\\", "/")
                rows.append({"path": rel, "class": cls, "abs": str(p)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ ORB feature stage
def _quiet_cv():
    """Runs in every worker process (Windows 'spawn' re-imports cv2 fresh in each one)."""
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(1)


def extract_one(args):
    path, n_features = args
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=10, edgeThreshold=15)
    out = []
    for im in (img, cv2.flip(img, 1)):  # original + mirror image
        kps, desc = orb.detectAndCompute(im, None)
        pts = np.float32([k.pt for k in kps]) if kps else np.zeros((0, 2), np.float32)
        out.append((pts, desc))
    return out


def extract_all(paths, n_features, workers):
    t0 = time.time()
    with mp.Pool(workers, initializer=_quiet_cv) as pool:
        feats = pool.map(extract_one, [(p, n_features) for p in paths], chunksize=32)
    print(f"  features for {len(paths)} images in {time.time() - t0:.0f}s")
    return feats


# --------------------------------------------------------- pairwise match stage
_A = _B = _BF = None
_CFG = None


def _init(feats_a, feats_b, cfg):
    global _A, _B, _BF, _CFG
    _quiet_cv()
    _A, _B, _CFG = feats_a, feats_b, cfg
    _BF = cv2.BFMatcher(cv2.NORM_HAMMING)


def _inliers(fa, fb):
    """RANSAC inlier count for (rotation + shift + scale) between two ORB sets."""
    (pa, da), (pb, db) = fa, fb
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return 0
    knn = _BF.knnMatch(da, db, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < _CFG["min_good"]:
        return 0
    src = pa[[g.queryIdx for g in good]]
    dst = pb[[g.trainIdx for g in good]]
    _, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return 0 if inl is None else int(inl.sum())


def _row(i):
    """Compare image i of set A against every needed image of set B."""
    fa = _A[i]
    if fa is None:
        return []
    start = i + 1 if _CFG["same_set"] else 0
    res = []
    for j in range(start, len(_B)):
        fb = _B[j]
        if fb is None:
            continue
        s = max(_inliers(fa[0], fb[0]), _inliers(fa[0], fb[1]))  # direct or mirrored
        if s >= REPORT_FLOOR:
            res.append((i, j, s))
    return res


def match_sets(feats_a, feats_b, same_set, workers, min_good, label):
    n_pairs = (len(feats_a) * (len(feats_a) - 1) // 2) if same_set else len(feats_a) * len(feats_b)
    est_min = n_pairs * 0.0011 / max(workers, 1) / 60
    print(f"  {label}: {n_pairs:,} pairs, ~{est_min:.0f} min with {workers} worker(s)")
    cfg = {"same_set": same_set, "min_good": min_good}
    pairs, t0 = [], time.time()
    with mp.Pool(workers, initializer=_init, initargs=(feats_a, feats_b, cfg)) as pool:
        for k, res in enumerate(pool.imap_unordered(_row, range(len(feats_a)), chunksize=4), 1):
            pairs.extend(res)
            if k % max(1, len(feats_a) // 20) == 0:
                print(f"    {100 * k / len(feats_a):.0f}%  ({time.time() - t0:.0f}s)")
    return pairs


# ------------------------------------------------------------------ union-find
def groups_from_pairs(n, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([remap[r] for r in roots])


# ------------------------------------------------------------- group splitting
def assign_groups(df, ratios, seed):
    """Greedy, deterministic assignment of WHOLE groups to splits, keeping
    per-class proportions near the targets. Largest groups go first."""
    rng = np.random.default_rng(seed)
    classes = sorted(df["label"].unique())
    totals = df.groupby("label").size().to_dict()
    target = {s: {c: ratios[s] * totals[c] for c in classes} for s in ratios}
    cur = {s: {c: 0 for c in classes} for s in ratios}

    comp = (df.groupby(["group_id", "label"]).size().unstack(fill_value=0)
            .reindex(columns=classes, fill_value=0))
    order = comp.sum(axis=1).to_frame("n")
    order["tie"] = rng.random(len(order))
    order = order.sort_values(["n", "tie"], ascending=[False, True]).index

    assignment = {}
    for gid in order:
        cnt = comp.loc[gid].to_dict()
        best, best_score = None, -1e18
        for s in ratios:
            score = sum(cnt[c] * (target[s][c] - cur[s][c]) / max(target[s][c], 1e-9)
                        for c in classes)
            if score > best_score:
                best, best_score = s, score
        assignment[gid] = best
        for c in classes:
            cur[best][c] += cnt[c]
    return df["group_id"].map(assignment)


# -------------------------------------------------------------------- montages
def thumb(path, size=112):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def save_group_montage(df, out_png, n_groups=14, per_row=8, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = df.groupby("group_id").size()
    multi = sizes[sizes >= 2].sort_values(ascending=False)
    if multi.empty:
        return False
    rng = np.random.default_rng(seed)
    top = list(multi.index[: n_groups // 2])
    rest = list(multi.index[n_groups // 2:])
    if rest:
        top += list(rng.choice(rest, size=min(len(rest), n_groups - len(top)), replace=False))
    fig, axes = plt.subplots(len(top), per_row, figsize=(per_row * 1.3, len(top) * 1.45))
    axes = np.atleast_2d(axes)
    for r, gid in enumerate(top):
        members = df[df.group_id == gid].head(per_row)
        for c in range(per_row):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(members):
                ax.imshow(thumb(members.iloc[c]["abs"]), cmap="gray", vmin=0, vmax=255)
                ax.set_title(members.iloc[c]["class"][:7], fontsize=6)
        axes[r][0].text(-0.1, 0.5, f"g{gid}\nn={sizes[gid]}", transform=axes[r][0].transAxes,
                        ha="right", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()
    return True


def save_borderline_montage(df, pairs, thr, out_png, n=10, seed=0):
    """Pairs scoring just around the threshold: the most informative ones
    for deciding whether --min_inliers is too loose or too strict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edge = [p for p in pairs if thr - 10 <= p[2] <= thr + 15]
    if not edge:
        return False
    rng = np.random.default_rng(seed)
    pick = [edge[i] for i in rng.permutation(len(edge))[:n]]
    fig, axes = plt.subplots(len(pick), 2, figsize=(3.2, len(pick) * 1.7))
    axes = np.atleast_2d(axes)
    for r, (i, j, s) in enumerate(pick):
        for c, idx in enumerate((i, j)):
            axes[r][c].imshow(thumb(df.iloc[idx]["abs"]), cmap="gray", vmin=0, vmax=255)
            axes[r][c].axis("off")
        axes[r][0].set_title(f"inliers={s} ({'LINKED' if s >= thr else 'not linked'})", fontsize=6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()
    return True


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Stage folder to analyse (e.g. dataset/resized), one sub-folder per class.")
    ap.add_argument("--classes", nargs=2, required=True, metavar=("NEG", "POS"),
                    help="Folder names, negative class first, e.g. --classes Healthy PCOS")
    ap.add_argument("--external_dir", default=None,
                    help="Optional independent dataset (same class folder names, same preprocessing).")
    ap.add_argument("--out_dir", default="reports/splits")
    ap.add_argument("--n_features", type=int, default=200, help="ORB keypoints per image.")
    ap.add_argument("--min_inliers", type=int, default=25,
                    help="RANSAC inliers needed to link two images. Synthetic test: copies >= 50, "
                         "different scans <= 14. Check borderline_pairs.png on your data.")
    ap.add_argument("--min_good", type=int, default=10,
                    help="Minimum ratio-test matches before RANSAC is attempted (speed filter).")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_per_group", type=int, default=0,
                    help="Keep at most this many (random) images per duplicate group, so a few heavily "
                         "re-saved scans cannot dominate train or test. 0 = no cap. Excluded images are "
                         "listed in excluded_images.csv and are NOT moved to another split.")
    ap.add_argument("--conflict_inliers", type=int, default=0,
                    help="Inlier threshold for calling two differently-labelled images 'the same scan "
                         "with conflicting labels' (default 0 = same as --min_inliers). Use a HIGHER value "
                         "than --min_inliers: a false conflict deletes data, whereas a missed conflict only "
                         "adds label noise that lowers (never inflates) measured accuracy.")
    ap.add_argument("--keep_conflicts", action="store_true",
                    help="Keep images that match an image of the other class (default: exclude them).")
    ap.add_argument("--reuse_pairs", action="store_true",
                    help="Skip the slow matching and reuse out_dir/pair_scores.csv from a previous run "
                         "on the SAME images; lets you try other --min_inliers in seconds.")
    args = ap.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        sys.exit("ERROR: --ratios must sum to 1")
    pos = args.classes[1]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = list_images(args.data_dir, args.classes)
    if df.empty:
        sys.exit("ERROR: no images found")
    df["label"] = (df["class"] == pos).astype(int)
    print(f"Found {len(df)} images: {df.groupby('class').size().to_dict()}")

    # ---- features (unreadable files are reported and dropped, never silently kept)
    feats = extract_all(df["abs"].tolist(), args.n_features, args.workers)
    bad = [i for i, f in enumerate(feats) if f is None]
    if bad:
        print(f"WARNING: dropping {len(bad)} unreadable image(s)")
        (out / "unreadable_files.txt").write_text("\n".join(df.iloc[bad]["abs"]))
        keep = [i for i in range(len(df)) if i not in set(bad)]
        df = df.iloc[keep].reset_index(drop=True)
        feats = [feats[i] for i in keep]
    few = [i for i, f in enumerate(feats) if f[0][1] is None or len(f[0][1]) < 20]
    if few:
        print(f"WARNING: {len(few)} image(s) have <20 keypoints and cannot be matched; they stay "
              f"singletons, so a near-duplicate of one of them could still leak. See few_keypoints.txt")
        (out / "few_keypoints.txt").write_text("\n".join(df.iloc[few]["path"]))

    # ---- groups
    print("Matching all pairs inside the main set ...")
    cache = out / "pair_scores.csv"
    if args.reuse_pairs and cache.exists():
        c = pd.read_csv(cache)
        if len(c) and c[["i", "j"]].max().max() >= len(df):
            sys.exit("ERROR: pair_scores.csv does not match this image set; rerun without --reuse_pairs")
        pairs_all = list(zip(c["i"], c["j"], c["inliers"]))
        print(f"  reusing {len(pairs_all):,} cached pair scores from {cache} (no matching rerun)")
    else:
        pairs_all = match_sets(feats, feats, True, args.workers, args.min_good, "main vs main")
        pd.DataFrame(pairs_all, columns=["i", "j", "inliers"]).to_csv(cache, index=False)
    pairs = [p for p in pairs_all if p[2] >= args.min_inliers]
    vals = np.array([p[2] for p in pairs_all])
    pd.Series(vals).describe().to_csv(out / "inlier_score_summary.csv")

    # Links between images of DIFFERENT classes are never used for grouping: one weak bridge
    # must not glue two unrelated families together, and a same-scan pair labelled both ways
    # is label noise. Images touching such a link are flagged and (by default) excluded.
    y = df["label"].values
    cross = [p for p in pairs if y[p[0]] != y[p[1]]]
    same = [p for p in pairs if y[p[0]] == y[p[1]]]
    df["group_id"] = groups_from_pairs(len(df), same)
    df["status"] = "kept"
    ci = args.conflict_inliers if args.conflict_inliers > 0 else args.min_inliers
    strong_cross = [p for p in cross if p[2] >= ci]
    conflict = sorted({k for i, j, _ in strong_cross for k in (i, j)})
    print(f"  {len(pairs)} linked pairs at >= {args.min_inliers} inliers ({len(cross)} connect DIFFERENT "
          f"classes, never used for grouping); {len(strong_cross)} cross-class pairs at >= {ci} inliers "
          f"define label conflicts")
    if conflict:
        df.loc[conflict, ["path", "class", "group_id"]].assign(
            reason="matches an image of the other class") \
            .to_csv(out / "label_conflict_images.csv", index=False)
        if args.keep_conflicts:
            print(f"WARNING: {len(conflict)} image(s) match an image of the other class "
                  f"(label_conflict_images.csv). Kept because of --keep_conflicts.")
        else:
            df.loc[conflict, "status"] = "excluded_label_conflict"
            print(f"WARNING: {len(conflict)} image(s) match an image of the OTHER class (same scan labelled "
                  f"both ways, or a false match). Excluded from all splits; see label_conflict_images.csv.")

    rng_cap = np.random.default_rng(args.seed)
    if args.max_per_group > 0:
        for gid, idx in df[df.status == "kept"].groupby("group_id").groups.items():
            if len(idx) > args.max_per_group:
                drop = rng_cap.choice(np.array(idx), size=len(idx) - args.max_per_group, replace=False)
                df.loc[drop, "status"] = "excluded_over_cap"
    df_all = df.copy()  # full table, indices match pair_scores.csv
    excl = df_all[df_all.status != "kept"]
    if len(excl):
        excl[["path", "class", "group_id", "status"]].to_csv(out / "excluded_images.csv", index=False)
    df = df_all[df_all.status == "kept"].copy()
    sizes = df.groupby("group_id").size()
    sizes_all = df_all.groupby("group_id").size()
    print(f"  groups before exclusions: {len(sizes_all)} (singletons {(sizes_all == 1).sum()}, "
          f"largest {sizes_all.max()})")
    print(f"  kept {len(df)} of {len(df_all)} images in {len(sizes)} groups "
          f"(excluded: {(excl.status == 'excluded_label_conflict').sum()} label-conflict, "
          f"{(excl.status == 'excluded_over_cap').sum()} over per-group cap)")
    if sizes.max() > 0.05 * len(df):
        print(f"WARNING: largest kept group holds {sizes.max()} images "
              f"({100 * sizes.max() / len(df):.1f}% of the kept data). Consider --max_per_group so one "
              f"scan cannot dominate a split.")

    # ---- split
    ratios = dict(zip(["train", "val", "test"], args.ratios))
    df["split"] = assign_groups(df, ratios, args.seed)
    df["source"] = "main"
    per_group = df.groupby("group_id")["split"].nunique()
    assert (per_group == 1).all(), "BUG: a group was assigned to more than one split"

    # ---- optional external set
    ext_df, ext_dropped = None, []
    if args.external_dir:
        ext_df = list_images(args.external_dir, args.classes)
        ext_df["label"] = (ext_df["class"] == pos).astype(int)
        print(f"External set: {len(ext_df)} images")
        efeats = extract_all(ext_df["abs"].tolist(), args.n_features, args.workers)
        ok = [i for i, f in enumerate(efeats) if f is not None]
        ext_df = ext_df.iloc[ok].reset_index(drop=True)
        efeats = [efeats[i] for i in ok]
        cross = [p for p in match_sets(efeats, feats, False, args.workers, args.min_good,
                                       "external vs main") if p[2] >= args.min_inliers]
        dup_idx = sorted({i for i, _, _ in cross})
        ext_dropped = ext_df.iloc[dup_idx]["path"].tolist()
        if ext_dropped:
            (out / "external_dropped_as_duplicates_of_main.txt").write_text("\n".join(ext_dropped))
            print(f"  dropped {len(ext_dropped)} external image(s) that duplicate a main-set image")
        ext_pairs = [p for p in match_sets(efeats, efeats, True, args.workers, args.min_good,
                                           "external vs external") if p[2] >= args.min_inliers]
        ext_df["group_id"] = groups_from_pairs(len(ext_df), ext_pairs) + int(df_all["group_id"].max()) + 1
        ext_df = ext_df.drop(index=dup_idx).reset_index(drop=True)
        ext_df["split"] = "external"
        ext_df["source"] = "external"

    final = pd.concat([df, ext_df], ignore_index=True) if ext_df is not None else df
    final[["path", "class", "label", "group_id", "split", "source"]].to_csv(out / "splits.csv", index=False)

    # ---- report
    rep = {"method": "orb_ransac", "min_inliers": args.min_inliers, "n_features": args.n_features,
           "seed": args.seed, "ratios_target": ratios, "n_images_main": int(len(df)),
           "n_groups_main": int(len(sizes)), "largest_group": int(sizes.max()),
           "max_per_group": args.max_per_group, "n_images_before_exclusions": int(len(df_all)),
           "n_excluded_label_conflict": int((df_all.status == "excluded_label_conflict").sum()),
           "n_excluded_over_cap": int((df_all.status == "excluded_over_cap").sum()),
           "splits": {}}
    for s in ["train", "val", "test", "external"]:
        sub = final[final.split == s]
        if sub.empty:
            continue
        rep["splits"][s] = {"images": int(len(sub)), "groups": int(sub.group_id.nunique()),
                            "pcos": int((sub.label == 1).sum()), "healthy": int((sub.label == 0).sum()),
                            "pcos_fraction": round(float((sub.label == 1).mean()), 3)}
    if ext_df is not None:
        rep["external_dropped_as_duplicates"] = len(ext_dropped)
    rep["leakage_check"] = "PASSED: every group is entirely inside one split"
    (out / "split_report.json").write_text(json.dumps(rep, indent=2))
    print("\n" + json.dumps(rep["splits"], indent=2))

    save_group_montage(df, out / "group_audit.png", seed=args.seed)
    save_borderline_montage(df_all, [(i, j, s) for i, j, s in pairs_all], args.min_inliers,
                            out / "borderline_pairs.png", seed=args.seed)
    print(f"\nWrote splits.csv, split_report.json, group_audit.png, borderline_pairs.png to {out}")
    print("NEXT: open group_audit.png (each row should be ONE scan in several variants) before training.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
