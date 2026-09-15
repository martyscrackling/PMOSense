"""
02_duplicate_detection.py

Phase 3 — detect exact, near, and rotated duplicates BEFORE resizing or
enhancement. Produces a duplicate_report.csv for human review. Deletes
nothing itself (see 03_remove_duplicates.py for that, which requires the
reviewed report as input).

Method:
  1. Exact duplicates -> MD5 file hash (byte-identical files).
  2. Near duplicates / rotated duplicates -> perceptual hash (pHash),
     compared at 0/90/180/270 degree rotations of each candidate.
     Two images are flagged if the minimum Hamming distance across the
     four rotations is below HAMMING_THRESHOLD.

Threshold justification:
  pHash produces a 64-bit fingerprint (8x8 low-frequency DCT coefficients).
  A Hamming distance of 0 means visually identical (mod rotation).
  Distances up to ~5 bits out of 64 (~92% agreement) are conventionally
  treated as "near-duplicate" in perceptual-hashing literature (Zauner,
  2010, pHash thresholds). We use 5 as the default and log the exact
  distance for every flagged pair so a human reviewer can judge borderline
  cases (distance 4-8) rather than trust the cutoff blindly.

Usage:
    python 02_duplicate_detection.py --raw_dir dataset/raw \
        --out reports/duplicate_report.csv --threshold 5
"""

import argparse
import csv
from itertools import combinations
from pathlib import Path

import pcos_utils as utils


def build_hash_index(files, raw_dir):
    """Return list of dicts: path, class, md5, phash bits per rotation.

    Unreadable/corrupted files are skipped here (not this script's job -
    01_dataset_inspection.py already reported them) rather than crashing
    the whole duplicate-detection run.
    """
    index = []
    skipped = []
    for path in files:
        try:
            img = utils.load_grayscale(str(path))
        except ValueError:
            skipped.append(str(path))
            continue
        md5 = utils.md5_of_file(str(path))
        rot_hashes = {angle: utils.phash(rot_img) for angle, rot_img in utils.rotations(img)}
        rel_path = Path(path).relative_to(raw_dir)
        index.append({"path": rel_path, "md5": md5, "rot_hashes": rot_hashes})
    if skipped:
        print(f"Skipped {len(skipped)} unreadable file(s) (see dataset_summary.json "
              f"from 01_dataset_inspection.py for the full corrupted-file list).")
    return index


def find_duplicates(index, threshold: int):
    rows = []

    # --- exact duplicates (md5) ---
    md5_groups = {}
    for entry in index:
        md5_groups.setdefault(entry["md5"], []).append(entry)
    for md5, group in md5_groups.items():
        if len(group) > 1:
            anchor = group[0]
            for dup in group[1:]:
                rows.append({
                    "image": str(anchor["path"]),
                    "suspected_duplicate": str(dup["path"]),
                    "duplicate_type": "exact",
                    "similarity_score": 0,
                    "recommended_action": "remove (keep anchor)",
                })

    # --- near / rotated duplicates (pHash), only for images not already
    #     flagged as exact duplicates of each other ---
    # Performance note: naive all-pairs comparison is O(n^2). For the full
    # 13,716-image dataset that is ~94M pairs, which is too slow in pure
    # Python. We cut the search space by only comparing images within the
    # same class folder (a PCOS image being a near-duplicate of a Healthy
    # image is not a realistic labeling scenario here, and this matches
    # how the duplicates described in the manuscript's data notes would
    # actually arise - re-uploads/crops within the same source case).
    # For datasets where cross-class duplicates ARE a real risk, remove
    # this grouping and use an LSH/BK-tree index instead of raw pairs.
    by_class = {}
    for idx, entry in enumerate(index):
        cls = entry["path"].parent.name
        by_class.setdefault(cls, []).append(idx)

    exact_pairs = {(r["image"], r["suspected_duplicate"]) for r in rows}
    candidate_pairs = []
    for idxs in by_class.values():
        candidate_pairs.extend(combinations(idxs, 2))

    for i, j in candidate_pairs:
        a, b = index[i], index[j]
        if (str(a["path"]), str(b["path"])) in exact_pairs:
            continue
        best_dist = 64
        best_angle = 0
        for angle, bits_b in b["rot_hashes"].items():
            d = utils.hamming_distance(a["rot_hashes"][0], bits_b)
            if d < best_dist:
                best_dist, best_angle = d, angle
        if best_dist <= threshold:
            dup_type = "near_duplicate" if best_angle == 0 else f"rotated_duplicate_{best_angle}"
            rows.append({
                "image": str(a["path"]),
                "suspected_duplicate": str(b["path"]),
                "duplicate_type": dup_type,
                "similarity_score": best_dist,
                "recommended_action": "review" if best_dist > 2 else "remove (keep anchor)",
            })

    return rows


def write_report(rows, out_path: str):
    utils.write_json({"note": "see CSV for the actual report"}, out_path.replace(".csv", "_note.json"))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "suspected_duplicate", "duplicate_type",
            "similarity_score", "recommended_action",
        ])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="dataset/raw")
    ap.add_argument("--out", default="reports/duplicate_report.csv")
    ap.add_argument("--threshold", type=int, default=5,
                     help="Max Hamming distance (0-64) to flag as near-duplicate.")
    args = ap.parse_args()

    files = utils.list_images(args.raw_dir)
    print(f"Hashing {len(files)} images...")
    index = build_hash_index(files, args.raw_dir)

    rows = find_duplicates(index, args.threshold)
    write_report(rows, args.out)

    exact = sum(1 for r in rows if r["duplicate_type"] == "exact")
    near = sum(1 for r in rows if r["duplicate_type"] == "near_duplicate")
    rotated = sum(1 for r in rows if r["duplicate_type"].startswith("rotated"))
    print(f"Exact duplicates:   {exact}")
    print(f"Near duplicates:    {near}")
    print(f"Rotated duplicates: {rotated}")
    print(f"Full report: {args.out}")
    print("Nothing has been deleted. Review the report, then run 03_remove_duplicates.py.")