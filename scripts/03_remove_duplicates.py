"""
03_remove_duplicates.py

Phase 3 (continued) — apply a REVIEWED duplicate_report.csv to produce
dataset/cleaned/: one representative image per duplicate group.

This script does NOT decide what counts as a duplicate — it trusts the
'recommended_action' column, which a human should have reviewed/edited
first (per thesis rule: "do not blindly delete near-duplicates without
verification"). Rows still marked "review" are skipped and kept as-is
until someone changes them to "remove" or "keep".

Copies (never moves/deletes) from dataset/raw/ into dataset/cleaned/, so
dataset/raw/ is never touched, satisfying "never overwrite the original
dataset."

Usage:
    python 03_remove_duplicates.py --raw_dir dataset/raw \
        --report reports/duplicate_report.csv \
        --cleaned_dir dataset/cleaned \
        --removal_log reports/removed_duplicates_log.jsonl
"""

import argparse
import csv
import shutil
from pathlib import Path

import pcos_utils as utils


def load_review_decisions(report_path: str):
    to_remove = set()
    with open(report_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["recommended_action"].startswith("remove"):
                to_remove.add(row["suspected_duplicate"])
    return to_remove


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="dataset/raw")
    ap.add_argument("--report", default="reports/duplicate_report.csv")
    ap.add_argument("--cleaned_dir", default="dataset/cleaned")
    ap.add_argument("--removal_log", default="reports/removed_duplicates_log.jsonl")
    args = ap.parse_args()

    to_remove = load_review_decisions(args.report)
    files = utils.list_images(args.raw_dir)

    kept, removed_dup, removed_corrupt = 0, 0, 0
    for path in files:
        rel = path.relative_to(args.raw_dir)

        # Corrupted/unreadable files never make it past this stage either -
        # they were already flagged by 01_dataset_inspection.py, but we
        # re-check here so 03 is safe to run standalone.
        try:
            utils.load_grayscale(str(path))
        except ValueError:
            utils.append_param_log(args.removal_log, {
                "removed_file": str(rel),
                "reason": "corrupted/unreadable image",
            })
            removed_corrupt += 1
            continue

        if str(rel) in to_remove:
            utils.append_param_log(args.removal_log, {
                "removed_file": str(rel),
                "reason": "flagged as duplicate in reviewed report",
            })
            removed_dup += 1
            continue

        dest = Path(args.cleaned_dir) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        kept += 1

    print(f"Kept:                {kept}")
    print(f"Removed (duplicate): {removed_dup}")
    print(f"Removed (corrupted): {removed_corrupt}")
    print(f"Full removal log: {args.removal_log}")
    print(f"Cleaned dataset written to {args.cleaned_dir} (raw/ untouched)")
