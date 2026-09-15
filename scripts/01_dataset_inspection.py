"""
01_dataset_inspection.py

Phase 2 — inspect dataset/raw/ BEFORE any modification and produce a
summary report. Never writes into dataset/raw/.

Usage:
    python 01_dataset_inspection.py --raw_dir dataset/raw --out reports/dataset_summary.json

Expected layout (adjust CLASS_DIRS if your folders differ):
    dataset/raw/PCOS/*.png
    dataset/raw/Healthy/*.png
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import pcos_utils as utils


def inspect(raw_dir: str) -> dict:
    files = utils.list_images(raw_dir)
    summary = {
        "total_files": len(files),
        "valid_images": 0,
        "corrupted_images": [],
        "formats": Counter(),
        "resolutions": Counter(),
        "channels": Counter(),
        "class_distribution": Counter(),
        "per_image": [],  # filename, class, w, h, channels, format, md5
    }

    for path in files:
        rel = path.relative_to(raw_dir)
        # class = top-level subfolder name, e.g. dataset/raw/PCOS/img001.png -> "PCOS"
        cls = rel.parts[0] if len(rel.parts) > 1 else "UNKNOWN"
        summary["formats"][path.suffix.lower()] += 1

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            summary["corrupted_images"].append(str(rel))
            continue

        summary["valid_images"] += 1
        summary["class_distribution"][cls] += 1

        if img.ndim == 2:
            h, w = img.shape
            ch = 1
        else:
            h, w, ch = img.shape
        summary["resolutions"][f"{w}x{h}"] += 1
        summary["channels"][ch] += 1

        summary["per_image"].append({
            "path": str(rel),
            "class": cls,
            "width": w,
            "height": h,
            "channels": ch,
            "md5": utils.md5_of_file(str(path)),
        })

    # Convert Counters to plain dicts for JSON
    for key in ("formats", "resolutions", "channels", "class_distribution"):
        summary[key] = dict(summary[key])

    return summary


def print_human_summary(summary: dict):
    print(f"Total files found:       {summary['total_files']}")
    print(f"Valid images:            {summary['valid_images']}")
    print(f"Corrupted/unreadable:    {len(summary['corrupted_images'])}")
    print(f"Formats:                 {summary['formats']}")
    print(f"Channel counts:          {summary['channels']}")
    print(f"Class distribution:      {summary['class_distribution']}")
    print(f"Distinct resolutions:    {len(summary['resolutions'])}")
    top_res = sorted(summary["resolutions"].items(), key=lambda x: -x[1])[:5]
    print(f"Most common resolutions: {top_res}")
    if summary["corrupted_images"]:
        print("Corrupted files (first 10):")
        for f in summary["corrupted_images"][:10]:
            print(f"  - {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="dataset/raw")
    ap.add_argument("--out", default="reports/dataset_summary.json")
    args = ap.parse_args()

    summary = inspect(args.raw_dir)
    print_human_summary(summary)
    utils.write_json(summary, args.out)
    print(f"\nFull per-image summary written to {args.out}")
