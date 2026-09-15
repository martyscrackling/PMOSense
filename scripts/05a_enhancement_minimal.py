"""
05a_enhancement_minimal.py

VARIANT A - matches what the manuscript's Chapter III actually states:
"resizing, normalization into [0,1] range" - nothing else. No filtering,
no CLAHE, no denoising.

This is the version to use if you want the implementation to match your
current methodology chapter exactly, with zero deviation to defend in
your oral exam.

Output images are saved as 8-bit PNGs for visual inspection (0-255), but
the normalization to [0,1] float is what actually feeds the CNN - that
conversion happens at model-input time, not as a saved image file (see
09_prepare_model_dataset.py, not yet requested).

Usage:
    python 05a_enhancement_minimal.py --in_dir dataset/resized \
        --out_dir dataset/enhanced_minimal
"""

import argparse
from pathlib import Path

import cv2

import pcos_utils as utils


def enhance_minimal(img):
    # "Enhancement" here is a no-op on pixel content - the manuscript's
    # own pipeline has no enhancement stage. We keep this as an explicit
    # pass-through function (rather than skipping the stage entirely) so
    # the folder structure and script numbering stay consistent with
    # Variant B, making the two easy to diff.
    return img.copy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="dataset/resized")
    ap.add_argument("--out_dir", default="dataset/enhanced_minimal")
    args = ap.parse_args()

    files = utils.list_images(args.in_dir)
    for path in files:
        img = utils.load_grayscale(str(path))
        out_img = enhance_minimal(img)
        rel = path.relative_to(args.in_dir)
        dest = Path(args.out_dir) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), out_img)

    print(f"Variant A (manuscript-literal, no enhancement) written for "
          f"{len(files)} images to {args.out_dir}")
