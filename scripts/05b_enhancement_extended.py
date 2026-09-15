"""
05b_enhancement_extended.py

VARIANT B - an extension beyond the manuscript's stated methodology.
Adds two steps, each independently justified for ultrasound rather than
applied by default:

1. Speckle-noise reduction: cv2.fastNlMeansDenoising
   Ultrasound images have multiplicative speckle noise (a known
   ultrasound artifact, not Gaussian sensor noise), which is why we use
   a non-local-means denoiser tuned for texture-preserving smoothing
   rather than a plain Gaussian blur (which would blur follicle
   boundaries indiscriminately). Median/bilateral filtering are the
   other historically-used options for this in ultrasound literature
   (see your Chapter II sources on speckle denoising) - non-local-means
   is used here because it best preserves edges while removing
   grain-like speckle, but bilateral filtering is a reasonable swap if
   you want faster runtime at some cost to edge preservation.

2. Contrast enhancement: CLAHE (Contrast Limited Adaptive Histogram
   Equalization)
   Ultrasound images are often low-contrast; CLAHE improves local
   contrast (making follicle boundaries against ovarian stroma more
   visible) without the global over-brightening that plain histogram
   equalization causes. The "contrast limit" (clipLimit) caps how much
   any single tile can be amplified, which is what keeps this from
   inventing structure that isn't there - a global equalization or an
   aggressive clip limit CAN create false edges, which is exactly the
   "artificial structures" risk your rules flag. clipLimit=2.0 and an
   8x8 tile grid are conservative starting values; treat both as
   hyperparameters to validate visually and, ideally, with a
   gynecologist, not as fixed constants.

If you adopt this variant, your methodology chapter needs a short
paragraph documenting these two steps, their parameters, and why they
were added - which is exactly what this docstring is drafted to become.

Usage:
    python 05b_enhancement_extended.py --in_dir dataset/resized \
        --out_dir dataset/enhanced_extended \
        --denoise_h 10 --clahe_clip 2.0 --clahe_grid 8
"""

import argparse
from pathlib import Path

import cv2

import pcos_utils as utils


def enhance_extended(img, denoise_h: float, clahe_clip: float, clahe_grid: int):
    denoised = cv2.fastNlMeansDenoising(img, h=denoise_h, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    enhanced = clahe.apply(denoised)
    return enhanced


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="dataset/resized")
    ap.add_argument("--out_dir", default="dataset/enhanced_extended")
    ap.add_argument("--denoise_h", type=float, default=10,
                     help="Filter strength for non-local-means denoising. Higher = more smoothing.")
    ap.add_argument("--clahe_clip", type=float, default=2.0,
                     help="CLAHE contrast limit. Higher = more aggressive local contrast.")
    ap.add_argument("--clahe_grid", type=int, default=8,
                     help="CLAHE tile grid size (NxN).")
    ap.add_argument("--log", default="reports/enhancement_params_log.json")
    args = ap.parse_args()

    files = utils.list_images(args.in_dir)
    for path in files:
        img = utils.load_grayscale(str(path))
        out_img = enhance_extended(img, args.denoise_h, args.clahe_clip, args.clahe_grid)
        rel = path.relative_to(args.in_dir)
        dest = Path(args.out_dir) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), out_img)

    utils.write_json({
        "denoise_h": args.denoise_h,
        "clahe_clip": args.clahe_clip,
        "clahe_grid": args.clahe_grid,
        "note": "Applied uniformly to all images in this run.",
    }, args.log)

    print(f"Variant B (denoise + CLAHE) written for {len(files)} images "
          f"to {args.out_dir}. Params logged to {args.log}")
