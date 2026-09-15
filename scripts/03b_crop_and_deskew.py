"""
03b_crop_and_deskew.py

Pipeline position:
    dataset/cleaned/ -> [this script] -> dataset/deskewed/ -> 04_resize_images.py

--- REVISION 4 NOTE ---
Revision 3's Hough-line rotation detector was wrong: it found the
*longest* straight line in the image, which on Image_441 was actually
one of the motion-blur streak lines (a real texture artifact, unrelated
to frame rotation), not the actual wedge-cut boundary. Rotating by that
angle didn't straighten the image at all - confirmed by testing both
signs and visually inspecting the result.

Replaced with a geometric method: fit a convex hull to the tissue
region and look for a PAIR of long, similarly-angled hull edges sitting
in DIAGONALLY OPPOSITE quadrants of the image (top-left + bottom-right,
or top-right + bottom-left). This is the actual geometric signature of
a rotated rectangle clipped by a square canvas - a real rotation always
cuts two opposite corners at the same angle, while incidental long hull
edges (from the natural curved fan boundary, or coincidental chords)
don't have a matching opposite-corner partner at a similar angle. Tested
against all 5 known real samples: only Image_441 (the one genuinely
rotated example) triggers a detection, with an angle that visually
straightens it; the other 4 correctly report no rotation needed.
---

--- REVISION 3 NOTE (superseded above, kept for history) ---
Two real bugs found on your Image_629.jpg / Image_441.jpg / Image_479.jpg
samples:

1. GRAY BARS NOT REMOVED (Image_629, Image_479): their bars measure
   ~154-157 brightness, below the old brightness_thresh of 170 - so they
   were being read as legitimate tissue. Real tissue in these same
   images averages ~65-72, so brightness_thresh is lowered to 140,
   which comfortably separates the two with margin on both sides.

2. ROTATION NEVER DETECTED (Image_441): this image's rotation-corner
   fill is BLACK, not white. Revision 2's rotation detector only looked
   for BRIGHT flat border-connected regions to estimate tilt (on
   purpose - see Revision 1/2 notes below on why brightness mattered
   for the natural black sector background) - which means it is
   fundamentally blind to black-filled rotation. Fixed by adding a
   SECOND, independent rotation estimator: a Hough line transform that
   directly finds the long, straight, high-contrast boundary line the
   corner crop itself creates - the rotation artifact is a very sharp
   edge, easily detected as a strong line by cv2.HoughLinesP,
   regardless of whether the surrounding fill is black or white. This
   is now the PRIMARY rotation check; the original brightness-based
   check still runs as a fallback for cases the Hough check misses.
---

Method, in order:
  1. ROTATION (two independent checks, either can trigger a correction):
     a. Hough-line check (primary): find long, straight edges via Canny
        + HoughLinesP. If enough of them cluster tightly around one
        non-axis-aligned angle, that's the tilt - this is what catches
        black-filled rotated frames like Image_441.
     b. Region-based check (fallback): fit a rotated bounding box to
        the "non-flat-bright-border-connected" content region, as in
        earlier revisions - catches bright-filled rotated frames if the
        Hough check happens to miss them (e.g. a very short/small tilt
        with no long clean edge for Hough to grab onto).
  2. BAR TRIMMING (row/column level): scan in from each of the 4 edges;
     a row/column counts as a padding bar if its mean intensity is
     above `brightness_thresh` AND its own std-dev is below
     `row_std_thresh` (i.e. uniformly bright across its full length).
  3. WEDGE PAINT-OUT (pixel level): on the cropped image, re-run local-
     texture-based padding detection (flat AND bright AND connected to
     the border) and paint any matches black - removes corner wedges
     that cropping alone can't (crops are rectangular, wedges aren't).

Every decision is logged per image, including WHICH rotation check (if
either) fired, for debugging future cases.

Usage:
    python 03b_crop_and_deskew.py --in_dir dataset/cleaned \
        --out_dir dataset/deskewed \
        --brightness_thresh 140 --row_std_thresh 12 \
        --texture_thresh 6 --angle_thresh 3 \
        --log reports/deskew_log.jsonl
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

import pcos_utils as utils


# ---------------------------------------------------------------------------
# Rotation check A: geometric convex-hull method (primary)
# ---------------------------------------------------------------------------

def estimate_angle_geometric(img: np.ndarray, min_len_frac: float = 0.15,
                              min_angle_dev: float = 7.0, pair_tol: float = 15.0):
    """Finds the tissue region's convex hull and looks for a pair of long,
    similarly-angled edges in diagonally opposite quadrants - the actual
    geometric signature of a rotated rectangle clipped by a square canvas.
    Returns (angle, found)."""
    h, w = img.shape
    diag = np.hypot(h, w)
    tissue = (img > 30).astype(np.uint8) * 255
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(tissue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, False
    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c).reshape(-1, 2)
    n = len(hull)

    def quadrant(pt):
        x, y = pt
        return (0 if x < w / 2 else 1, 0 if y < h / 2 else 1)

    edges = []
    for i in range(n):
        p1, p2 = hull[i], hull[(i + 1) % n]
        dx, dy = float(p2[0] - p1[0]), float(p2[1] - p1[1])
        length = np.hypot(dx, dy)
        if length < min_len_frac * diag:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        if angle <= -90: angle += 180
        elif angle > 90: angle -= 180
        if angle > 45: angle -= 90
        elif angle <= -45: angle += 90
        if abs(angle) < min_angle_dev:
            continue  # too close to axis-aligned to be a real wedge cut
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        edges.append({"length": length, "angle": angle, "quadrant": quadrant(mid)})

    best_pair = None
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            qa, qb = edges[i]["quadrant"], edges[j]["quadrant"]
            if not (qa[0] != qb[0] and qa[1] != qb[1]):  # must be diagonally opposite
                continue
            if abs(edges[i]["angle"] - edges[j]["angle"]) > pair_tol:
                continue
            total_len = edges[i]["length"] + edges[j]["length"]
            if best_pair is None or total_len > best_pair[0]:
                w_angle = (edges[i]["length"] * edges[i]["angle"] +
                           edges[j]["length"] * edges[j]["angle"]) / total_len
                best_pair = (total_len, w_angle)

    if best_pair is None:
        return 0.0, False
    return best_pair[1], True


# ---------------------------------------------------------------------------
# Rotation check B: region/brightness based (fallback, from earlier revisions)
# ---------------------------------------------------------------------------

def local_std(img: np.ndarray, k: int = 9) -> np.ndarray:
    img_f = img.astype(np.float32)
    mean = cv2.blur(img_f, (k, k))
    mean_sq = cv2.blur(img_f * img_f, (k, k))
    return np.sqrt(np.clip(mean_sq - mean * mean, 0, None))


def border_connected_padding_mask(img: np.ndarray, texture_thresh: float,
                                   brightness_thresh: int, k: int = 9) -> np.ndarray:
    """255 where a pixel is flat+bright AND connected to the image
    border; 0 elsewhere (real tissue, follicle interiors, and the
    sector's natural black background are all excluded)."""
    std = local_std(img, k)
    flat_bright = ((std < texture_thresh) & (img > brightness_thresh)).astype(np.uint8) * 255
    flat_bright = cv2.morphologyEx(flat_bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    h, w = flat_bright.shape
    flood_src = flat_bright.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(w):
        for y in (0, h - 1):
            if flood_src[y, x] == 255 and ff_mask[y + 1, x + 1] == 0:
                cv2.floodFill(flood_src, ff_mask, (x, y), 128)
    for y in range(h):
        for x in (0, w - 1):
            if flood_src[y, x] == 255 and ff_mask[y + 1, x + 1] == 0:
                cv2.floodFill(flood_src, ff_mask, (x, y), 128)
    return (flood_src == 128).astype(np.uint8) * 255


def largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def estimate_angle_region(img: np.ndarray, texture_thresh: float,
                           brightness_thresh: int, min_content_frac: float):
    h, w = img.shape
    padding = border_connected_padding_mask(img, texture_thresh, brightness_thresh)
    content = 255 - padding
    contour = largest_contour(content)
    if contour is None or cv2.contourArea(contour) < min_content_frac * h * w:
        return 0.0, False

    (_, _), (rw, rh), angle = cv2.minAreaRect(contour)
    if rw < rh:
        angle += 90
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    return angle, True


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ---------------------------------------------------------------------------
# Bar trimming (row/column level) + wedge paint-out (pixel level)
# ---------------------------------------------------------------------------

def trim_bars(img: np.ndarray, brightness_thresh: int, row_std_thresh: float,
              max_trim_frac: float = 0.35):
    h, w = img.shape
    row_mean = img.mean(axis=1)
    row_std = img.std(axis=1)
    col_mean = img.mean(axis=0)
    col_std = img.std(axis=0)

    def is_bar(mean_v, std_v):
        return mean_v > brightness_thresh and std_v < row_std_thresh

    max_row_trim = int(h * max_trim_frac)
    max_col_trim = int(w * max_trim_frac)

    top = 0
    while top < max_row_trim and is_bar(row_mean[top], row_std[top]):
        top += 1
    bottom = h
    while (h - bottom) < max_row_trim and bottom > top and is_bar(row_mean[bottom - 1], row_std[bottom - 1]):
        bottom -= 1
    left = 0
    while left < max_col_trim and is_bar(col_mean[left], col_std[left]):
        left += 1
    right = w
    while (w - right) < max_col_trim and right > left and is_bar(col_mean[right - 1], col_std[right - 1]):
        right -= 1

    return img[top:bottom, left:right], (top, bottom, left, right)


# ---------------------------------------------------------------------------
# Full pipeline for one image
# ---------------------------------------------------------------------------

def process(img: np.ndarray, texture_thresh: float, brightness_thresh: int,
            row_std_thresh: float, angle_thresh: float, min_content_frac: float):
    angle_geom, found_geom = estimate_angle_geometric(img)
    angle_region, found_region = estimate_angle_region(img, texture_thresh, brightness_thresh, min_content_frac)

    # Both estimators return angles in the SAME convention (already
    # normalized to the direct cv2.getRotationMatrix2D correction angle -
    # confirmed empirically on Image_441: rotating by the geometric
    # estimator's raw output value, unnegated, produced the correctly
    # straightened result).
    if found_geom and abs(angle_geom) > angle_thresh:
        angle, rotation_source = angle_geom, "geometric"
    elif found_region and abs(angle_region) > angle_thresh:
        angle, rotation_source = angle_region, "region"
    else:
        angle, rotation_source = 0.0, "none"

    rotated_flag = angle != 0.0
    img_t = rotate_image(img, angle) if rotated_flag else img

    trimmed, trim_box = trim_bars(img_t, brightness_thresh, row_std_thresh)
    if trimmed.size == 0:
        return img_t, {"status": "trim_failed", "angle": float(angle),
                        "rotation_source": rotation_source, "trim_box": trim_box,
                        "residual_padding_pixels_painted": 0}

    padding_final = border_connected_padding_mask(trimmed, texture_thresh, brightness_thresh)
    padding_dilated = cv2.dilate(padding_final, np.ones((9, 9), np.uint8))
    residual_px = int(np.count_nonzero(padding_dilated))
    result = trimmed.copy()
    result[padding_dilated == 255] = 0

    status = "rotated_trimmed_painted" if rotated_flag else "trimmed_painted"
    return result, {
        "status": status,
        "angle": float(angle),
        "rotation_source": rotation_source,
        "trim_box": list(trim_box),
        "residual_padding_pixels_painted": residual_px,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="dataset/cleaned")
    ap.add_argument("--out_dir", default="dataset/deskewed")
    ap.add_argument("--texture_thresh", type=float, default=6.0,
                     help="Pixel-level local std-dev below this counts as 'flat' "
                          "(used for the region-based rotation fallback and wedge paint-out).")
    ap.add_argument("--brightness_thresh", type=int, default=140,
                     help="Intensity above this counts as 'bright' padding. Lowered from 170 "
                          "after finding real gray bars around 154-157 with real tissue "
                          "averaging 65-72 in the same images - 140 keeps comfortable margin "
                          "on both sides. Re-check against a histogram if bars still survive.")
    ap.add_argument("--row_std_thresh", type=float, default=12.0,
                     help="A row/column counts as a uniform padding bar if its own std-dev "
                          "is below this - i.e. it's uniformly bright across its full length.")
    ap.add_argument("--angle_thresh", type=float, default=3.0,
                     help="Minimum tilt (degrees) before rotation is applied.")
    ap.add_argument("--min_content_frac", type=float, default=0.02)
    ap.add_argument("--log", default="reports/deskew_log.jsonl")
    ap.add_argument("--review_list", default="reports/deskew_needs_review.txt")
    args = ap.parse_args()

    files = utils.list_images(args.in_dir)
    needs_review = []

    for path in files:
        img = utils.load_grayscale(str(path))
        out_img, info = process(img, args.texture_thresh, args.brightness_thresh,
                                 args.row_std_thresh, args.angle_thresh, args.min_content_frac)

        rel = path.relative_to(args.in_dir)
        dest = Path(args.out_dir) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), out_img)

        utils.append_param_log(args.log, {"file": str(rel), **info})
        if info["status"] == "trim_failed":
            needs_review.append(str(rel))

    if needs_review:
        Path(args.review_list).parent.mkdir(parents=True, exist_ok=True)
        Path(args.review_list).write_text("\n".join(needs_review))

    print(f"Processed {len(files)} images -> {args.out_dir}")
    print(f"Flagged for manual review: {len(needs_review)}")
    if needs_review:
        print(f"See {args.review_list}")
    print(f"Full per-image log: {args.log}")
