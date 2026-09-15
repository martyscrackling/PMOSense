"""
03b_crop_and_deskew.py

Pipeline position:
    dataset/cleaned/ -> [this script] -> dataset/deskewed/ -> 04_resize_images.py

--- REVISION 2 NOTE ---
Revision 1 used a purely pixel-level (local-texture) test for padding.
On real JPEGs the boundary between a flat bar and real tissue is itself
blurry (JPEG compression smears it into a gradient a few pixels wide),
so a per-pixel test leaves a faint residual sliver right at that
boundary - neither clearly "flat" nor clearly "textured". This revision
fixes that by deciding bar removal at the ROW/COLUMN level instead of
the pixel level: a whole row's mean+std is far more stable than any
individual pixel near a blurry edge, so the cut point is sharp instead
of ragged. Region-based (pixel-level) detection is still used, but now
only for the second pass (wedges and anything the row/column pass
didn't catch), where painting-over rather than cropping is used
instead.

Method, in order:
  1. ROTATION: fit a rotated bounding box to the "content" region
     (non-flat-bright-border-connected pixels, same test as before) and
     rotate the image straight if it's tilted by more than
     `angle_thresh` degrees. Skipped entirely for images that don't
     need it (most of them) to avoid unnecessary interpolation blur.
  2. BAR TRIMMING (row/column level): scan in from each of the 4 edges;
     a row (or column) counts as part of a padding bar if its mean
     intensity is above `brightness_thresh` AND its std-dev is below
     `row_std_thresh` (i.e., the WHOLE row is uniformly bright, not
     just a lucky pixel). Stop at the first row/column that fails this
     test - that boundary is used as a hard crop line. This removes
     Image_031-style full-width/height bars cleanly.
  3. WEDGE PAINT-OUT (pixel level): on the now-cropped image, re-run the
     original local-texture-based padding detection (flat AND bright
     AND connected to the border) and paint any matches black. This is
     what removes Image_717-style triangular corner wedges, which
     cropping can never fully remove without cutting into real tissue
     (crops are always rectangular; wedges aren't).

Every decision (rotation angle, crop box, pixels painted) is logged per
image.

Usage:
    python 03b_crop_and_deskew.py --in_dir dataset/cleaned \
        --out_dir dataset/deskewed \
        --brightness_thresh 170 --row_std_thresh 12 \
        --texture_thresh 6 --angle_thresh 3 \
        --log reports/deskew_log.jsonl
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

import pcos_utils as utils


# ---------------------------------------------------------------------------
# Shared pixel-level padding detector (used for rotation-angle estimation
# AND for the final wedge paint-out pass)
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


# ---------------------------------------------------------------------------
# Stage 1: rotation
# ---------------------------------------------------------------------------

def estimate_rotation_angle(img: np.ndarray, texture_thresh: float,
                             brightness_thresh: int, min_content_frac: float):
    h, w = img.shape
    padding = border_connected_padding_mask(img, texture_thresh, brightness_thresh)
    content = 255 - padding
    contour = largest_contour(content)
    if contour is None or cv2.contourArea(contour) < min_content_frac * h * w:
        return 0.0, False  # (angle, found_content)

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
# Stage 2: row/column bar trimming (sharp, aggregate-statistics based)
# ---------------------------------------------------------------------------

def trim_bars(img: np.ndarray, brightness_thresh: int, row_std_thresh: float,
              max_trim_frac: float = 0.35):
    """Strip contiguous bright+uniform rows/columns from each of the 4
    edges. max_trim_frac caps how much can be trimmed from one side, as
    a safety limit in case brightness_thresh is set too aggressively."""
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
    angle, found = estimate_rotation_angle(img, texture_thresh, brightness_thresh, min_content_frac)
    if not found:
        return img, {"status": "skipped_no_content_found", "angle": 0.0,
                      "trim_box": None, "residual_padding_pixels_painted": 0}

    rotated_flag = abs(angle) > angle_thresh
    img_t = rotate_image(img, angle) if rotated_flag else img
    if not rotated_flag:
        angle = 0.0

    trimmed, trim_box = trim_bars(img_t, brightness_thresh, row_std_thresh)
    if trimmed.size == 0:
        return img_t, {"status": "trim_failed", "angle": float(angle),
                        "trim_box": trim_box, "residual_padding_pixels_painted": 0}

    # Second pass: paint out anything the crop couldn't remove (wedges).
    # Dilated by more than the earlier bar-trim pass: a sharp boundary
    # between padding and real background produces elevated local std
    # right at the edge itself (that's what an edge is), so it dodges
    # the "flat" test no matter how texture_thresh is tuned - a few
    # extra pixels of dilation is what actually removes that rim.
    padding_final = border_connected_padding_mask(trimmed, texture_thresh, brightness_thresh)
    padding_dilated = cv2.dilate(padding_final, np.ones((9, 9), np.uint8))
    residual_px = int(np.count_nonzero(padding_dilated))
    result = trimmed.copy()
    result[padding_dilated == 255] = 0

    status = "rotated_trimmed_painted" if rotated_flag else "trimmed_painted"
    return result, {
        "status": status,
        "angle": float(angle),
        "trim_box": list(trim_box),
        "residual_padding_pixels_painted": residual_px,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="dataset/cleaned")
    ap.add_argument("--out_dir", default="dataset/deskewed")
    ap.add_argument("--texture_thresh", type=float, default=6.0,
                     help="Pixel-level local std-dev below this counts as 'flat' "
                          "(used for rotation-angle estimation and wedge paint-out).")
    ap.add_argument("--brightness_thresh", type=int, default=170,
                     help="Intensity above this counts as 'bright' padding, both for the "
                          "pixel-level test and the row/column test.")
    ap.add_argument("--row_std_thresh", type=float, default=12.0,
                     help="A row/column counts as a uniform padding bar if its own std-dev "
                          "is below this - i.e. it's uniformly bright across its full length, "
                          "not just bright on average. Raise if real bars aren't being fully "
                          "trimmed; lower if real tissue rows are being cut into.")
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
        if info["status"] in ("skipped_no_content_found", "trim_failed"):
            needs_review.append(str(rel))

    if needs_review:
        Path(args.review_list).parent.mkdir(parents=True, exist_ok=True)
        Path(args.review_list).write_text("\n".join(needs_review))

    print(f"Processed {len(files)} images -> {args.out_dir}")
    print(f"Flagged for manual review: {len(needs_review)}")
    if needs_review:
        print(f"See {args.review_list}")
    print(f"Full per-image log: {args.log}")
