"""
04_resize_images.py

Phase 4 — resize cleaned images to a uniform size for the CNNs.

Method comparison (why aspect-ratio-preserving resize + padding was
chosen over the alternatives):

  1. Direct resize (stretch to 224x224):
     Simple, and what the manuscript literally says ("resizing into
     224x224"). But it distorts the width/height ratio of non-square
     ultrasound frames, which stretches follicle shapes non-uniformly -
     a circular follicle can become elliptical. Bad for anything that
     later measures follicle geometry (circularity, aspect ratio).

  2. Aspect-ratio-preserving resize + padding (letterbox) -- CHOSEN:
     Scales the longer side to fit 224 and pads the shorter side with
     a constant (black) border. Preserves true shape/aspect ratio of
     anatomical structures. The one cost is some wasted border pixels,
     which is harmless for CNN classification and is the standard
     approach in medical imaging pipelines for this reason.

  3. Center crop:
     Preserves aspect ratio and avoids padding artifacts, but risks
     cropping out ovarian tissue near the frame edges - not acceptable
     here since we cannot guarantee the ovary/follicles are centered
     in every source image.

Follicle-measurement consistency:
  Because U-Net/Mask R-CNN follicle detection is meant to run on these
  same standardized images (per your Q&A decision), every image gets a
  recorded scale_factor and pad offsets in resize_log.jsonl. Any
  follicle measured in pixels on the 224x224 image can be converted
  back to original-resolution pixels via:
      original_px = (resized_px - pad_offset) / scale_factor
  This is what prevents the resize step from silently corrupting
  follicle size measurements (per thesis rule #6).

Usage:
    python 04_resize_images.py --in_dir dataset/cleaned \
        --out_dir dataset/resized --size 224 \
        --log reports/resize_log.jsonl
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

import pcos_utils as utils


def resize_with_padding(img: np.ndarray, target: int):
    h, w = img.shape
    scale = target / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = target - new_w
    pad_h = target - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 borderType=cv2.BORDER_CONSTANT, value=0)
    return padded, scale, (left, top)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="dataset/deskewed")
    ap.add_argument("--out_dir", default="dataset/resized")
    ap.add_argument("--size", type=int, default=224,
                     help="224 is compatible with ResNet-50, EfficientNet-B0, "
                          "and DenseNet-121's default ImageNet input head.")
    ap.add_argument("--log", default="reports/resize_log.jsonl")
    args = ap.parse_args()

    files = utils.list_images(args.in_dir)
    for path in files:
        img = utils.load_grayscale(str(path))
        out_img, scale, offset = resize_with_padding(img, args.size)

        rel = path.relative_to(args.in_dir)
        dest = Path(args.out_dir) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), out_img)

        utils.append_param_log(args.log, {
            "file": str(rel),
            "original_size": [img.shape[1], img.shape[0]],
            "target_size": args.size,
            "scale_factor": scale,
            "pad_offset_xy": list(offset),
        })

    print(f"Resized {len(files)} images to {args.size}x{args.size} "
          f"(aspect-preserving + padding). Log: {args.log}")
