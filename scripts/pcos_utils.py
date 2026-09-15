"""
00_utils.py
Shared helpers for the PCOSense preprocessing pipeline.

Design notes (why things are done this way):
- All functions are pure/stateless where possible so each pipeline stage
  can be re-run independently and results stay reproducible.
- Every stage that changes an image logs its parameters to a JSON
  sidecar or a CSV row, per thesis requirement #11/#12/#13 (record
  parameters, log removed duplicates, preserve filename lineage).
- Nothing in this file ever overwrites a file in dataset/raw/.
"""

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Hashing (exact + perceptual, implemented without third-party deps since
# this environment has no network access to install `imagehash`)
# ---------------------------------------------------------------------------

def md5_of_file(path: str) -> str:
    """Exact-duplicate hash: identical bytes -> identical hash."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(image_gray: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    """
    Perceptual hash via DCT (same algorithm family as the `imagehash`
    library's phash, reimplemented here with cv2.dct so we don't depend
    on a package that can't be installed in this offline environment).

    Returns a flat boolean array of length hash_size**2.
    """
    img_size = hash_size * highfreq_factor
    resized = cv2.resize(image_gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low)
    return (dct_low > med).flatten()


def hamming_distance(bits_a: np.ndarray, bits_b: np.ndarray) -> int:
    return int(np.count_nonzero(bits_a != bits_b))


def rotations(image_gray: np.ndarray):
    """Yield (angle, rotated_image) for 0/90/180/270 — used to catch
    rotated-duplicate pairs as flagged in the manuscript's data-quality
    notes ('duplicate images... for robustness testing')."""
    yield 0, image_gray
    yield 90, cv2.rotate(image_gray, cv2.ROTATE_90_CLOCKWISE)
    yield 180, cv2.rotate(image_gray, cv2.ROTATE_180)
    yield 270, cv2.rotate(image_gray, cv2.ROTATE_90_COUNTERCLOCKWISE)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def list_images(root: str, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def write_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def append_param_log(log_path: str, entry: dict):
    """Append one JSON-lines record of parameters used, per-image or
    per-run. Keeps a reproducibility trail without needing a database."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
