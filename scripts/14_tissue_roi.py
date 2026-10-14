"""
14_tissue_roi.py - neutralise FRAMING so a classifier cannot use black margins / how much of the
frame the ovary fills. For every image it finds the tissue region and re-crops to it, then resizes
to a square. File names, class folders and extensions are kept, so the same splits.csv still works
with --data_dir pointing at the output folder.

Modes:
  bbox   : crop the tissue bounding box (+ margin), stretch to a square.
  center : like bbox, then keep only the central --center_frac of that box (drops the black fan
           corners and the outer edge) before resizing. Stronger framing removal.

This REMOVES framing cues only. It cannot remove other differences between the classes (zoom level
of the follicles, scanner look). Re-run 13_shortcut_check.py on the output to measure what is left.

  python .\\scripts\\14_tissue_roi.py --in_dir .\\dataset\\enhanced_extended --out_dir .\\dataset\\roi_center ^
         --mode center --center_frac 0.7
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
import cv2
import numpy as np
import pandas as pd

cv2.ocl.setUseOpenCL(False)
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def tissue_box(im, thr=25):
    g = cv2.GaussianBlur(im, (5, 5), 0)
    m = (g > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.where(lab == k)
    # percentiles so isolated bright specks attached to the region do not stretch the box
    x0, x1 = np.percentile(xs, [0.5, 99.5])
    y0, y1 = np.percentile(ys, [0.5, 99.5])
    return int(x0), int(y0), int(x1) + 1, int(y1) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=["bbox", "center"], default="center")
    ap.add_argument("--center_frac", type=float, default=0.7)
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--thr", type=int, default=25)
    a = ap.parse_args()

    root, out = Path(a.in_dir), Path(a.out_dir)
    files = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in EXTS]
    if not files:
        sys.exit(f"ERROR: no images under {root}")
    log, failed = [], []
    for p in files:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            failed.append(str(p))
            continue
        h, w = im.shape
        box = tissue_box(im, a.thr)
        if box is None or (box[2] - box[0]) < 20 or (box[3] - box[1]) < 20:
            box = (0, 0, w, h)
            failed.append(f"{p} (no tissue found; copied whole)")
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        if a.mode == "center":
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            bw, bh = bw * a.center_frac, bh * a.center_frac
            x0, x1, y0, y1 = cx - bw / 2, cx + bw / 2, cy - bh / 2, cy + bh / 2
        mx, my = bw * a.margin, bh * a.margin
        x0, x1 = int(max(0, x0 - mx)), int(min(w, x1 + mx))
        y0, y1 = int(max(0, y0 - my)), int(min(h, y1 + my))
        crop = im[y0:y1, x0:x1]
        res = cv2.resize(crop, (a.size, a.size), interpolation=cv2.INTER_AREA if crop.shape[0] > a.size
                         else cv2.INTER_CUBIC)
        dst = out / p.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if dst.suffix.lower() in (".jpg", ".jpeg") else []
        cv2.imwrite(str(dst), res, params)
        log.append({"path": str(p.relative_to(root)).replace("\\", "/"), "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "in_w": w, "in_h": h})
    Path("reports").mkdir(exist_ok=True)
    pd.DataFrame(log).to_csv(Path("reports") / f"roi_log_{out.name}.csv", index=False)
    print(f"wrote {len(log)} images to {out}  (mode={a.mode}, center_frac={a.center_frac})")
    if failed:
        print(f"{len(failed)} problem file(s), first few: {failed[:5]}")


if __name__ == "__main__":
    main()
