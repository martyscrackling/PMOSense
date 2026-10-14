"""
13_shortcut_check.py - can the class be predicted WITHOUT looking at anatomy?

Fits simple models on trivial per-image statistics (brightness, share of black pixels, size of the
non-black region, black margins, sharpness, histogram bins, file size) using the SAME train/val/test
split, and reports AUC. If these crude features already separate Healthy from PCOS, then a high CNN
score cannot be trusted as evidence of learning pathology, and model selection by AUC is unreliable.

  python .\\scripts\\13_shortcut_check.py --splits .\\reports\\splits\\splits.csv ^
         --data_dir .\\dataset\\enhanced_extended --out_dir .\\reports\\shortcut
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

cv2.ocl.setUseOpenCL(False)


def feats(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    h, w = im.shape
    f = {}
    f["mean"] = im.mean()
    f["std"] = im.std()
    f["frac_black"] = (im < 10).mean()
    f["frac_bright"] = (im > 200).mean()
    tissue = im > 25
    f["tissue_frac"] = tissue.mean()
    ys, xs = np.where(tissue)
    if len(xs):
        f["bbox_w"] = (xs.max() - xs.min() + 1) / w
        f["bbox_h"] = (ys.max() - ys.min() + 1) / h
        f["cx"] = xs.mean() / w
        f["cy"] = ys.mean() / h
    else:
        f["bbox_w"] = f["bbox_h"] = f["cx"] = f["cy"] = 0.0
    f["black_rows"] = (im.max(axis=1) < 25).mean()
    f["black_cols"] = (im.max(axis=0) < 25).mean()
    f["sharp"] = cv2.Laplacian(im, cv2.CV_64F).var()
    for k, v in enumerate(np.histogram(im, bins=8, range=(0, 256))[0] / im.size):
        f[f"hist{k}"] = v
    f["file_kb"] = os.path.getsize(path) / 1024
    f["h"], f["w"] = h, w
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="reports/shortcut")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.splits)
    df = df[df.split.isin(["train", "val", "test"])].reset_index(drop=True)
    def resolve(p):
        """splits.csv paths come from one folder; another stage's folder may use another extension."""
        q = Path(a.data_dir) / p
        if q.exists():
            return str(q)
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG"):
            r = q.with_suffix(ext)
            if r.exists():
                return str(r)
        return str(q)

    rows, keep = [], []
    for k, p in enumerate(df.path):
        f = feats(resolve(p))
        if f is not None:
            rows.append(f)
            keep.append(k)
    X = pd.DataFrame(rows)
    d = df.iloc[keep].reset_index(drop=True)
    X.assign(path=d.path, label=d.label, split=d.split).to_csv(out / "trivial_features.csv", index=False)

    tr = (d.split == "train").values
    print(f"{len(d)} images, {X.shape[1]} trivial features. Train class balance: "
          f"{d.label[tr].mean():.2f} PCOS\n")
    models = {
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        "gradient boosting": HistGradientBoostingClassifier(max_depth=3, max_iter=150, random_state=a.seed),
    }
    res = []
    for name, m in models.items():
        m.fit(X[tr], d.label[tr])
        for s in ["val", "test"]:
            msk = (d.split == s).values
            if msk.any() and d.label[msk].nunique() == 2:
                auc = roc_auc_score(d.label[msk], m.predict_proba(X[msk])[:, 1])
                res.append({"model": name, "split": s, "auc": round(auc, 3)})
    r = pd.DataFrame(res)
    print(r.to_string(index=False))
    r.to_csv(out / "shortcut_auc.csv", index=False)

    # which single feature alone is most predictive (train+val+test pooled; just descriptive)
    single = {c: abs(roc_auc_score(d.label, X[c]) - 0.5) + 0.5 for c in X.columns if X[c].nunique() > 1}
    top = sorted(single.items(), key=lambda kv: -kv[1])[:6]
    print("\nStrongest single features (AUC folded to >= 0.5, pooled over all kept images):")
    for c, v in top:
        print(f"  {c:<12} {v:.3f}   mean Healthy {X[c][d.label == 0].mean():.3f} | PCOS {X[c][d.label == 1].mean():.3f}")
    print("\nReading it: trivial-feature AUC near 0.5 = no obvious shortcut. Around 0.7+ = the class is "
          "partly predictable from framing/brightness alone; ~0.9+ = a CNN's high AUC mostly proves "
          "nothing about follicles.")


if __name__ == "__main__":
    main()
