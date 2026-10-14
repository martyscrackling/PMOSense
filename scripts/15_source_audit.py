"""
15_source_audit.py - which image SOURCES exist, and does any source contain BOTH classes?

Native image size (width x height) is a fingerprint of where a picture came from. If Healthy and
PCOS images never share a size, the class is confounded with the source, and a CNN can win by
recognising the source. If some size contains both classes in useful numbers, training/testing on
that subset removes this confound.

Run on the folder BEFORE resizing (e.g. dataset\\cleaned, or dataset\\raw). Reads headers only.

  python .\\scripts\\15_source_audit.py --data_dir .\\dataset\\cleaned --classes Healthy PCOS ^
         --out_dir .\\reports\\sources
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--classes", nargs=2, required=True)
    ap.add_argument("--out_dir", default="reports/sources")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min_each", type=int, default=30,
                    help="a size counts as 'usable mixed source' if it has at least this many images of EACH class")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for cls in a.classes:
        d = Path(a.data_dir) / cls
        if not d.is_dir():
            sys.exit(f"ERROR: class folder not found: {d}")
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in EXTS:
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                except Exception:
                    continue
                rows.append({"path": str(p.relative_to(a.data_dir)).replace("\\", "/"), "class": cls,
                             "w": w, "h": h, "size": f"{w}x{h}", "ext": p.suffix.lower()})
    df = pd.DataFrame(rows)
    df.to_csv(out / "native_sizes.csv", index=False)
    print(f"{len(df)} images, {df['size'].nunique()} distinct native sizes\n")

    ct = df.pivot_table(index="size", columns="class", values="path", aggfunc="count", fill_value=0)
    for c in a.classes:
        if c not in ct.columns:
            ct[c] = 0
    ct["total"] = ct[a.classes].sum(axis=1)
    ct = ct.sort_values("total", ascending=False)
    ct["mixed_usable"] = (ct[a.classes[0]] >= a.min_each) & (ct[a.classes[1]] >= a.min_each)
    ct.to_csv(out / "size_by_class.csv")
    print(f"Top {a.top} native sizes (by count):")
    print(ct.head(a.top)[a.classes + ["total", "mixed_usable"]].to_string())
    rest = ct.iloc[a.top:]
    if len(rest):
        print(f"  ... {len(rest)} more sizes holding {int(rest['total'].sum())} images "
              f"({', '.join(f'{c} {int(rest[c].sum())}' for c in a.classes)})")

    print("\nFile extensions by class:")
    print(df.pivot_table(index="ext", columns="class", values="path", aggfunc="count", fill_value=0).to_string())

    mixed = ct[ct.mixed_usable]
    print()
    if len(mixed):
        print(f"{len(mixed)} size(s) contain >= {a.min_each} images of EACH class: "
              f"{', '.join(mixed.index)}  ->  {int(mixed[a.classes].sum().sum())} images usable as a "
              f"source-matched subset.")
    else:
        print(f"NO native size contains >= {a.min_each} images of each class. Class and source are fully "
              f"confounded in this folder: a CNN cannot be shown to learn pathology from it.")
    print("(Sizes are read from the files, so run it on a stage BEFORE 03b/04 resizing.)")


if __name__ == "__main__":
    main()
