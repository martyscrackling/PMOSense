# PCOSense Preprocessing Pipeline

Implements Phases 2–5 of the preprocessing pipeline (dataset inspection →
duplicate removal → resizing → enhancement) for the images that will
train ResNet-50, EfficientNet-B0, and DenseNet-121, as specified in your
thesis manuscript. Follicle detection (Phase 6+) is **not yet
implemented** — see "Follicle detection: the annotation gap" below
before writing that code, since it changes what's needed first.

## Folder structure

```
dataset/
  raw/                 <- put your real dataset here, one subfolder per class:
                          raw/PCOS/*.png, raw/Healthy/*.png
                          NEVER modified by any script.
  cleaned/              output of 03_remove_duplicates.py
  deskewed/             output of 03b_crop_and_deskew.py (fixes white borders/rotation)
  resized/              output of 04_resize_images.py
  enhanced_minimal/     output of 05a (Variant A — manuscript-literal)
  enhanced_extended/    output of 05b (Variant B — + denoise/CLAHE)
  final/                (reserved for 09_prepare_model_dataset.py — not built yet)
reports/                dataset_summary.json, duplicate_report.csv,
                        removal/resize/enhancement logs — all reproducibility
                        records land here
figures/                comparison figures
scripts/                the pipeline itself, run in numeric order
```

## Running it on your real dataset

```bash
cd scripts
python 01_dataset_inspection.py --raw_dir ../dataset/raw --out ../reports/dataset_summary.json
python 02_duplicate_detection.py --raw_dir ../dataset/raw --out ../reports/duplicate_report.csv --threshold 5
# --- open reports/duplicate_report.csv, review flagged pairs, edit
#     recommended_action if you disagree with any, THEN: ---
python 03_remove_duplicates.py --raw_dir ../dataset/raw --report ../reports/duplicate_report.csv --cleaned_dir ../dataset/cleaned
# fixes white borders / rotated frames with white corner fill:
python 03b_crop_and_deskew.py --in_dir ../dataset/cleaned --out_dir ../dataset/deskewed
# check reports/deskew_needs_review.txt afterward - anything listed there
python 04_resize_images.py --in_dir ../dataset/cleaned --out_dir ../dataset/resized --size 224
# pick ONE enhancement variant:
python 05a_enhancement_minimal.py --in_dir ../dataset/resized --out_dir ../dataset/enhanced_minimal
python 05b_enhancement_extended.py --in_dir ../dataset/resized --out_dir ../dataset/enhanced_extended
```

At 13,716 images, step 2 (duplicate detection) is the slow one. It
currently only compares images within the same class folder to keep the
pair count manageable (see the comment in `02_duplicate_detection.py`)
— if that's too slow in practice, the next optimization is an LSH/BK-tree
index instead of all-pairs comparison; ask and I'll build that.

## Variant A vs Variant B — what you're choosing between

See `figures/enhancement_variant_comparison.png` for the visual
(generated on a synthetic ultrasound-like test image, since nox real
images have been uploaded yet — rerun `demo_compare.py` once you upload
one).

|                        | Variant A (05a)                  | Variant B (05b)                       |
|------------------------|-----------------------------------|-----------------------------------------|
| Matches manuscript Ch. III as written | Yes — resize+normalize+dedup only | No — adds a step not currently described |
| Speckle noise          | Untouched                        | Reduced (non-local-means denoising)     |
| Local contrast         | Untouched                        | Improved (CLAHE, clip=2.0, 8×8 tiles)   |
| Risk of inventing structure | None (pass-through)        | Low if clip limit stays conservative — validate visually/with a gynecologist |
| Methodology chapter impact | None                        | Needs a short paragraph documenting the 2 added steps + parameters |

If you pick Variant B, say so and I'll draft that methodology paragraph
for you (Phase 14/J in the original task list).

## Follicle detection: the annotation gap

You chose the manuscript's U-Net/Mask R-CNN approach over classical CV.
That is the medically/architecturally correct choice given what your
Chapter III describes — but it comes with a hard prerequisite that
classical CV doesn't have: **U-Net and Mask R-CNN are supervised
segmentation models. They need pixel-level (or at least bounding-box)
follicle annotations to train on.**

Nothing in your manuscript's dataset description (either Kaggle source)
mentions follicle-level segmentation masks — both datasets appear to be
labeled only at the image level (PCOS-positive / Healthy), which is
enough for the ResNet/EfficientNet/DenseNet classifiers but not enough
to train a follicle segmenter from scratch.

Before I write the follicle-detection scripts, you'll need one of these,
and the choice changes the implementation substantially:

1. **A separate annotated dataset** with follicle segmentation masks
   (some ultrasound follicle-segmentation datasets exist in the
   literature — I can search for candidates if useful) — then U-Net is
   trained normally, supervised, on that dataset.
2. **A pretrained follicle/ovarian-segmentation model** you fine-tune
   instead of training from scratch — reduces the annotation burden but
   you'd still want some labeled images to validate/fine-tune on.
3. **Manual annotation of a subset** of your own images (e.g. with a
   tool like CVAT or Labelme), enough for a small U-Net trained on a
   few hundred masks — realistic for a thesis timeline but is itself a
   task to plan for (who annotates, how many images, inter-annotator
   agreement with your 3 gynecologists).
4. **A weak-supervision bootstrap**: use classical CV (thresholding/
   contours) to generate rough candidate masks, have gynecologists
   correct a sample, then train U-Net on the corrected set — a hybrid
   that uses the classical pipeline as a labeling aid rather than the
   final detector.

None of these are things I can decide for you — they depend on what
data/access you actually have. Once you tell me which path applies,
I'll build the corresponding script(s) (06_follicle_detection.py and
07_follicle_measurement.py, matching the numbering in your original task
list).
