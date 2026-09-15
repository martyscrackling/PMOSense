"""
demo_compare.py
Generates a synthetic ultrasound-like image (since no real dataset has
been uploaded yet) and runs it through resize -> Variant A -> Variant B,
so the two enhancement approaches can be compared visually before
running on the real dataset. Delete this script once real data arrives.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import cv2
import matplotlib.pyplot as plt

from pcos_utils import phash  # sanity check import path works
import importlib
resize_mod = importlib.import_module("04_resize_images".replace("-", "_")) if False else None

# --- synthetic ultrasound-like image: low contrast, speckle noise, follicle-like blobs ---
rng = np.random.default_rng(42)
size = 300
base = np.full((size, size), 90, dtype=np.float64)  # low-contrast gray background

# elliptical "ovary" region, slightly brighter stroma
yy, xx = np.mgrid[0:size, 0:size]
ovary_mask = ((xx - 150) ** 2 / 120 ** 2 + (yy - 150) ** 2 / 90 ** 2) < 1
base[ovary_mask] += 25

# follicle-like dark circular blobs (low intensity, round)
follicles = [(110, 110, 14), (150, 100, 10), (190, 120, 12), (130, 160, 9),
             (170, 175, 15), (100, 180, 8), (200, 170, 10)]
for fx, fy, r in follicles:
    fmask = (xx - fx) ** 2 + (yy - fy) ** 2 < r ** 2
    base[fmask] -= 45

# speckle (multiplicative) noise, characteristic of ultrasound
speckle = rng.normal(1.0, 0.25, base.shape)
noisy = base * speckle
noisy = np.clip(noisy, 0, 255).astype(np.uint8)

cv2.imwrite("dataset/raw/synthetic_demo.png", noisy)

# --- resize (aspect-preserving + pad) ---
h, w = noisy.shape
target = 224
scale = target / max(h, w)
resized = cv2.resize(noisy, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
pad = target - resized.shape[0]
resized = cv2.copyMakeBorder(resized, pad // 2, pad - pad // 2, pad // 2, pad - pad // 2,
                              cv2.BORDER_CONSTANT, value=0)

# --- Variant A: minimal (manuscript-literal) ---
variant_a = resized.copy()

# --- Variant B: denoise + CLAHE ---
denoised = cv2.fastNlMeansDenoising(resized, h=10, templateWindowSize=7, searchWindowSize=21)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
variant_b = clahe.apply(denoised)

# --- comparison figure ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
titles = ["Raw (synthetic)", "Resized (224x224,\naspect-preserving)",
          "Variant A\n(manuscript-literal:\nno enhancement)",
          "Variant B\n(denoise + CLAHE)"]
imgs = [noisy, resized, variant_a, variant_b]
for ax, im, title in zip(axes, imgs, titles):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
plt.tight_layout()
plt.savefig("figures/enhancement_variant_comparison.png", dpi=150)
print("Saved figures/enhancement_variant_comparison.png")

# quick contrast metric to make the difference concrete, not just visual
print(f"Std dev (contrast proxy) - resized: {resized.std():.1f}, "
      f"Variant A: {variant_a.std():.1f}, Variant B: {variant_b.std():.1f}")
