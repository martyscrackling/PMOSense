"""
18_explain.py - WHERE does the trained model look?  Grad-CAM (+ optional LIME / SHAP) and a
region-ablation test aimed at the source-shortcut problem found in the model-selection stage.

Run it on the selected model, e.g. runs_v2\\efficientnet_b0\\seed0, on the held-out TEST split:

  python .\\scripts\\18_explain.py --run_dir .\\runs_v2\\efficientnet_b0\\seed0 ^
         --splits .\\reports\\splits_v2\\splits.csv --data_dir .\\dataset\\roi_center ^
         --out_dir .\\reports\\explain\\effb0_seed0

  add  --lime  and/or  --shap  for the slower methods (pip install lime shap).

What it writes (out_dir):
  gradcam_examples.png   original + Grad-CAM overlay: confident-correct Healthy, confident-correct PCOS, errors
  gradcam_stats.csv      per image: how much CAM mass falls in the centre / the outer frame / on black pixels,
                         each compared with the AREA of that region (ratio 1.0 = no preference)
  ablation.json          test AUC with the full image, with only the centre kept, and with the centre blanked
  explain_summary.json   the per-class averages and the ablation AUCs in one place
  lime_examples.png / shap_examples.png   (only with --lime / --shap)

HOW TO READ IT (the point of this script):
  * Grad-CAM mass on the central ellipse well above its area share (ratio > 1) and little on black
    pixels / the outer frame  ->  the model looks at tissue.
  * Ratios near or below 1 for the centre, or high ratios on black pixels / the frame  ->  the model reads
    framing, not anatomy (the shortcut found by 13_shortcut_check.py).
  * ablation: if the AUC stays high when the CENTRE is blanked (periphery_only), the label is readable
    from the periphery alone, which holds no ovary. That is a direct sign of a source shortcut.
    (Blanking creates its own edge the model never saw in training, so treat the numbers as a probe,
    not as a score.)
  * Heatmaps explain what the model used; they do not prove it is clinically correct. Show them to
    your gynecologist reviewers and ask whether the highlighted area is a follicle / ovary.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_train_module():
    p = Path(__file__).with_name("11_train_model.py")
    if not p.exists():
        sys.exit(f"ERROR: {p} not found (18_explain.py must sit next to 11_train_model.py)")
    spec = importlib.util.spec_from_file_location("trainmod", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def file_md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def target_layer(model, name):
    if name == "resnet50":
        return model.layer4
    return model.features  # efficientnet_b0, densenet121


class Net:
    """model + normalisation, taking images in [0,1] (N,3,S,S) and returning logits (N,)."""

    def __init__(self, model, mean, std, device):
        self.model, self.device = model, device
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def norm(self, x01):
        return (x01 - self.mean) / self.std

    @torch.no_grad()
    def probs(self, x01, bs=32):
        self.model.eval()
        out = []
        for i in range(0, len(x01), bs):
            xb = self.norm(x01[i:i + bs].to(self.device))
            out.append(torch.sigmoid(self.model(xb)[:, 0]).cpu())
        return torch.cat(out).numpy()


def gradcam(net, name, x01, sign):
    """Grad-CAM for one batch. sign=+1 explains 'PCOS', -1 explains 'Healthy'. Returns (N,S,S) >= 0."""
    model = net.model
    model.eval()
    store = {}
    layer = target_layer(model, name)

    def fwd(_m, _i, out):
        store["a"] = out.detach().clone()  # copy: DenseNet applies an in-place ReLU right after
        out.register_hook(lambda g: store.__setitem__("g", g.detach().clone()))

    h = layer.register_forward_hook(fwd)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            x = net.norm(x01.to(net.device))
            logit = model(x)[:, 0]
            (logit * torch.as_tensor(sign, dtype=logit.dtype, device=logit.device)).sum().backward()
    finally:
        h.remove()
    a, g = store["a"], store["g"]
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x01.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return cam.cpu().numpy()


def region_masks(S):
    yy, xx = np.mgrid[0:S, 0:S]
    u, v = (xx + 0.5) / S - 0.5, (yy + 0.5) / S - 0.5
    central = (u / 0.4) ** 2 + (v / 0.4) ** 2 <= 1.0
    frame = (np.abs(u) > 0.4) | (np.abs(v) > 0.4)
    return central, frame


def show_cam(ax, img_gray, cam):
    ax.imshow(img_gray, cmap="gray", vmin=0, vmax=1)
    if cam is not None and cam.max() > 0:
        ax.imshow(cam / cam.max(), cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g. runs_v2\\efficientnet_b0\\seed0 (needs best.pt + config.json)")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--data_dir", required=True, help="the SAME image folder the model was trained on")
    ap.add_argument("--external_dir", default=None)
    ap.add_argument("--split", default="test", choices=["val", "test", "external"])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--n", type=int, default=12, help="images shown in the Grad-CAM figure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lime", action="store_true")
    ap.add_argument("--shap", action="store_true")
    ap.add_argument("--n_lime", type=int, default=4)
    ap.add_argument("--lime_samples", type=int, default=800)
    ap.add_argument("--n_shap", type=int, default=6)
    ap.add_argument("--n_background", type=int, default=24)
    a = ap.parse_args()

    run = Path(a.run_dir)
    for f in ("best.pt", "config.json"):
        if not (run / f).exists():
            sys.exit(f"ERROR: {run / f} not found")
    cfg = json.loads((run / "config.json").read_text())
    name, S = cfg["model"], int(cfg.get("img_size", 224))
    if cfg.get("splits_md5") and cfg["splits_md5"] != file_md5(a.splits):
        print("WARNING: --splits differs from the splits.csv this model was trained with "
              "(md5 mismatch). The 'test' images may include training images.")
    out = Path(a.out_dir or f"reports/explain/{run.parent.name}_{run.name}")
    out.mkdir(parents=True, exist_ok=True)

    tm = load_train_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tm.build_model(name, pretrained=False)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    model.to(device).eval()
    net = Net(model, tm.IMAGENET_MEAN, tm.IMAGENET_STD, device)

    df = pd.read_csv(a.splits)
    roots = {"main": a.data_dir, "external": a.external_dir}
    d = df[df.split == a.split].reset_index(drop=True)
    if d.empty:
        sys.exit(f"ERROR: no '{a.split}' rows in {a.splits}")
    tf = transforms.Compose([transforms.Resize((S, S)), transforms.ToTensor()])

    def load(root, rel):
        p = Path(root) / rel
        if not p.exists():
            sys.exit(f"ERROR: image not found: {p}  (is --data_dir the folder the model was trained on?)")
        return tf(Image.open(p).convert("L").convert("RGB"))

    print(f"model={name}  run={run}  split={a.split} ({len(d)} images)  device={device}")
    X = torch.stack([load(roots[s], p) for p, s in zip(d.path, d.source)])
    y = d.label.values.astype(int)
    prob = net.probs(X)
    pred = (prob >= 0.5).astype(int)
    try:
        print(f"AUC on {a.split}: {roc_auc_score(y, prob):.4f}   (accuracy at p=0.5: {(pred == y).mean():.3f})")
    except ValueError:
        pass

    # ------------------------------------------------------------- Grad-CAM, all images
    central, frame = region_masks(S)
    a_central, a_frame = central.mean(), frame.mean()
    rows, cams = [], np.zeros((len(d), S, S), np.float32)
    for i in range(0, len(d), 16):
        xb = X[i:i + 16]
        sign = np.where(pred[i:i + 16] == 1, 1.0, -1.0)  # explain the class the model chose
        cb = gradcam(net, name, xb, sign)
        cams[i:i + len(cb)] = cb
    for i in range(len(d)):
        c = cams[i]
        tot = c.sum()
        black = (X[i, 0].numpy() < 10 / 255.0)
        a_black = float(black.mean())
        if tot <= 0:
            rows.append(dict(path=d.path[i], label=y[i], prob_pcos=prob[i], central_mass=np.nan,
                             frame_mass=np.nan, black_mass=np.nan, black_area=a_black))
            continue
        rows.append(dict(path=d.path[i], label=y[i], prob_pcos=prob[i],
                         central_mass=float(c[central].sum() / tot), frame_mass=float(c[frame].sum() / tot),
                         black_mass=float(c[black].sum() / tot) if a_black > 0 else np.nan, black_area=a_black))
    st = pd.DataFrame(rows)
    st["central_enrich"] = st.central_mass / a_central
    st["frame_enrich"] = st.frame_mass / a_frame
    st["black_enrich"] = np.where(st.black_area > 0.01, st.black_mass / st.black_area.clip(lower=1e-9), np.nan)
    st.to_csv(out / "gradcam_stats.csv", index=False)

    names = {0: "Healthy", 1: "PCOS"}
    summary = {"model": name, "run": str(run), "split": a.split, "n_images": int(len(d)),
               "area_central": float(a_central), "area_frame": float(a_frame), "per_class": {}}
    print(f"\nGrad-CAM mass vs area  (1.0 = no preference; central ellipse = {a_central:.0%} of the image, "
          f"outer frame = {a_frame:.0%})")
    print(f"{'class':8s} {'n':>4s} {'central x':>10s} {'frame x':>9s} {'black-pixel x':>14s}")
    for lab in (0, 1):
        s = st[st.label == lab]
        if s.empty:
            continue
        r = dict(n=int(len(s)), central_enrich=float(s.central_enrich.mean()),
                 frame_enrich=float(s.frame_enrich.mean()),
                 black_enrich=float(s.black_enrich.mean()) if s.black_enrich.notna().any() else None)
        summary["per_class"][names[lab]] = r
        be = f"{r['black_enrich']:.2f}" if r["black_enrich"] is not None else "   n/a"
        print(f"{names[lab]:8s} {r['n']:4d} {r['central_enrich']:10.2f} {r['frame_enrich']:9.2f} {be:>14s}")

    # example figure: confident-correct Healthy / PCOS + the worst errors
    rng = np.random.default_rng(a.seed)
    k = max(1, a.n // 3)
    conf = np.abs(prob - 0.5)
    picks = []
    for lab in (0, 1):
        ok = np.where((y == lab) & (pred == lab))[0]
        ok = ok[np.argsort(-conf[ok])][:max(k * 3, 1)]
        picks += [(int(j), "correct") for j in rng.choice(ok, size=min(k, len(ok)), replace=False)] if len(ok) else []
    wrong = np.where(pred != y)[0]
    wrong = wrong[np.argsort(-conf[wrong])][:k]
    picks += [(int(j), "ERROR") for j in wrong]
    ncols = 6
    nblk = int(np.ceil(len(picks) / ncols))
    fig, axs = plt.subplots(2 * nblk, ncols, figsize=(2.3 * ncols, 2.5 * 2 * nblk), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for n_, (j, tag) in enumerate(picks):
        r, c = 2 * (n_ // ncols), n_ % ncols
        g = X[j, 0].numpy()
        axs[r, c].imshow(g, cmap="gray", vmin=0, vmax=1)
        axs[r, c].set_title(f"{names[y[j]]} | p(PCOS)={prob[j]:.2f}\n{tag}", fontsize=8,
                            color="crimson" if tag == "ERROR" else "black")
        show_cam(axs[r + 1, c], g, cams[j])
    fig.suptitle(f"Grad-CAM - {name} ({a.split}); top row image, bottom row heatmap of the class the model chose",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "gradcam_examples.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------- region ablation
    Xc = X * torch.from_numpy(central.astype(np.float32))[None, None]          # keep centre only
    Xp = X * torch.from_numpy((~central).astype(np.float32))[None, None]       # blank the centre
    abl = {}
    for tag, xx in [("full_image", X), ("centre_only", Xc), ("periphery_only", Xp)]:
        p = net.probs(xx)
        try:
            abl[tag] = float(roc_auc_score(y, p))
        except ValueError:
            abl[tag] = None
    (out / "ablation.json").write_text(json.dumps(abl, indent=2))
    summary["ablation_auc"] = abl
    print("\nRegion ablation, AUC on the same images:")
    for k_, v in abl.items():
        print(f"  {k_:15s} {v:.3f}" if v is not None else f"  {k_:15s} n/a")
    if abl.get("periphery_only") is not None and abl["periphery_only"] >= 0.8:
        print("  -> the periphery alone (no centre) still separates the classes: the model can read the "
              "label from framing / source, not only from the ovary.")
    elif abl.get("periphery_only") is not None:
        print("  -> the periphery alone does NOT separate the classes well: the model depends on the centre.")

    # ------------------------------------------------------------- LIME (optional)
    if a.lime:
        try:
            from lime import lime_image
            from skimage.segmentation import slic
        except ImportError:
            print("\nLIME skipped: pip install lime scikit-image")
        else:
            idx = [j for j, _ in picks][:a.n_lime]
            ex = lime_image.LimeImageExplainer(random_state=a.seed)

            def fn(arr):
                t = torch.from_numpy(np.asarray(arr, dtype=np.float32)).permute(0, 3, 1, 2)
                t = t / 255.0 if t.max() > 1.5 else t
                pp = net.probs(t, bs=64)
                return np.stack([1 - pp, pp], axis=1)

            fig, axs = plt.subplots(2, len(idx), figsize=(2.8 * len(idx), 5.6), squeeze=False)
            for c, j in enumerate(idx):
                img = (X[j].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                e = ex.explain_instance(img, fn, labels=(1,), hide_color=0, num_samples=a.lime_samples,
                                        batch_size=64, progress_bar=False,
                                        segmentation_fn=lambda im: slic(im, n_segments=40, compactness=10,
                                                                        start_label=0))
                heat = np.zeros(e.segments.shape, np.float32)
                for sid, w in e.local_exp[1]:
                    heat[e.segments == sid] = w
                m = max(abs(heat).max(), 1e-6)
                axs[0, c].imshow(X[j, 0].numpy(), cmap="gray", vmin=0, vmax=1)
                axs[0, c].set_title(f"{names[y[j]]} | p={prob[j]:.2f}", fontsize=8)
                axs[1, c].imshow(X[j, 0].numpy(), cmap="gray", vmin=0, vmax=1)
                axs[1, c].imshow(heat, cmap="RdBu_r", vmin=-m, vmax=m, alpha=0.55)
                axs[0, c].axis("off")
                axs[1, c].axis("off")
                print(f"  LIME {c + 1}/{len(idx)} done")
            fig.suptitle("LIME: red = supports PCOS, blue = supports Healthy", fontsize=10)
            fig.tight_layout()
            fig.savefig(out / "lime_examples.png", dpi=130)
            plt.close(fig)

    # ------------------------------------------------------------- SHAP (optional)
    if a.shap:
        try:
            import shap
        except ImportError:
            print("\nSHAP skipped: pip install shap")
        else:
            try:
                tr = df[df.split == "train"]
                bg = pd.concat([g.sample(min(len(g), a.n_background // 2), random_state=a.seed)
                                for _, g in tr.groupby("label")])
                BG = torch.stack([load(roots[s], p) for p, s in zip(bg.path, bg.source)])
                idx = [j for j, _ in picks][:a.n_shap]

                class Wrap(torch.nn.Module):
                    def __init__(self, m):
                        super().__init__()
                        self.m = m

                    def forward(self, x):
                        return self.m(x)

                wrapped = Wrap(model).to(device).eval()
                expl = shap.GradientExplainer(wrapped, net.norm(BG.to(device)))
                sv = expl.shap_values(net.norm(X[idx].to(device)), nsamples=100)
                sv = np.array(sv[0] if isinstance(sv, list) else sv)
                if sv.ndim == 5:
                    sv = sv[..., 0]
                heat = sv.sum(axis=1)  # (n,S,S): sum over the 3 identical channels
                fig, axs = plt.subplots(2, len(idx), figsize=(2.8 * len(idx), 5.6), squeeze=False)
                for c, j in enumerate(idx):
                    m = max(np.abs(heat[c]).max(), 1e-9)
                    axs[0, c].imshow(X[j, 0].numpy(), cmap="gray", vmin=0, vmax=1)
                    axs[0, c].set_title(f"{names[y[j]]} | p={prob[j]:.2f}", fontsize=8)
                    axs[1, c].imshow(X[j, 0].numpy(), cmap="gray", vmin=0, vmax=1)
                    axs[1, c].imshow(heat[c], cmap="RdBu_r", vmin=-m, vmax=m, alpha=0.6)
                    axs[0, c].axis("off")
                    axs[1, c].axis("off")
                fig.suptitle("SHAP (gradient explainer): red = pushes toward PCOS, blue = toward Healthy", fontsize=10)
                fig.tight_layout()
                fig.savefig(out / "shap_examples.png", dpi=130)
                plt.close(fig)
            except Exception as e:  # SHAP/torch version mismatches are common; do not lose the other outputs
                print(f"\nSHAP failed ({type(e).__name__}: {e}). Grad-CAM / LIME outputs are unaffected.")

    (out / "explain_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}\\gradcam_examples.png, gradcam_stats.csv, ablation.json, explain_summary.json")


if __name__ == "__main__":
    main()
