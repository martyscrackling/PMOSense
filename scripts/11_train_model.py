"""
11_train_model.py

Model-selection step 2 of 3: train ONE candidate CNN with transfer learning.
Run it once per model (and per seed); 12_compare_models.py then compares them.

Candidates (exactly the three in the manuscript):
    resnet50 | efficientnet_b0 | densenet121

Everything that could favour one model over another is held identical:
same splits.csv, same images, same augmentation, same optimiser, same
schedule, same loss, same early-stopping rule, same seeds. Only the
backbone changes.

Training recipe (a standard, deliberately simple recipe):
  * ImageNet-pretrained weights (IMAGENET1K_V1 for all three, so none gets
    a newer training recipe than the others). Grayscale ultrasound is
    replicated to 3 channels and normalised with ImageNet mean/std.
  * The classification layer is replaced by Dropout(0.3) + Linear(.., 1):
    one logit, sigmoid -> P(PCOS).
  * Phase 1 (--warmup_epochs): backbone frozen (BatchNorm statistics too),
    only the new head trains (lr --lr_head).
  * Phase 2: everything unfrozen, backbone lr --lr_backbone, cosine decay.
  * Loss: BCEWithLogits with pos_weight = n_healthy / n_pcos (training set).
  * Training-time augmentation ONLY (never saved to disk): horizontal flip,
    small rotation / shift / scale, mild brightness+contrast. Validation and
    test images are never augmented. No vertical flip, no elastic warps.
  * Model checkpoint = epoch with the best VALIDATION AUC. The test set is
    never used for any decision here - predictions on it are only written
    out for 12_compare_models.py.

Outputs, under  <out_root>/<model>/seed<k>/ :
    best.pt, config.json, history.csv,
    preds_val.csv, preds_test.csv, (preds_external.csv if present)

IMPORTANT: this script was written without access to PyTorch, so it has
been syntax-checked but NOT executed. Do a 1-epoch smoke run first:
    python 11_train_model.py --model resnet50 --splits ../reports/splits/splits.csv \
        --data_dir ../dataset/enhanced_minimal --epochs 1 --max_per_split 64 --no_pretrained

Usage
    python 11_train_model.py --model resnet50 --seed 0 \
        --splits ../reports/splits/splits.csv \
        --data_dir ../dataset/enhanced_minimal \
        [--external_dir ../dataset_external/enhanced_minimal] \
        --out_root ../runs
"""

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MODEL_NAMES = ["resnet50", "efficientnet_b0", "densenet121"]
HEAD_PREFIX = {"resnet50": "fc", "efficientnet_b0": "classifier", "densenet121": "classifier"}
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------- model
def build_model(name, pretrained):
    if name == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 1))
    elif name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, 1))
    elif name == "densenet121":
        w = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.densenet121(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier.in_features, 1))
    else:
        raise ValueError(name)
    return m


def set_backbone_trainable(model, name, trainable):
    prefix = HEAD_PREFIX[name]
    for pname, p in model.named_parameters():
        p.requires_grad = True if pname.startswith(prefix) else trainable


def set_train_mode(model, backbone_frozen):
    model.train()
    if backbone_frozen:  # keep frozen BatchNorm layers' running stats untouched
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


# ----------------------------------------------------------------------- data
class UltrasoundDataset(Dataset):
    def __init__(self, df, roots, tfm):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(np.float32).tolist()
        self.sources = df["source"].tolist()
        self.roots, self.tfm = roots, tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(Path(self.roots[self.sources[i]]) / self.paths[i]).convert("L").convert("RGB")
        return self.tfm(img), torch.tensor(self.labels[i], dtype=torch.float32)


def make_transforms(size):
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for x, _ in loader:
        probs.append(torch.sigmoid(model(x.to(device)).squeeze(1)).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros(0)


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(np.asarray(y).tolist())) > 1 else float("nan")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--splits", required=True, help="splits.csv from 10_make_splits.py")
    ap.add_argument("--data_dir", required=True, help="Image stage to train on, e.g. dataset/enhanced_minimal")
    ap.add_argument("--external_dir", default=None,
                    help="Same stage for the external dataset (needed only if splits.csv has an 'external' split).")
    ap.add_argument("--out_root", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=7, help="Early-stop patience (epochs, counted after warm-up).")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=0 if os.name == "nt" else 4)
    ap.add_argument("--no_pretrained", action="store_true", help="Random init (offline smoke tests only).")
    ap.add_argument("--max_per_split", type=int, default=0, help="Debug: cap images per split.")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    df = pd.read_csv(args.splits)
    if args.max_per_split:
        df = pd.concat([d.sample(min(len(d), args.max_per_split), random_state=0)
                        for _, d in df.groupby("split")], ignore_index=True)
    roots = {"main": args.data_dir, "external": args.external_dir}
    if (df.split == "external").any() and not args.external_dir:
        print("NOTE: splits.csv has an external split but --external_dir was not given; skipping external predictions.")
        df = df[df.split != "external"]

    parts = {s: df[df.split == s].reset_index(drop=True) for s in ["train", "val", "test", "external"]
             if (df.split == s).any()}
    for need in ("train", "val", "test"):
        if need not in parts:
            raise SystemExit(f"ERROR: no '{need}' rows in {args.splits}")
    print({s: len(d) for s, d in parts.items()})

    # missing files fail loudly now, not 20 minutes into training
    for s, d in parts.items():
        miss = [p for p, src in zip(d.path, d.source) if not (Path(roots[src]) / p).exists()]
        if miss:
            raise SystemExit(f"ERROR: {len(miss)} file(s) from split '{s}' not found under {roots}. First: {miss[0]}")

    train_tf, eval_tf = make_transforms(args.img_size)
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(UltrasoundDataset(parts["train"], roots, train_tf), shuffle=True, drop_last=False, **kw)
    eval_loaders = {s: DataLoader(UltrasoundDataset(d, roots, eval_tf), shuffle=False, **kw)
                    for s, d in parts.items() if s != "train"}

    n_pos = int((parts["train"].label == 1).sum())
    n_neg = int((parts["train"].label == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"train: {n_pos} PCOS / {n_neg} healthy -> pos_weight={pos_weight.item():.3f}")

    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    head_params = [p for n, p in model.named_parameters() if n.startswith(HEAD_PREFIX[args.model])]
    back_params = [p for n, p in model.named_parameters() if not n.startswith(HEAD_PREFIX[args.model])]
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": args.lr_head}, {"params": back_params, "lr": args.lr_backbone}],
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = Path(args.out_root) / args.model / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    history, best = [], {"auc": -1.0, "loss": 1e9, "epoch": 0}
    no_improve, t_start = 0, time.time()
    val_y = parts["val"]["label"].to_numpy()

    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.warmup_epochs
        set_backbone_trainable(model, args.model, trainable=not frozen)
        set_train_mode(model, backbone_frozen=frozen)

        run_loss, n_seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * len(y)
            n_seen += len(y)
        scheduler.step()

        # validation (loss + AUC) - the ONLY signal used for checkpointing / stopping
        val_probs = predict(model, eval_loaders["val"], device)
        eps = 1e-7
        vp = np.clip(val_probs, eps, 1 - eps)
        pw = pos_weight.item()
        val_loss = float(-np.mean(pw * val_y * np.log(vp) + (1 - val_y) * np.log(1 - vp)))
        val_auc = safe_auc(val_y, val_probs)
        history.append({"epoch": epoch, "phase": "warmup" if frozen else "finetune",
                        "train_loss": run_loss / max(n_seen, 1), "val_loss": val_loss, "val_auc": val_auc})
        print(f"epoch {epoch:02d} [{history[-1]['phase']}] train_loss={history[-1]['train_loss']:.4f} "
              f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

        improved = (not np.isnan(val_auc)) and (val_auc > best["auc"] + 1e-6 or
                                                (abs(val_auc - best["auc"]) <= 1e-6 and val_loss < best["loss"]))
        if improved:
            best = {"auc": val_auc, "loss": val_loss, "epoch": epoch}
            torch.save(model.state_dict(), out_dir / "best.pt")
            no_improve = 0
        elif not frozen:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"early stopping at epoch {epoch} (best epoch {best['epoch']})")
                break

    if not (out_dir / "best.pt").exists():  # e.g. val AUC undefined every epoch
        torch.save(model.state_dict(), out_dir / "best.pt")
    train_seconds = time.time() - t_start

    # predictions from the best checkpoint
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    for s, loader in eval_loaders.items():
        probs = predict(model, loader, device)
        pd.DataFrame({"path": parts[s]["path"], "label": parts[s]["label"], "group_id": parts[s]["group_id"],
                      "prob_pcos": probs}).to_csv(out_dir / f"preds_{s}.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    cfg = {k: v for k, v in vars(args).items()}
    cfg.update({"n_params": int(n_params), "best_epoch": best["epoch"], "best_val_auc": best["auc"],
                "train_seconds": round(train_seconds, 1), "device": str(device),
                "torch": torch.__version__, "python": platform.python_version(),
                "splits_md5": file_md5(args.splits), "pos_weight": pos_weight.item(),
                "n_images": {s: int(len(d)) for s, d in parts.items()}})
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"done: best epoch {best['epoch']}, val AUC {best['auc']:.4f}, {train_seconds / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
