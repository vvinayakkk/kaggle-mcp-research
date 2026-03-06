"""
Smart Kaggle notebook generator.
Creates complete, GPU-optimised, submission-ready notebooks tailored for
image classification, NLP, tabular, and general ML tasks.
"""
from __future__ import annotations

import json
import textwrap
from datetime import datetime
from typing import Optional

# ── nbformat is a soft dependency; fall back to raw JSON if unavailable ────────
try:
    import nbformat
    _HAS_NBFORMAT = True
except ImportError:
    _HAS_NBFORMAT = False


# ════════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════════

def generate_kaggle_notebook(
    task_description: str,
    dataset_info: str,
    architecture_description: str,
    competition_slug: str = "",
    task_type: str = "image_classification",
    use_gpu: bool = True,
    num_epochs: int = 30,
    batch_size: int = 32,
    image_size: int = 224,
    extra_notes: str = "",
) -> str:
    """
    Generate a complete, runnable Kaggle notebook (as .ipynb JSON string).

    task_type options:
        image_classification  – EfficientNet / ConvNeXt + torchvision pipeline
        nlp_classification    – BERT / DistilBERT + HuggingFace Trainer
        tabular               – LightGBM + CatBoost + Optuna ensembling
        object_detection      – YOLOv8 + Ultralytics
        general               – generic sklearn/PyTorch template

    Returns: the notebook as a string (save to a .ipynb file before pushing).
    """
    task_type = task_type.lower().strip()

    generators = {
        "image_classification": _image_classification_notebook,
        "nlp_classification":   _nlp_notebook,
        "tabular":              _tabular_notebook,
        "object_detection":     _object_detection_notebook,
    }
    builder = generators.get(task_type, _general_notebook)

    cells_src = builder(
        task_description=task_description,
        dataset_info=dataset_info,
        architecture_description=architecture_description,
        competition_slug=competition_slug,
        num_epochs=num_epochs,
        batch_size=batch_size,
        image_size=image_size,
        extra_notes=extra_notes,
    )

    if _HAS_NBFORMAT:
        nb = nbformat.v4.new_notebook()
        nb.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        nb.metadata["kaggle"] = {
            "accelerator": "GPU" if use_gpu else "None",
            "dataSources": [{
                "sourceType": "competition",
                "sourceId": competition_slug,
            }] if competition_slug else [],
            "isInternetEnabled": True,
        }
        nb.cells = [
            nbformat.v4.new_markdown_cell(src) if src.startswith("#") else nbformat.v4.new_code_cell(src)
            for src in cells_src
        ]
        return nbformat.writes(nb)
    else:
        # Manual JSON construction
        cells = []
        for src in cells_src:
            cell_type = "markdown" if src.startswith("#") else "code"
            cells.append({
                "cell_type": cell_type,
                "metadata":  {},
                "source":    src,
                "outputs":   [] if cell_type == "code" else None,
                "execution_count": None if cell_type == "code" else None,
            })
            if cell_type == "markdown":
                del cells[-1]["outputs"]
                del cells[-1]["execution_count"]

        nb = {
            "nbformat":       4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.10.0"},
            },
            "cells": cells,
        }
        return json.dumps(nb, indent=2)


# ════════════════════════════════════════════════════════════════════════════════
# IMAGE CLASSIFICATION TEMPLATE
# ════════════════════════════════════════════════════════════════════════════════

def _image_classification_notebook(
    task_description, dataset_info, architecture_description,
    competition_slug, num_epochs, batch_size, image_size, extra_notes, **_
):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return [
        f"""# 🏆 Auto-Generated Kaggle Notebook — Image Classification
**Task:** {task_description}
**Dataset:** {dataset_info}
**Architecture:** {architecture_description}
**Generated:** {ts}
{f"**Competition:** {competition_slug}" if competition_slug else ""}
{"**Notes:** " + extra_notes if extra_notes else ""}
""",

        # ── SETUP ────────────────────────────────────────────────────────────────
        """\
# ── 1. SETUP & IMPORTS ──────────────────────────────────────────────────────
import os, sys, json, time, random, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import torchvision
import torchvision.transforms as T
from torchvision import models

from PIL import Image
import matplotlib.pyplot as plt
import sklearn.metrics as skm

warnings.filterwarnings("ignore")

# ── seeds ─────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# ── device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
""",

        # ── CONFIG ───────────────────────────────────────────────────────────────
        f"""\
# ── 2. CONFIGURATION ─────────────────────────────────────────────────────────
CFG = dict(
    # Data
    data_dir      = Path("/kaggle/input/{competition_slug or 'dataset'}"),
    output_dir    = Path("/kaggle/working/outputs"),
    model_dir     = Path("/kaggle/working/models"),

    # Model
    backbone      = "efficientnet_v2_s",   # swap to 'convnext_small' or 'vit_b_16'
    num_classes   = 6,                      # ← set correct number of classes
    image_size    = {image_size},
    pretrained    = True,
    drop_rate     = 0.3,

    # Training
    num_epochs    = {num_epochs},
    batch_size    = {batch_size},
    lr            = 3e-4,
    min_lr        = 1e-6,
    weight_decay  = 1e-4,
    warmup_epochs = 3,
    label_smooth  = 0.1,
    mixup_alpha   = 0.4,
    cutmix_alpha  = 1.0,

    # Checkpoint
    save_best     = True,
    patience      = 8,
    grad_clip     = 5.0,

    # Mixed precision
    amp           = True,

    seed          = 42,
)

for d in [CFG["output_dir"], CFG["model_dir"]]:
    d.mkdir(parents=True, exist_ok=True)

print("Config loaded:", json.dumps({{k: str(v) for k,v in CFG.items()}}, indent=2))
""",

        # ── DATA DISCOVERY ───────────────────────────────────────────────────────
        """\
# ── 3. DATA DISCOVERY ────────────────────────────────────────────────────────
print("\\n=== INPUT FILES ===")
for p in Path("/kaggle/input").rglob("*"):
    print(" " + str(p))

# Auto-detect CSV files
csv_files = list(Path("/kaggle/input").rglob("*.csv"))
print(f"\\nFound {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"  {f}")

# Load train/test CSVs (adjust column names as needed)
train_df, test_df = None, None
for f in csv_files:
    name = f.name.lower()
    if "train" in name and train_df is None:
        train_df = pd.read_csv(f)
        print(f"\\nTrain: {f} — shape {train_df.shape}")
        print(train_df.head(3))
        print(train_df.dtypes)
    elif "test" in name and test_df is None:
        test_df = pd.read_csv(f)
        print(f"\\nTest: {f} — shape {test_df.shape}")
        print(test_df.head(3))

if train_df is None:
    raise FileNotFoundError("Could not find train CSV — check data_dir and column names")

# Infer class mapping
label_col = [c for c in train_df.columns if "label" in c.lower() or "class" in c.lower() or "category" in c.lower()]
label_col = label_col[0] if label_col else train_df.columns[-1]
img_col   = [c for c in train_df.columns if "image" in c.lower() or "file" in c.lower() or "path" in c.lower() or "name" in c.lower()]
img_col   = img_col[0] if img_col else train_df.columns[0]

print(f"\\nImage column: {img_col}  |  Label column: {label_col}")
classes = sorted(train_df[label_col].unique().tolist())
print(f"Classes ({len(classes)}): {classes}")
CFG["num_classes"] = len(classes)
class2idx = {c: i for i, c in enumerate(classes)}
idx2class = {i: c for c, i in class2idx.items()}
""",

        # ── DATASET CLASS ────────────────────────────────────────────────────────
        """\
# ── 4. DATASET & AUGMENTATION ────────────────────────────────────────────────
def get_transforms(split: str, image_size: int):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == "train":
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(0.3, 0.3, 0.3, 0.1),
            T.RandomGrayscale(p=0.05),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.2),
        ])
    else:
        return T.Compose([
            T.Resize(int(image_size * 1.1)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


class ImageDataset(Dataset):
    def __init__(self, df, img_dir, img_col, label_col=None, class2idx=None, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = Path(img_dir)
        self.img_col   = img_col
        self.label_col = label_col
        self.class2idx = class2idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _find_image(self, name):
        # handle various image path formats
        for ext in ["", ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            p = self.img_dir / (name + ext)
            if p.exists(): return p
            p2 = self.img_dir / name
            if p2.exists(): return p2
        # recursive search
        cands = list(self.img_dir.rglob(f"{Path(name).stem}*"))
        if cands: return cands[0]
        raise FileNotFoundError(f"Image not found: {name}")

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        name = str(row[self.img_col])
        path = self._find_image(name)
        img  = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.label_col and self.class2idx:
            label = self.class2idx[row[self.label_col]]
            return img, label
        return img, row[self.img_col]


# Detect image directory
img_dirs = [
    d for d in Path("/kaggle/input").rglob("*")
    if d.is_dir() and len(list(d.glob("*.jpg")) + list(d.glob("*.png"))) > 10
]
print("Image directories found:", img_dirs)
img_dir = img_dirs[0] if img_dirs else Path("/kaggle/input")
print(f"Using image dir: {img_dir}")
""",

        # ── SPLIT & LOADERS ──────────────────────────────────────────────────────
        """\
# ── 5. TRAIN/VAL SPLIT ───────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold

# Encode labels for stratification
train_df["_label_idx"] = train_df[label_col].map(class2idx)

# 80/20 split (or use CV fold 0)
skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(skf.split(train_df, train_df["_label_idx"]))
tr_idx, va_idx = folds[0]

tr_df = train_df.iloc[tr_idx]
va_df = train_df.iloc[va_idx]
print(f"Train: {len(tr_df):,}  |  Val: {len(va_df):,}")

# Class distribution
print("\\nClass distribution (train):")
print(tr_df[label_col].value_counts())

tr_ds = ImageDataset(tr_df, img_dir, img_col, label_col, class2idx,
                      transform=get_transforms("train", CFG["image_size"]))
va_ds = ImageDataset(va_df, img_dir, img_col, label_col, class2idx,
                      transform=get_transforms("val", CFG["image_size"]))

tr_loader = DataLoader(tr_ds, batch_size=CFG["batch_size"],  shuffle=True,
                       num_workers=2, pin_memory=True, drop_last=True)
va_loader = DataLoader(va_ds, batch_size=CFG["batch_size"]*2, shuffle=False,
                       num_workers=2, pin_memory=True)
print(f"Batches — train: {len(tr_loader)}  |  val: {len(va_loader)}")
""",

        # ── MODEL ────────────────────────────────────────────────────────────────
        """\
# ── 6. MODEL ─────────────────────────────────────────────────────────────────
def build_model(backbone: str, num_classes: int, drop_rate: float, pretrained: bool):
    weights = "IMAGENET1K_V1" if pretrained else None
    try:
        m = getattr(models, backbone)(weights=weights)
    except AttributeError:
        # Fallback to efficientnet_v2_s
        m = models.efficientnet_v2_s(weights="IMAGENET1K_V1" if pretrained else None)

    # Replace classifier head
    if hasattr(m, "classifier"):
        in_feat = m.classifier[-1].in_features if hasattr(m.classifier[-1], "in_features") else m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_feat, num_classes),
        )
    elif hasattr(m, "fc"):
        in_feat = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(in_feat, num_classes))
    elif hasattr(m, "head"):
        in_feat = m.head.in_features if hasattr(m.head, "in_features") else m.head.fc.in_features
        m.head  = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(in_feat, num_classes))
    return m


model = build_model(CFG["backbone"], CFG["num_classes"], CFG["drop_rate"], CFG["pretrained"])
model = model.to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {CFG['backbone']}  |  Params: {total_params/1e6:.1f}M")
""",

        # ── TRAINING ─────────────────────────────────────────────────────────────
        """\
# ── 7. LOSS / OPTIMISER / SCHEDULER ─────────────────────────────────────────
class LabelSmoothingCE(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.classes   = classes

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_val  = self.smoothing / (self.classes - 1)
        one_hot     = torch.full_like(pred, smooth_val)
        one_hot.scatter_(1, target.unsqueeze(1), confidence)
        log_prob    = F.log_softmax(pred, dim=1)
        return -(one_hot * log_prob).sum(dim=1).mean()


criterion = LabelSmoothingCE(CFG["num_classes"], CFG["label_smooth"])
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=CFG["lr"],
    steps_per_epoch=len(tr_loader),
    epochs=CFG["num_epochs"],
    pct_start=CFG["warmup_epochs"] / CFG["num_epochs"],
)
scaler = GradScaler(enabled=CFG["amp"])


def mixup_data(x, y, alpha=0.4):
    if alpha <= 0: return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def train_epoch(loader):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        imgs, la, lb, lam = mixup_data(imgs, labels, CFG["mixup_alpha"])
        with autocast(enabled=CFG["amp"]):
            logits = model(imgs)
            loss   = lam * criterion(logits, la) + (1 - lam) * criterion(logits, lb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        scheduler.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(loader):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with autocast(enabled=CFG["amp"]):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        all_preds .extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    loss = total_loss / len(loader.dataset)
    f1   = skm.f1_score(all_labels, all_preds, average="macro")
    acc  = skm.accuracy_score(all_labels, all_preds)
    return loss, f1, acc
""",

        # ── TRAIN LOOP ───────────────────────────────────────────────────────────
        """\
# ── 8. TRAINING LOOP ─────────────────────────────────────────────────────────
best_f1       = 0.0
patience_left = CFG["patience"]
history       = []
best_ckpt     = CFG["model_dir"] / "best_model.pth"

print(f"\\nTraining for {CFG['num_epochs']} epochs on {DEVICE}\\n")
for epoch in range(1, CFG["num_epochs"] + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_epoch(tr_loader)
    va_loss, va_f1, va_acc = eval_epoch(va_loader)
    elapsed = time.time() - t0
    lr_now  = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else CFG["lr"]

    row = dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc,
               va_loss=va_loss, va_f1=va_f1, va_acc=va_acc, lr=lr_now)
    history.append(row)

    improved = va_f1 > best_f1
    if improved:
        best_f1 = va_f1
        patience_left = CFG["patience"]
        if CFG["save_best"]:
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "f1": va_f1, "cfg": CFG}, best_ckpt)
    else:
        patience_left -= 1

    flag = " ← BEST" if improved else ""
    print(f"Ep {epoch:3d}/{CFG['num_epochs']} | "
          f"TR loss={tr_loss:.4f} acc={tr_acc:.3f} | "
          f"VA loss={va_loss:.4f} f1={va_f1:.4f} acc={va_acc:.3f} | "
          f"LR={lr_now:.2e} | {elapsed:.0f}s{flag}")

    if patience_left == 0:
        print(f"\\nEarly stopping at epoch {epoch}.")
        break

print(f"\\nBest Val F1: {best_f1:.4f}")
pd.DataFrame(history).to_csv(CFG["output_dir"] / "history.csv", index=False)
""",

        # ── INFERENCE ────────────────────────────────────────────────────────────
        """\
# ── 9. INFERENCE & SUBMISSION ────────────────────────────────────────────────
# Load best checkpoint
ckpt = torch.load(best_ckpt, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_f1={ckpt['f1']:.4f})")

# TTA (Test-Time Augmentation) — multi-crop average
tta_transforms = [
    get_transforms("val", CFG["image_size"]),                     # centre crop
    T.Compose([T.Resize(CFG["image_size"]+32), T.CenterCrop(CFG["image_size"]),
               T.RandomHorizontalFlip(p=1.0), T.ToTensor(),
               T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
]

@torch.no_grad()
def predict_tta(df, img_dir, img_col):
    model.eval()
    all_probs = None
    for tfm in tta_transforms:
        ds  = ImageDataset(df, img_dir, img_col, transform=tfm)
        dl  = DataLoader(ds, batch_size=CFG["batch_size"]*2, shuffle=False, num_workers=2)
        probs = []
        for imgs, _ in dl:
            imgs = imgs.to(DEVICE)
            with autocast(enabled=CFG["amp"]):
                logits = model(imgs)
            probs.append(F.softmax(logits, dim=-1).cpu().numpy())
        probs = np.concatenate(probs, axis=0)
        all_probs = probs if all_probs is None else all_probs + probs
    return all_probs / len(tta_transforms)

if test_df is not None:
    probs      = predict_tta(test_df, img_dir, img_col)
    pred_idx   = probs.argmax(1)
    pred_labels = [idx2class[i] for i in pred_idx]

    sub_df               = test_df.copy()
    sub_df[label_col]    = pred_labels
    sub_path             = CFG["output_dir"] / "submission.csv"
    sub_df[[img_col, label_col]].to_csv(sub_path, index=False)
    print(f"\\nSubmission saved: {sub_path}")
    print(sub_df[[img_col, label_col]].head(10))
    print("\\nPrediction distribution:")
    print(pd.Series(pred_labels).value_counts())
else:
    print("No test set found — skipping submission generation.")
""",

        # ── RESULTS ──────────────────────────────────────────────────────────────
        """\
# ── 10. RESULTS & VISUALISATION ─────────────────────────────────────────────
hist_df = pd.read_csv(CFG["output_dir"] / "history.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(hist_df["epoch"], hist_df["tr_loss"], label="Train")
axes[0].plot(hist_df["epoch"], hist_df["va_loss"], label="Val")
axes[0].set_title("Loss"); axes[0].legend()

axes[1].plot(hist_df["epoch"], hist_df["va_f1"], color="green")
axes[1].axhline(best_f1, color="red", linestyle="--", label=f"Best={best_f1:.4f}")
axes[1].set_title("Val F1"); axes[1].legend()

axes[2].plot(hist_df["epoch"], hist_df["lr"])
axes[2].set_title("Learning Rate")

plt.tight_layout()
plt.savefig(CFG["output_dir"] / "training_curves.png", dpi=150)
plt.show()

# Save comprehensive results JSON
results = {
    "best_val_f1":  float(best_f1),
    "best_epoch":   int(hist_df["va_f1"].idxmax() + 1),
    "total_epochs": len(hist_df),
    "final_val_acc": float(hist_df["va_acc"].iloc[-1]),
    "backbone":     CFG["backbone"],
    "num_classes":  CFG["num_classes"],
    "classes":      classes,
    "config":       {k: str(v) for k, v in CFG.items()},
    "timestamp":    datetime.now().isoformat(),
}
with open(CFG["output_dir"] / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
for k, v in results.items():
    if k not in ("config", "classes"):
        print(f"  {k}: {v}")

print("\\nOutput files:")
for p in CFG["output_dir"].iterdir():
    print(f"  {p.name} ({p.stat().st_size/1024:.1f} KB)")
""",
    ]


# ════════════════════════════════════════════════════════════════════════════════
# NLP TEMPLATE
# ════════════════════════════════════════════════════════════════════════════════

def _nlp_notebook(task_description, dataset_info, architecture_description,
                  competition_slug, num_epochs, batch_size, extra_notes, **_):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return [
        f"""# 🏆 Auto-Generated Kaggle Notebook — NLP Classification
**Task:** {task_description}  |  **Generated:** {ts}
""",
        """\
# ── SETUP ────────────────────────────────────────────────────────────────────
import os, json, time, warnings, random
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import sklearn.metrics as skm
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
""",
        f"""\
# ── CONFIG ───────────────────────────────────────────────────────────────────
CFG = dict(
    model_name   = "distilbert-base-uncased",   # or 'microsoft/deberta-v3-small'
    max_len      = 256,
    batch_size   = {batch_size},
    num_epochs   = {num_epochs},
    lr           = 2e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    output_dir   = Path("/kaggle/working/outputs"),
)
CFG["output_dir"].mkdir(parents=True, exist_ok=True)
""",
        """\
# ── DATA ─────────────────────────────────────────────────────────────────────
csv_files = list(Path("/kaggle/input").rglob("*.csv"))
train_df = pd.read_csv(next(f for f in csv_files if "train" in f.name.lower()))
test_df  = pd.read_csv(next((f for f in csv_files if "test" in f.name.lower()), csv_files[0]))

# Detect text and label columns
text_col  = next((c for c in train_df.columns if any(k in c.lower() for k in ["text","review","comment","sentence","essay"])), train_df.columns[0])
label_col = next((c for c in train_df.columns if any(k in c.lower() for k in ["label","target","class","sentiment"])), train_df.columns[-1])
print(f"text_col={text_col}  label_col={label_col}")
print(train_df.head(3))

classes   = sorted(train_df[label_col].unique())
class2idx = {c: i for i, c in enumerate(classes)}
train_df[label_col] = train_df[label_col].map(class2idx)
CFG["num_labels"] = len(classes)
print("Classes:", classes)
""",
        """\
# ── DATASET & TRAINING ───────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

class TextDataset(Dataset):
    def __init__(self, df, text_col, label_col=None):
        self.texts  = df[text_col].tolist()
        self.labels = df[label_col].tolist() if label_col and label_col in df.columns else None

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], truncation=True, padding="max_length",
                        max_length=CFG["max_len"], return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

from sklearn.model_selection import train_test_split
tr_df, va_df = train_test_split(train_df, test_size=0.15, stratify=train_df[label_col], random_state=SEED)
tr_ds = TextDataset(tr_df, text_col, label_col)
va_ds = TextDataset(va_df, text_col, label_col)

model = AutoModelForSequenceClassification.from_pretrained(CFG["model_name"], num_labels=CFG["num_labels"])

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"f1": skm.f1_score(p.label_ids, preds, average="macro"),
            "accuracy": skm.accuracy_score(p.label_ids, preds)}

training_args = TrainingArguments(
    output_dir              = str(CFG["output_dir"] / "checkpoints"),
    num_train_epochs        = CFG["num_epochs"],
    per_device_train_batch_size = CFG["batch_size"],
    per_device_eval_batch_size  = CFG["batch_size"] * 2,
    learning_rate           = CFG["lr"],
    weight_decay            = CFG["weight_decay"],
    warmup_ratio            = CFG["warmup_ratio"],
    evaluation_strategy     = "epoch",
    save_strategy           = "epoch",
    load_best_model_at_end  = True,
    metric_for_best_model   = "f1",
    fp16                    = torch.cuda.is_available(),
    report_to               = "none",
    logging_steps           = 50,
    seed                    = SEED,
)
trainer = Trainer(
    model=model, args=training_args,
    train_dataset=tr_ds, eval_dataset=va_ds,
    compute_metrics=compute_metrics,
)
trainer.train()

# Predict on test set
te_ds   = TextDataset(test_df, text_col)
te_preds = trainer.predict(te_ds).predictions.argmax(-1)
idx2class = {i: c for c, i in class2idx.items()}
test_df[label_col] = [idx2class[i] for i in te_preds]
test_df.to_csv(CFG["output_dir"] / "submission.csv", index=False)
print("Done! submission.csv saved.")
""",
    ]


# ════════════════════════════════════════════════════════════════════════════════
# TABULAR TEMPLATE
# ════════════════════════════════════════════════════════════════════════════════

def _tabular_notebook(task_description, dataset_info, architecture_description,
                      competition_slug, num_epochs, batch_size, extra_notes, **_):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return [
        f"# 🏆 Auto-Generated Kaggle Notebook — Tabular\n**Task:** {task_description}  |  **Generated:** {ts}\n",
        """\
import os, json, warnings, numpy as np, pandas as pd
from pathlib import Path
import lightgbm as lgb
import catboost as cb
import optuna
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as skm
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42; np.random.seed(SEED)
OUTPUT = Path("/kaggle/working/outputs"); OUTPUT.mkdir(exist_ok=True)
""",
        """\
csv_files = list(Path("/kaggle/input").rglob("*.csv"))
train_df  = pd.read_csv(next(f for f in csv_files if "train" in f.name.lower()))
test_df   = pd.read_csv(next(f for f in csv_files if "test"  in f.name.lower()))
target    = next(c for c in train_df.columns if c.lower() in ("target","label","class","y"))
id_col    = next((c for c in test_df.columns if c.lower() in ("id","passengerid","rowid")), None)
feat_cols = [c for c in train_df.columns if c != target and c != id_col]

X, y = train_df[feat_cols], train_df[target]
X_test = test_df[feat_cols]
print(f"Train: {X.shape}  |  Test: {X_test.shape}  |  Target: {y.value_counts().to_dict()}")
""",
        """\
# ── OPTUNA TUNING + K-FOLD ENSEMBLE ─────────────────────────────────────────
def lgb_objective(trial):
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate":   trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth":       trial.suggest_int("max_depth", 3, 10),
        "num_leaves":      trial.suggest_int("num_leaves", 20, 200),
        "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    scores = []
    for tr_idx, va_idx in StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, y):
        m = lgb.LGBMClassifier(**params, random_state=SEED, verbose=-1)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        scores.append(skm.f1_score(y.iloc[va_idx], m.predict(X.iloc[va_idx]), average="macro"))
    return np.mean(scores)

study = optuna.create_study(direction="maximize")
study.optimize(lgb_objective, n_trials=30, show_progress_bar=True)
print("Best LGB params:", study.best_params)

# Final ensemble (5-fold)
oof_preds  = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)):
    m = lgb.LGBMClassifier(**study.best_params, random_state=SEED, verbose=-1)
    m.fit(X.iloc[tr], y.iloc[tr])
    oof_preds[va] = m.predict(X.iloc[va])
    test_preds   += m.predict_proba(X_test)[:, 1] / 5

print("OOF F1:", skm.f1_score(y, oof_preds.round(), average="macro"))
sub = test_df[[id_col]] if id_col else pd.DataFrame()
sub[target] = test_preds.round().astype(int)
sub.to_csv(OUTPUT / "submission.csv", index=False)
print("Done! submission.csv saved.")
""",
    ]


# ════════════════════════════════════════════════════════════════════════════════
# OBJECT DETECTION TEMPLATE
# ════════════════════════════════════════════════════════════════════════════════

def _object_detection_notebook(task_description, dataset_info, architecture_description,
                                competition_slug, num_epochs, batch_size, extra_notes, **_):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return [
        f"# 🏆 Auto-Generated Kaggle Notebook — Object Detection\n**Task:** {task_description}  |  **Generated:** {ts}\n",
        f"""\
# Install YOLOv8
import subprocess
subprocess.run(["pip", "install", "ultralytics", "-q"])
from ultralytics import YOLO
from pathlib import Path
import json

OUTPUT = Path("/kaggle/working/outputs"); OUTPUT.mkdir(exist_ok=True)
model = YOLO("yolov8m.pt")   # nano/small/medium/large/xlarge
results = model.train(
    data    = "/kaggle/input/{competition_slug or 'dataset'}/data.yaml",
    epochs  = {num_epochs},
    imgsz   = 640,
    batch   = {batch_size},
    device  = 0,
    project = str(OUTPUT),
    name    = "yolov8_run",
    save    = True,
)
print("Training done:", results)
""",
    ]


# ════════════════════════════════════════════════════════════════════════════════
# GENERAL TEMPLATE
# ════════════════════════════════════════════════════════════════════════════════

def _general_notebook(task_description, dataset_info, architecture_description,
                      competition_slug, num_epochs, batch_size, extra_notes, **_):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return [
        f"# 🏆 Auto-Generated Kaggle Notebook\n**Task:** {task_description}  |  **Generated:** {ts}\n",
        """\
import os, json, numpy as np, pandas as pd
from pathlib import Path
OUTPUT = Path("/kaggle/working/outputs"); OUTPUT.mkdir(exist_ok=True)
print("Files in /kaggle/input:")
for p in Path("/kaggle/input").rglob("*"):
    print(" ", p)
""",
        f"""\
# ── IMPLEMENT YOUR SOLUTION BELOW ────────────────────────────────────────────
# Architecture: {architecture_description}
# Dataset:      {dataset_info}
# Notes:        {extra_notes}

# << YOUR CODE HERE >>
""",
    ]
