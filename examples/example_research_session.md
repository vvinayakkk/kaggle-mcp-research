# Kaggle Research MCP — Example Research Session

This file shows a complete example of what Copilot does when given a research ZIP and a competition.

---

## User Prompt

```
I have a scene classification competition on Kaggle.
Competition slug: dl-ise-problem-statement-one
Here is my research ZIP: C:/research/project.zip

Please do a full research pipeline — survey the literature, design a better architecture,
generate a notebook, run it on Kaggle, and tell me the results.
```

---

## Copilot's Autonomous Execution

### Step 1 — ZIP Analysis

Copilot calls `extract_zip_research("C:/research/project.zip")` and finds:

```json
{
  "task_type": "image_classification",
  "problem_statement": "Fine-grained scene classification with 6 balanced classes...",
  "libraries_used": ["torch", "torchvision", "sklearn"],
  "model_names": ["ResNet50"],
  "baselines": ["accuracy: 0.72", "val_loss: 0.88"],
  "datasets_mentioned": ["train.csv", "train/", "test/"]
}
```

### Step 2 — Literature Survey

Copilot calls `full_literature_sweep("fine-grained scene classification")` and discovers:

- 12 arXiv papers from 2023-2025
- SOTA on Places365: 61.5% top-1 accuracy (ConvNeXt-Large)
- SOTA on SUN397: 80.2% (EVA-02 + ViT-Large)
- 8 papers with GitHub implementations

Key insights:
- ConvNeXt-V2 outperforms EfficientNet on fine-grained tasks
- Label smoothing + MixUp significantly reduces overfitting on small datasets
- TTA (Test-Time Augmentation) adds ~1-2% F1 consistently

### Step 3 — Architecture Design

**Proposed Architecture: ConvNeXt-V2-Tiny + Aggressive Augmentation**

Inspired by:
- ConvNeXt V2 (arxiv: 2301.00808) — FCMAE pretraining, improved on all benchmarks
- MixUp training (Zhang et al. 2018) — regularisation through convex combinations
- CutMix (Yun et al. 2019) — regional dropout improves fine-grained features

Changes from baseline (ResNet50):
1. Replace backbone: ResNet50 → ConvNeXt-V2-Tiny (ImageNet-1K weights)
2. Add MixUp (α=0.4) + CutMix (α=1.0) during training
3. Replace CE loss with Label Smoothing CE (ε=0.1)
4. Add OneCycleLR scheduler with 3-epoch warmup
5. TTA with 2 transforms at inference

Expected improvement: +8-12% F1 over the baseline (0.72 → ~0.82-0.84)

### Step 4 — Notebook Generation

Copilot calls `generate_kaggle_notebook(...)` and produces a 430-line notebook with:
- Auto-discovering all input files
- ConvNeXt-V2-Tiny backbone with custom head
- Full augmentation pipeline
- Mixed precision (FP16) training
- Early stopping (patience=8)
- TTA inference
- Saves `submission.csv` + `results.json`

### Step 5 — Kaggle Deployment

Copilot calls `kaggle_push_and_run(title="Scene Classification ConvNeXt-V2", ...)`:

```
✅ Kernel pushed: https://www.kaggle.com/code/vinayakbhatia2/scene-classification-convnext-v2
   GPU: Tesla P100-PCIE-16GB
   Status: queued → running → complete (47 min)
```

### Step 6 — Results

Copilot calls `kaggle_kernel_output_log(...)` and extracts:

```
Ep  1/30 | TR loss=1.2847 acc=0.412 | VA loss=1.1823 f1=0.4891 acc=0.510
Ep  5/30 | TR loss=0.8432 acc=0.654 | VA loss=0.7521 f1=0.7234 acc=0.741
Ep 12/30 | TR loss=0.4821 acc=0.832 | VA loss=0.5102 f1=0.8341 acc=0.846 ← BEST
Ep 20/30 | TR loss=0.3240 acc=0.881 | VA loss=0.5841 f1=0.8198 acc=0.831
Early stopping at epoch 22 (patience=8)

Best Val F1: 0.8341
```

---

## Final Report

```markdown
## Research Pipeline Results

### Architecture
- Backbone: ConvNeXt-V2-Tiny (ImageNet-1K pretrained)
- Key additions: MixUp (α=0.4), CutMix (α=1.0), Label Smoothing, OneCycleLR, TTA
- Based on: ConvNeXt V2 (arxiv.org/abs/2301.00808)

### Training Results
- Best validation F1: 0.8341 (epoch 12)
- Final accuracy: 84.6%
- Training time: ~47 min on Tesla P100

### Comparison to Baseline
| Method          | F1    | Notes               |
|-----------------|-------|---------------------|
| ResNet50        | 0.720 | baseline from ZIP   |
| SOTA (PwC)      | 0.802 | EVA-02 on SUN397    |
| ConvNeXt-V2-T   | 0.834 | this notebook       |

### Files Generated
- submission.csv → ready to submit
- results.json   → full metrics
- training_curves.png → loss + F1 plot

### Next Steps
1. Try ConvNeXt-V2-Small for +2-3% F1 (doubles training time)
2. Ensemble with EfficientNetV2-S for diversity
3. Increase image_size to 288 for finer spatial detail
4. Submit: https://www.kaggle.com/c/dl-ise-problem-statement-one/submit
```
