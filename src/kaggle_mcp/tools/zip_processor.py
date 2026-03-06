"""
ZIP / directory processor.
Extracts research material uploaded by the user, infers problem type,
identifies existing datasets, baselines, and code patterns.
"""
from __future__ import annotations

import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Optional


# ════════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════════

def extract_and_analyze_zip(zip_path: str, extract_to: str = "") -> str:
    """
    Extract a user-supplied ZIP and produce a structured summary:
      - list of extracted files (type, size)
      - inferred task type
      - datasets mentioned
      - baseline scores found
      - code patterns (libraries, model names, metrics)
      - problem statement (if found in a README/PDF/DOCX)
    Returns JSON string.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        return json.dumps({"error": f"File not found: {zip_path}"})
    if not zipfile.is_zipfile(zip_path):
        return json.dumps({"error": f"Not a valid ZIP file: {zip_path}"})

    root = Path(extract_to) if extract_to else zip_path.parent / (zip_path.stem + "_extracted")
    root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
        names = zf.namelist()

    return analyze_directory(str(root))


def analyze_directory(dir_path: str) -> str:
    """
    Analyze an already-extracted directory and return a structured summary.
    Works on any directory — ZIP extraction is optional.
    """
    root = Path(dir_path)
    if not root.exists():
        return json.dumps({"error": f"Directory not found: {dir_path}"})

    all_files = [p for p in root.rglob("*") if p.is_file()]
    file_index = _categorise_files(all_files)

    # Read text from code + docs
    code_text = _read_text_files(file_index.get("code", []))
    doc_text  = _read_text_files(file_index.get("docs", []))
    csv_info  = _inspect_csvs(file_index.get("data", []))
    nb_info   = _inspect_notebooks(file_index.get("notebooks", []))

    all_text  = code_text + "\n" + doc_text + "\n" + nb_info

    result = {
        "source_directory": str(root),
        "total_files": len(all_files),
        "file_breakdown": {k: len(v) for k, v in file_index.items()},
        "files": {
            k: [str(p.relative_to(root)) for p in v]
            for k, v in file_index.items()
        },
        "task_type":         _infer_task_type(all_text),
        "problem_statement": _extract_problem_statement(doc_text + nb_info),
        "libraries_used":    _detect_libraries(code_text + nb_info),
        "model_names":       _detect_models(all_text),
        "metrics_found":     _detect_metrics(all_text),
        "baselines":         _extract_baseline_scores(all_text),
        "datasets_mentioned":_detect_datasets(all_text),
        "csv_summary":       csv_info,
        "suggested_approach": "",  # filled below
    }

    result["suggested_approach"] = _suggest_approach(result)
    return json.dumps(result, indent=2)


def summarize_code_file(file_path: str) -> str:
    """
    Read a single Python / Jupyter file and return a structured summary
    with imports, class/function names, and key logic snippets.
    """
    path = Path(file_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {file_path}"})

    if path.suffix == ".ipynb":
        text = _read_notebook_as_text(path)
    else:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return json.dumps({"error": str(e)})

    imports   = _extract_imports(text)
    functions = re.findall(r"^def\s+(\w+)", text, re.MULTILINE)
    classes   = re.findall(r"^class\s+(\w+)", text, re.MULTILINE)
    metrics   = _detect_metrics(text)
    models    = _detect_models(text)

    return json.dumps({
        "file":      str(path.name),
        "lines":     len(text.splitlines()),
        "imports":   imports[:30],
        "functions": functions[:20],
        "classes":   classes[:10],
        "metrics":   metrics,
        "models":    models,
        "snippet":   text[:1500],
    }, indent=2)


# ════════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _categorise_files(files):
    categories = {
        "code":      [],
        "notebooks": [],
        "data":      [],
        "images":    [],
        "docs":      [],
        "models":    [],
        "other":     [],
    }
    ext_map = {
        ".py":    "code",
        ".ipynb": "notebooks",
        ".csv":   "data",
        ".tsv":   "data",
        ".json":  "data",
        ".parquet": "data",
        ".xlsx":  "data",
        ".jpg":   "images",
        ".jpeg":  "images",
        ".png":   "images",
        ".gif":   "images",
        ".pdf":   "docs",
        ".md":    "docs",
        ".txt":   "docs",
        ".rst":   "docs",
        ".docx":  "docs",
        ".pth":   "models",
        ".pt":    "models",
        ".h5":    "models",
        ".pkl":   "models",
        ".joblib":"models",
    }
    for f in files:
        cat = ext_map.get(f.suffix.lower(), "other")
        categories[cat].append(f)
    return categories


def _read_text_files(paths, max_chars_each=4000, max_total=30000):
    parts = []
    total = 0
    for p in paths:
        if total >= max_total:
            break
        try:
            if p.suffix == ".ipynb":
                text = _read_notebook_as_text(p)
            else:
                text = p.read_text(encoding="utf-8", errors="ignore")
            chunk = text[:max_chars_each]
            parts.append(f"===FILE: {p.name}===\n{chunk}")
            total += len(chunk)
        except Exception:
            pass
    return "\n".join(parts)


def _read_notebook_as_text(path):
    try:
        import nbformat
        nb = nbformat.read(str(path), as_version=4)
        lines = []
        for cell in nb.cells:
            lines.append(f"# [{cell.cell_type.upper()}]\n{cell.source}")
        return "\n\n".join(lines)
    except Exception:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        # strip JSON noise
        text = re.sub(r'"(cell_type|metadata|outputs|source)":\s*', "", raw)
        return text[:8000]


def _inspect_csvs(csv_paths, preview_rows=3):
    summaries = []
    for p in csv_paths[:5]:
        try:
            import pandas as pd
            df = pd.read_csv(p, nrows=preview_rows)
            summaries.append({
                "file":    p.name,
                "columns": list(df.columns),
                "dtypes":  {c: str(df[c].dtype) for c in df.columns},
                "preview": df.head(preview_rows).to_dict(orient="records"),
            })
        except Exception as e:
            summaries.append({"file": p.name, "error": str(e)})
    return summaries


def _inspect_notebooks(nb_paths):
    texts = []
    for p in nb_paths[:5]:
        texts.append(_read_notebook_as_text(p))
    return "\n\n".join(texts)


def _extract_imports(text):
    imports = re.findall(r"^(?:import|from)\s+(\S+)", text, re.MULTILINE)
    return sorted(set(i.split(".")[0] for i in imports))


def _detect_libraries(text):
    lib_keywords = [
        "torch", "tensorflow", "keras", "sklearn", "lightgbm", "xgboost",
        "catboost", "transformers", "huggingface", "diffusers", "timm",
        "mmdet", "detectron2", "ultralytics", "yolo", "efficientnet",
        "segmentation_models", "albumentations", "cv2", "scipy",
        "fastai", "optuna", "wandb", "mlflow",
    ]
    t = text.lower()
    return [lib for lib in lib_keywords if lib in t]


def _detect_models(text):
    model_patterns = [
        r"EfficientNet\w*",
        r"ConvNeXt\w*",
        r"ResNet\w*",
        r"ViT\w*",
        r"Swin\w*",
        r"BERT\w*",
        r"GPT\w*",
        r"LLaMA\w*",
        r"DeBERTa\w*",
        r"YOLO\w*",
        r"RegNet\w*",
        r"DenseNet\w*",
        r"MobileNet\w*",
        r"InceptionV?\w*",
        r"VGG\w*",
        r"AlexNet\w*",
        r"UNet\w*",
        r"FPN\w*",
        r"DETR\w*",
        r"SAM\w*",
        r"CLIP\w*",
        r"LoRA\w*",
    ]
    found = set()
    for pat in model_patterns:
        found.update(re.findall(pat, text, re.IGNORECASE))
    return list(found)


def _detect_metrics(text):
    metric_keywords = ["accuracy", "f1", "auc", "roc", "precision", "recall",
                        "mae", "mse", "rmse", "map", "mAP", "iou", "dice",
                        "bleu", "rouge", "perplexity", "loss", "val_loss"]
    # also look for numeric scores: "accuracy: 0.94" or "F1 = 0.87"
    patterns = [
        r"(?:" + "|".join(metric_keywords) + r")\s*[=:]\s*([\d.]+)",
        r"([\d.]+)\s*(?:accuracy|f1|f1-score|auc|mAP)",
    ]
    scores = []
    for pat in patterns:
        scores.extend(re.findall(pat, text, re.IGNORECASE))
    return list(set(scores))[:15]


def _extract_baseline_scores(text):
    """Try to pull baseline/SOTA numbers from text."""
    lines = text.splitlines()
    baseline_lines = []
    for line in lines:
        if any(k in line.lower() for k in ["baseline", "benchmark", "sota", "our method",
                                             "accuracy", "f1", "result", "score", "performance"]):
            clean = line.strip()
            if clean and len(clean) < 200:
                baseline_lines.append(clean)
    return baseline_lines[:20]


def _detect_datasets(text):
    dataset_keywords = [
        "imagenet", "cifar", "coco", "voc", "ade20k", "cityscapes",
        "flickr", "openimages", "places365", "sun397",
        "glue", "squad", "mnli", "sst", "common crawl",
        "kaggle", "huggingface", "roboflow",
    ]
    t = text.lower()
    found = [kw for kw in dataset_keywords if kw in t]
    # also detect CSV / folder names that look like datasets
    csv_refs = re.findall(r"[\w-]+\.csv", text)
    return list(set(found + csv_refs))[:15]


def _infer_task_type(text: str) -> str:
    text_l = text.lower()
    rules = [
        (["image classif", "scene classif", "fine-grained", "recognition"],           "image_classification"),
        (["object detect", "bounding box", "yolo", "frcnn", "anchor"],                "object_detection"),
        (["semantic segment", "instance segment", "panoptic", "unet"],                "image_segmentation"),
        (["text classif", "sentiment", "nlp", "bert", "gpt", "llm", "transformers"],  "nlp_classification"),
        (["tabular", "structured data", "lightgbm", "xgboost", "catboost"],           "tabular"),
        (["depth estimation", "monocular depth"],                                      "depth_estimation"),
        (["image generation", "diffusion", "gan", "vae", "stable diffusion"],         "generative"),
        (["regression", "predict", "forecast"],                                        "regression"),
    ]
    for keywords, task_type in rules:
        if any(kw in text_l for kw in keywords):
            return task_type
    if re.search(r"\.(jpg|jpeg|png|gif)", text_l):
        return "image_classification"
    return "general"


def _extract_problem_statement(text: str, max_chars: int = 1500) -> str:
    """Pull the most descriptive paragraph found in doc/notebook text."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > 80]
    # prefer paragraphs that mention problem/task/goal
    scored = []
    for p in paragraphs[:40]:
        score = sum(1 for kw in ["problem", "task", "goal", "objective", "classification",
                                   "detect", "segment", "dataset", "competition", "baseline",
                                   "benchmark", "accuracy", "f1", "metric"] if kw in p.lower())
        scored.append((score, p))
    scored.sort(reverse=True)
    if scored:
        return scored[0][1][:max_chars]
    return text[:max_chars] if text else ""


def _suggest_approach(result: dict) -> str:
    task = result.get("task_type", "general")
    libs = result.get("libraries_used", [])
    models = result.get("model_names", [])

    suggestions = {
        "image_classification": (
            "Use EfficientNetV2-M or ConvNeXt-Base as backbone. "
            "Apply heavy augmentation (RandAugment, MixUp, CutMix). "
            "Train with AdamW + OneCycleLR + label smoothing. "
            "Use TTA at inference. Consider ViT or Swin for large datasets."
        ),
        "object_detection": (
            "Use YOLOv8-L or Faster R-CNN with a strong backbone (ConvNeXt-S). "
            "Apply mosaic + copy-paste augmentation. "
            "Use mixed-precision training. Consider DETR for small datasets."
        ),
        "nlp_classification": (
            "Fine-tune DeBERTa-v3-base or distilBERT. "
            "Use AdamW with warmup + weight decay. "
            "Consider data augmentation (back-translation, synonym replacement). "
            "Ensemble multiple checkpoints."
        ),
        "tabular": (
            "Use LightGBM + CatBoost + XGBoost stacked ensemble. "
            "Tune with Optuna (100+ trials). Feature engineering: interactions, statistics. "
            "Use 5-fold CV with proper OOF blending."
        ),
    }
    return suggestions.get(task, "Analyse the data carefully, establish strong baselines, then iterate.")
