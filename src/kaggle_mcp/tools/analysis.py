"""
Analysis and interpretation tools.

Tools:
  design_ablation_study          — comprehensive ablation experiment plan
  interpret_training_log         — diagnose training dynamics from log text
  estimate_kaggle_feasibility    — will this fit Kaggle GPU limits?
  suggest_ensemble_strategy      — how to combine multiple models
  identify_hard_samples          — what edge cases to watch for
  generate_hypothesis_test_plan  — statistical significance testing plan
  hyperparameter_sensitivity     — which knobs matter most for this task
  compute_experiment_matrix      — full experiment grid with priority ranking
"""
from __future__ import annotations

import json
import re
from typing import Optional


# ════════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASES
# ════════════════════════════════════════════════════════════════════════════════

# GPU memory for common architectures at batch_size=16, fp16, 224px
_GPU_MEM_GB: dict[str, float] = {
    "resnet50": 3.2, "resnet101": 5.1, "resnet152": 7.0,
    "efficientnet_b0": 2.1, "efficientnet_b4": 4.8, "efficientnet_v2_s": 4.2,
    "efficientnet_v2_m": 7.1, "efficientnet_v2_l": 11.5,
    "convnext_tiny": 3.5, "convnext_base": 8.0, "convnext_large": 14.0,
    "swin_t": 4.0, "swin_s": 6.5, "swin_b": 9.0,
    "vit_b_16": 6.8, "vit_l_16": 14.5,
    "yolov8n": 2.5, "yolov8s": 3.8, "yolov8m": 6.2, "yolov8l": 10.0,
    "deberta_base": 5.0, "deberta_large": 9.0, "deberta_v3_large": 11.0,
}

# Kaggle GPU environments (2024)
_KAGGLE_GPU_ENVS: dict[str, dict] = {
    "P100":  {"vram_gb": 16,  "tflops": 9.3,   "hours_per_week": 30},
    "T4x2":  {"vram_gb": 30,  "tflops": 16.4,  "hours_per_week": 30},
    "P100x2": {"vram_gb": 32, "tflops": 18.6,  "hours_per_week": 30},
    "TPU":   {"vram_gb": 64,  "tflops": 180,   "hours_per_week": 20},
}

# Epoch time estimates (minutes) per 10K images, batch=32, fp16
_EPOCH_MINS_PER_10K: dict[str, float] = {
    "resnet50": 1.2, "efficientnet_v2_s": 1.8, "convnext_base": 2.5,
    "swin_b": 3.2, "vit_b_16": 2.8, "vit_l_16": 7.5,
    "yolov8m": 4.0, "deberta_large": 8.0,
}


# ════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _detect_arch(text: str) -> str:
    text_l = text.lower().replace("-", "_").replace(" ", "_")
    for arch in _GPU_MEM_GB:
        if arch.replace("_", "") in text_l.replace("_", ""):
            return arch
    return "custom"


def _parse_log_metrics(log_text: str) -> list[dict]:
    """Extract (epoch, train_loss, val_loss, val_metric) tuples from log text."""
    records = []
    # Support many common log formats
    for line in log_text.splitlines():
        line_l = line.lower()
        epoch_m  = re.search(r"ep(?:och)?\s*[:\|]?\s*(\d+)", line_l)
        tloss_m  = re.search(r"tr(?:ain)?[\s_]loss[:\s=]+([0-9.]+)", line_l)
        vloss_m  = re.search(r"v(?:al|alid)?[\s_]loss[:\s=]+([0-9.]+)", line_l)
        vmetric_m= re.search(r"(?:f1|acc|auc|map|iou|score)[:\s=]+([0-9.]+)", line_l)
        if epoch_m:
            records.append({
                "epoch":      int(epoch_m.group(1)),
                "train_loss": float(tloss_m.group(1)) if tloss_m else None,
                "val_loss":   float(vloss_m.group(1)) if vloss_m else None,
                "val_metric": float(vmetric_m.group(1)) if vmetric_m else None,
            })
    return records


# ════════════════════════════════════════════════════════════════════════════════
# PUBLIC TOOLS
# ════════════════════════════════════════════════════════════════════════════════

def design_ablation_study(
    arch_description:  str,
    task_type:         str = "image_classification",
    available_compute: str = "P100",
) -> str:
    """
    Designs a comprehensive, prioritised ablation study.

    For each proposed component: what to remove/swap, what metric to track,
    expected impact range, compute cost, and whether it's essential vs nice-to-have.

    Returns: { components, ablation_table, priority_order, minimal_ablation,
               full_ablation, expected_table_in_paper }
    """
    text_l  = arch_description.lower()
    arch    = _detect_arch(arch_description)
    gpu_env = _KAGGLE_GPU_ENVS.get(available_compute, _KAGGLE_GPU_ENVS["P100"])

    # ── Generate ablation components ─────────────────────────────────────────
    components = []

    # Backbone ablation
    components.append({
        "component":      "Backbone",
        "baseline":       arch,
        "ablation_swap":  "Replace with ResNet50 (standard baseline)",
        "metric_change":  "Expected F1 drop: 5-15%",
        "priority":       "HIGH — validates backbone choice",
        "compute_cost":   "1 run",
        "essential":      True,
    })

    # Augmentation ablation
    if any(k in text_l for k in ["mixup", "cutmix", "augment"]):
        components.append({
            "component":     "Augmentation",
            "baseline":      "Full augmentation pipeline",
            "ablation_swap": "w/o MixUp; w/o CutMix; w/o RandAugment (3 separate runs)",
            "metric_change": "Expected F1 drop per aug: 1-5%",
            "priority":      "HIGH — each component must earn its complexity",
            "compute_cost":  "3 runs",
            "essential":     True,
        })

    # Loss function ablation
    if any(k in text_l for k in ["label_smooth", "focal", "arcface", "supcon"]):
        components.append({
            "component":     "Loss Function",
            "baseline":      "Current loss",
            "ablation_swap": "CrossEntropyLoss (vanilla baseline)",
            "metric_change": "Expected F1 change: ±2-8%",
            "priority":      "HIGH — novel loss needs justification",
            "compute_cost":  "1 run",
            "essential":     True,
        })

    # Scheduler ablation
    if any(k in text_l for k in ["cosine", "onecycle", "warmup"]):
        components.append({
            "component":     "LR Scheduler",
            "baseline":      "Current scheduler",
            "ablation_swap": "StepLR (standard) vs Cosine vs OneCycleLR",
            "metric_change": "Expected F1 change: ±1-4%",
            "priority":      "MEDIUM",
            "compute_cost":  "2 runs",
            "essential":     False,
        })

    # Pretraining ablation
    if any(k in text_l for k in ["imagenet", "pretrain", "finetune"]):
        components.append({
            "component":     "Pre-training",
            "baseline":      "ImageNet-21K or 1K weights",
            "ablation_swap": "Random init (train from scratch)",
            "metric_change": "Expected F1 drop: 10-30%",
            "priority":      "HIGH — quantifies value of transfer learning",
            "compute_cost":  "1 run (may need more epochs)",
            "essential":     True,
        })

    # Cross-validation ablation
    if any(k in text_l for k in ["kfold", "cross_valid", "5-fold"]):
        components.append({
            "component":     "Cross-validation",
            "baseline":      "5-fold stratified CV",
            "ablation_swap": "Single 80/20 split",
            "metric_change": "Variance estimate: ±3-8% std dev on single fold",
            "priority":      "MEDIUM — justifies compute overhead",
            "compute_cost":  "Compare 1 run vs 5 runs",
            "essential":     False,
        })

    # Image resolution ablation (for vision tasks)
    if task_type in ("image_classification", "object_detection"):
        size_m = re.search(r"(\d{3})(?:\s*px|\s*x\s*\d{3}|\s*image)", text_l)
        if size_m:
            size = int(size_m.group(1))
            components.append({
                "component":     "Input Resolution",
                "baseline":      f"{size}px",
                "ablation_swap": f"{size//2}px; {size*3//2}px",
                "metric_change": f"Larger: +1-5% F1, 2× compute. Smaller: -2-8% F1, 0.5× compute.",
                "priority":      "MEDIUM",
                "compute_cost":  "2 runs",
                "essential":     False,
            })

    # Always add: full model vs just proposed component
    essential   = [c for c in components if c["essential"]]
    non_essential = [c for c in components if not c["essential"]]

    total_runs_full    = sum(int(re.search(r"\d+", c["compute_cost"]).group()) for c in components)
    total_runs_minimal = sum(int(re.search(r"\d+", c["compute_cost"]).group()) for c in essential)

    # Paper table format
    table_header = "| Component | Base | Ablation | Expected ΔF1 | Priority |"
    table_rows   = [
        f"| {c['component']} | {c['baseline'][:25]} | {c['ablation_swap'][:30]} | {c['metric_change'][:25]} | {c['priority']} |"
        for c in components
    ]

    return json.dumps({
        "architecture":             arch,
        "task_type":                task_type,
        "total_components":         len(components),
        "ablation_components":      components,
        "priority_order":           [c["component"] for c in sorted(
                                        components, key=lambda x: x["priority"].startswith("HIGH"), reverse=True)],
        "minimal_ablation_runs":    total_runs_minimal,
        "full_ablation_runs":       total_runs_full,
        "minimal_ablation_set":     [c["component"] for c in essential],
        "paper_table_markdown":     "\n".join([table_header, "|---|---|---|---|---|"] + table_rows),
        "advice": (
            "Run essential ablations first. If they all support your design choices, "
            "the paper is defensible. Then add non-essential for thoroughness. "
            f"Total estimated runs on {available_compute}: {total_runs_full} × (1 base run)."
        ),
    }, indent=2)


def interpret_training_log(
    log_text:   str,
    metric_name: str = "f1",
) -> str:
    """
    Diagnoses training dynamics from a training log.

    Identifies: overfitting, underfitting, LR issues, instability, convergence,
                plateau, and provides concrete recommendations.

    Returns: { diagnosis, issues_detected, recommendations, next_experiments }
    """
    records = _parse_log_metrics(log_text)

    if not records:
        return json.dumps({
            "error": "Could not parse training log. Expected format: 'Epoch N | train_loss X | val_loss Y | f1 Z'",
            "supported_formats": [
                "Epoch 5 | train_loss=0.42 | val_loss=0.60 | f1=0.72",
                "Ep 5/30 | TR loss=0.42 acc=0.85 | VA loss=0.60 f1=0.72",
                "epoch=5, loss=0.42, val_loss=0.60, val_f1=0.72",
            ],
        }, indent=2)

    train_losses  = [r["train_loss"]  for r in records if r.get("train_loss")  is not None]
    val_losses    = [r["val_loss"]    for r in records if r.get("val_loss")    is not None]
    val_metrics   = [r["val_metric"]  for r in records if r.get("val_metric")  is not None]

    issues        = []
    recommendations = []
    diagnosis     = "HEALTHY"

    # ── Overfitting ───────────────────────────────────────────────────────────
    if train_losses and val_losses and len(train_losses) >= 3:
        t_last = train_losses[-1]
        v_last = val_losses[-1]
        t_first = train_losses[0]

        gap = v_last - t_last
        if gap > 0.3:
            diagnosis = "OVERFITTING"
            issues.append(
                f"SEVERE OVERFITTING: val_loss={v_last:.3f} >> train_loss={t_last:.3f} "
                f"(gap={gap:.3f}). Model memorising training data."
            )
            recommendations += [
                "Add/increase dropout (try 0.3 → 0.5)",
                "Increase weight_decay (try 1e-4 → 5e-4)",
                "Add label smoothing (0.1 → 0.2)",
                "Reduce model capacity (smaller backbone)",
                "Add MixUp/CutMix augmentation",
                "Collect more training data or use pseudo-labelling",
            ]
        elif gap > 0.1:
            if diagnosis == "HEALTHY":
                diagnosis = "MILD OVERFITTING"
            issues.append(f"Mild overfitting (gap={gap:.3f}). Monitor closely.")
            recommendations.append("Add light regularisation: weight_decay=1e-4 + label_smoothing=0.05")

    # ── Underfitting / high training loss ────────────────────────────────────
    if train_losses and train_losses[-1] > 1.0:
        if diagnosis == "HEALTHY":
            diagnosis = "UNDERFITTING"
        issues.append(f"High final train_loss={train_losses[-1]:.3f}. Model may be underfitting.")
        recommendations += [
            "Increase model capacity (larger backbone)",
            "Reduce regularisation if any",
            "Check learning rate — may be too high or too low",
            "Verify data loading/normalisation is correct",
            "Train for more epochs",
        ]

    # ── Plateau detection ─────────────────────────────────────────────────────
    if val_metrics and len(val_metrics) >= 5:
        recent_metrics = val_metrics[-5:]
        metric_range   = max(recent_metrics) - min(recent_metrics)
        if metric_range < 0.005:
            if diagnosis == "HEALTHY":
                diagnosis = "PLATEAU"
            issues.append(
                f"PLATEAU: {metric_name} stuck at ~{val_metrics[-1]:.4f} for last 5 epochs."
            )
            recommendations += [
                "Try learning rate warmup or cyclical LR",
                "Lower LR by 10× (ReduceLROnPlateau triggered)",
                "Progressive unfreezing — unfreeze deeper layers",
                "Try a different augmentation strategy",
            ]

    # ── Instability / spiky loss ───────────────────────────────────────────────
    if len(train_losses) >= 4:
        diffs = [abs(train_losses[i] - train_losses[i-1]) for i in range(1, len(train_losses))]
        mean_diff = sum(diffs) / len(diffs)
        if mean_diff > 0.1:
            issues.append(
                f"UNSTABLE TRAINING: mean step-to-step loss change={mean_diff:.3f}. "
                "Training is noisy."
            )
            recommendations += [
                "Add gradient clipping (max_norm=1.0)",
                "Reduce learning rate by 2-5×",
                "Increase batch size (or use gradient accumulation)",
                "Verify no NaN in data (check image loading pipeline)",
            ]

    # ── Best epoch analysis ───────────────────────────────────────────────────
    best_epoch = None
    if val_metrics:
        best_val   = max(val_metrics)
        best_idx   = val_metrics.index(best_val)
        best_record = records[best_idx] if best_idx < len(records) else {}
        best_epoch  = best_record.get("epoch", best_idx + 1)

        total_epochs = records[-1].get("epoch", len(records))
        if best_epoch < total_epochs * 0.5:
            issues.append(
                f"EARLY CONVERGENCE: Best {metric_name}={best_val:.4f} at epoch {best_epoch}, "
                f"but trained until epoch {total_epochs}. {total_epochs - best_epoch} epochs wasted."
            )
            recommendations.append(
                f"Add early stopping with patience=10. "
                f"Or reduce total epochs to ~{best_epoch + 5}."
            )

    return json.dumps({
        "diagnosis":           diagnosis,
        "epochs_parsed":       len(records),
        "final_train_loss":    train_losses[-1] if train_losses else None,
        "final_val_loss":      val_losses[-1]   if val_losses else None,
        "best_val_metric":     max(val_metrics) if val_metrics else None,
        "best_metric_epoch":   best_epoch,
        "issues_detected":     issues,
        "recommendations":     recommendations,
        "training_summary": {
            "train_loss_trend":  "improving" if len(train_losses) > 1 and train_losses[-1] < train_losses[0] else "stagnant",
            "val_metric_trend":  "improving" if len(val_metrics) > 1  and val_metrics[-1] > val_metrics[0]  else "stagnant",
            "convergence_status": diagnosis,
        },
    }, indent=2)


def estimate_kaggle_feasibility(
    arch_description:  str,
    dataset_info:      str = "",
    num_epochs:        int = 30,
    batch_size:        int = 32,
    image_size:        int = 224,
    use_accumulation:  bool = False,
) -> str:
    """
    Estimates whether training will fit within Kaggle's GPU quota and time limits.

    Kaggle limits (2024):
      • GPU T4 x2: 30h/week, ~30GB VRAM total, 9h per session
      • GPU P100:  30h/week, 16GB VRAM, 9h per session
      • TPU v3-8:  20h/week, 9h per session

    Returns: { will_fit_on_oom_check, estimated_hours, recommended_env,
               memory_breakdown, optimisations, risk_level }
    """
    arch  = _detect_arch(arch_description)
    text_l = arch_description.lower()

    # Base memory for architecture at fp32, BS=16, 224px
    base_mem_gb = _GPU_MEM_GB.get(arch, 8.0)

    # Scale by image size
    size_scale   = (image_size / 224) ** 2
    # Scale by batch size
    bs_scale     = batch_size / 16
    # fp16 halves memory
    fp16         = any(k in text_l for k in ["amp", "fp16", "mixed_precision", "autocast"])
    fp_scale     = 0.55 if fp16 else 1.0

    required_mem = base_mem_gb * size_scale * bs_scale * fp_scale
    # Add optimizer states (Adam ~ 2× params in memory), activations buffer
    total_mem    = required_mem * 2.5

    # Estimate training time
    n_samples_match  = re.search(r"(\d[\d,_]*)\s*(image|sample|row)", (dataset_info or "").lower())
    n_samples = int(n_samples_match.group(1).replace(",", "").replace("_", "")) if n_samples_match else 10000

    arch_key       = arch.split("_")[0]  # convnext from convnext_base
    mins_per_10k   = max(val for k, val in _EPOCH_MINS_PER_10K.items() if arch_key in k) \
                     if any(arch_key in k for k in _EPOCH_MINS_PER_10K) else 2.5
    mins_per_epoch = mins_per_10k * (n_samples / 10000) * size_scale * (batch_size / 32)
    total_hrs      = (mins_per_epoch * num_epochs) / 60

    # Find suitable Kaggle env
    recommendation = "P100"
    for env_name, env in _KAGGLE_GPU_ENVS.items():
        if env["vram_gb"] >= total_mem * 1.2:  # 20% safety margin
            recommendation = env_name
            break

    chosen_env = _KAGGLE_GPU_ENVS.get(recommendation, _KAGGLE_GPU_ENVS["P100"])
    oom_risk   = total_mem > chosen_env["vram_gb"]
    time_risk  = total_hrs > 9  # 9h Kaggle session limit

    optimisations = []
    if oom_risk or total_mem > 12:
        optimisations.append("Enable AMP (FP16) — halves memory, 30-40% faster")
        optimisations.append("Use gradient checkpointing — trades compute for memory")
    if total_mem > 20:
        optimisations.append("Reduce batch size and use gradient accumulation (4× to simulate bs=128)")
        optimisations.append("Consider smaller backbone — ViT-L/16 → ViT-B/16 saves 10GB")
    if time_risk:
        optimisations.append(f"Training {total_hrs:.1f}h exceeds 9h Kaggle session limit")
        optimisations.append("Reduce epochs or use faster backbone (EfficientNetV2-S vs L)")
        optimisations.append("Save checkpoint every 5 epochs to resume across sessions")

    return json.dumps({
        "architecture":          arch,
        "estimated_memory_gb":   round(total_mem, 1),
        "estimated_hours":       round(total_hrs, 1),
        "recommended_env":       recommendation,
        "env_vram_gb":           chosen_env["vram_gb"],
        "oom_risk":              oom_risk,
        "time_limit_risk":       time_risk,
        "risk_level":            (
            "HIGH — likely OOM or timeout" if oom_risk or time_risk else
            "MEDIUM — close to limits, monitor" if total_mem > chosen_env["vram_gb"] * 0.7 else
            "LOW — comfortable"
        ),
        "memory_breakdown": {
            "model_params_gb":      round(base_mem_gb,     1),
            "activations_buffer_gb": round(required_mem * 0.8, 1),
            "optimiser_states_gb":  round(required_mem * 0.7, 1),
            "total_estimated_gb":   round(total_mem, 1),
        },
        "optimisations":         optimisations,
        "fp16_enabled":          fp16,
        "note": (
            "These are estimates based on architecture heuristics. "
            "Actual memory varies ±30% based on implementation details. "
            "Always test with a small batch first before full run."
        ),
    }, indent=2)


def suggest_ensemble_strategy(
    task_type:    str,
    model_descriptions: list,
    num_folds:    int = 5,
    metric:       str = "f1",
) -> str:
    """
    Recommends how to combine multiple models for maximum performance gain.

    Args:
        task_type: 'image_classification', 'object_detection', 'nlp_classification', 'tabular'
        model_descriptions: list of model description strings
        num_folds: number of CV folds used

    Returns: { strategy, ensemble_weights, expected_gain, implementation_code_hint }
    """
    n_models = len(model_descriptions)
    archs    = [_detect_arch(m) for m in (model_descriptions or ["base_model"])]

    strategies: dict[str, dict] = {
        "image_classification": {
            "primary":   "Average softmax probabilities (simple average, equal weights)",
            "advanced":  "Weighted average: val F1 as weights, re-normalised",
            "best":      "Stacking: train a LightGBM or LogReg on OOF predictions from all folds",
            "diversity_tip": "Maximum diversity = different backbone families: ConvNet + ViT + old-to-new",
        },
        "nlp_classification": {
            "primary":   "Average logits or softmax probabilities",
            "advanced":  "Weighted ensemble by validation metric per model",
            "best":      "OOF stacking + include TF-IDF features",
            "diversity_tip": "Use different model families: DeBERTa + BERT + DistilBERT",
        },
        "object_detection": {
            "primary":   "Weighted Boxes Fusion (WBF) — better than NMS for ensembles",
            "advanced":  "NMS ensemble with IoU=0.5, score_thr=0.001",
            "best":      "WBF with per-model confidence weights from validation mAP",
            "diversity_tip": "Mix anchor-based (YOLOv8) + anchor-free (DETR) + multi-scale",
        },
        "tabular": {
            "primary":   "Average predictions from LGBM + CatBoost + XGBoost",
            "advanced":  "Rank-based blending: average rank percentiles, not raw scores",
            "best":      "3-level stacking: models → meta-features → LightGBM meta-learner",
            "diversity_tip": "GBM + Neural Net + Linear model for maximum diversity",
        },
    }

    strat = strategies.get(task_type, strategies["image_classification"])

    # Compute diversity score (1 = perfectly diverse, 0 = all same)
    unique_families = len(set(
        arch.split("_")[0] if arch != "custom" else f"custom_{i}"
        for i, arch in enumerate(archs)
    ))
    diversity_score = min(unique_families / max(n_models, 1), 1.0)

    expected_gain = (
        f"+4-8% {metric}" if diversity_score >= 0.8 else
        f"+2-5% {metric}" if diversity_score >= 0.5 else
        f"+1-3% {metric} (low diversity — models too similar)"
    )

    impl_hints = {
        "image_classification": (
            "# Simple average\n"
            "preds = np.mean([m.predict_proba(X) for m in models], axis=0)\n"
            "# Weighted average by validation score\n"
            "weights = np.array(val_scores) / sum(val_scores)\n"
            "preds = sum(w * m.predict_proba(X) for w, m in zip(weights, models))"
        ),
        "object_detection": (
            "from ensemble_boxes import weighted_boxes_fusion\n"
            "boxes_list, scores_list, labels_list = zip(*[model(img) for model in models])\n"
            "boxes, scores, labels = weighted_boxes_fusion(\n"
            "    list(boxes_list), list(scores_list), list(labels_list),\n"
            "    iou_thr=0.5, skip_box_thr=0.001\n)"
        ),
        "tabular": (
            "import lightgbm as lgb\n"
            "oof_preds = np.column_stack([model.oof_predictions for model in base_models])\n"
            "meta_model = lgb.LGBMClassifier(n_estimators=200)\n"
            "meta_model.fit(oof_preds, y_train)\n"
        ),
    }

    return json.dumps({
        "task_type":          task_type,
        "models_given":       archs,
        "diversity_score":    round(diversity_score, 2),
        "strategy": {
            "simple":    strat["primary"],
            "advanced":  strat["advanced"],
            "best":      strat["best"],
            "diversity":  strat["diversity_tip"],
        },
        "expected_gain":      expected_gain,
        "num_folds":          num_folds,
        "total_oof_runs":     n_models * num_folds,
        "implementation_hint": impl_hints.get(task_type, impl_hints["image_classification"]),
        "warning": (
            "Low model diversity will not ensemble well — correlation between predictions "
            "is too high. Add a fundamentally different architecture for meaningful gain."
        ) if diversity_score < 0.5 else None,
    }, indent=2)


def identify_hard_samples(
    task_type:   str,
    class_names: list = None,
    dataset_info: str = "",
) -> str:
    """
    Returns known hard cases and edge cases for a task type.
    Guides manual inspection and targeted augmentation.

    Returns: { hard_categories, confusion_matrix_patterns, augmentation_prescriptions,
               manual_inspection_checklist }
    """
    class_names = class_names or []
    text_l      = dataset_info.lower()

    hard_cases: dict[str, list] = {
        "image_classification": [
            "Near-duplicate images across classes (inter-class similarity)",
            "Partial occlusion — object only 20-40% visible",
            "Unusual viewpoint / extreme zoom in or out",
            "Low-light or over-exposed images",
            "Images with multiple objects from different classes",
            "Rare texture/colour variants within a class",
            "Small objects (< 32×32 px effective region)",
            "Abstract or artistic representations of the class",
        ],
        "object_detection": [
            "Heavily overlapping bounding boxes (IoU > 0.7)",
            "Objects at image border (cropped by camera frame)",
            "Very small objects (< 16×16 px in original image)",
            "Dense crowds of the same class",
            "Objects in unusual orientations (upside down, side view)",
            "Reflections or shadows mistaken for objects",
        ],
        "nlp_classification": [
            "Sarcasm and irony (sentiment opposite of literal meaning)",
            "Very short texts (< 10 tokens) — insufficient context",
            "Code switching (multiple languages in one sample)",
            "Domain-specific jargon not in pretraining corpora",
            "Negation patterns ('not bad' = positive)",
            "Ambiguous sentences that require world knowledge",
        ],
        "tabular": [
            "Missing values in test set not seen during training",
            "Feature values outside training distribution (test time)",
            "High-cardinality category values seen only in test",
            "Temporal leakage near time boundaries",
            "Outlier rows with extreme feature values",
        ],
    }

    # Class-specific confusion patterns based on class names
    confusion_hints = []
    if class_names:
        similar_classes = [
            ("cat", "dog"), ("car", "truck"), ("bicycle", "motorcycle"),
            ("shirt", "t-shirt"), ("sofa", "chair"), ("lamp", "vase"),
        ]
        for c1, c2 in similar_classes:
            if c1 in [n.lower() for n in class_names] and c2 in [n.lower() for n in class_names]:
                confusion_hints.append(
                    f"HIGH CONFUSION EXPECTED: '{c1}' and '{c2}' are visually similar. "
                    f"Use class-aware augmentation and higher resolution for these classes."
                )

    targeted_augs = {
        "image_classification": [
            "GridDistortion for partial occlusion training",
            "RandomBrightness + RandomContrast for lighting variation",
            "CoarseDropout (Cutout) for occlusion robustness",
            "Perspective transform for viewpoint variation",
        ],
        "nlp_classification": [
            "Back-translation for linguistic variation",
            "Token insertion/deletion for short-text cases",
            "Synonym replacement with WordNet",
        ],
        "object_detection": [
            "Mosaic augmentation for multi-object scenes",
            "Copy-Paste for small object frequency increase",
            "Multi-scale training for scale invariance",
        ],
        "tabular": [
            "Gaussian noise injection on continuous features",
            "Feature masking (simulate missing values)",
            "SMOTE or class weight adjustment for imbalance",
        ],
    }

    return json.dumps({
        "task_type":             task_type,
        "classes":               class_names,
        "known_hard_cases":      hard_cases.get(task_type, []),
        "confusion_matrix_patterns": confusion_hints or ["Inspect confusion matrix after training to identify specific pairs."],
        "targeted_augmentations": targeted_augs.get(task_type, []),
        "manual_inspection_checklist": [
            "Look at the 50 most-confident WRONG predictions after training",
            "Check if hard cases were part of training or only in validation",
            "Compute per-class F1 — imbalanced classes show lowest F1",
            "Visualise GradCAM/attention maps for misclassified samples",
            "Check image quality stats for wrong predictions (blur, brightness)",
        ],
    }, indent=2)


def generate_hypothesis_test_plan(
    metric:          str = "f1",
    baseline_score:  float = 0.80,
    proposed_score:  float = 0.83,
    n_test_samples:  int = 1000,
    n_experiments:   int = 5,
) -> str:
    """
    Designs a statistical significance testing plan.
    Answers: 'Is my improvement real or just noise?'

    Returns: { test_type, effect_size, recommended_n, power_analysis,
               implementation, interpretation_guide }
    """
    import math

    delta = proposed_score - baseline_score

    # Cohen's h for proportions (approximation for F1/accuracy)
    def cohen_h(p1, p2):
        return 2 * (math.asin(math.sqrt(max(0, min(1, p2)))) -
                    math.asin(math.sqrt(max(0, min(1, p1)))))

    effect = cohen_h(baseline_score, proposed_score)
    effect_size = (
        "large"  if abs(effect) >= 0.8 else
        "medium" if abs(effect) >= 0.5 else
        "small"  if abs(effect) >= 0.2 else
        "negligible"
    )

    # Required sample size for 80% power, α=0.05 (approximation)
    if abs(delta) > 0.001:
        required_n = int(8 * (baseline_score * (1 - baseline_score)) / (delta ** 2))
    else:
        required_n = float("inf")

    is_significant_hint = n_test_samples >= required_n * 0.8

    return json.dumps({
        "metric":                metric,
        "baseline":              baseline_score,
        "proposed":              proposed_score,
        "delta":                 round(delta, 4),
        "effect_size":           effect_size,
        "cohen_h":               round(effect, 3),
        "required_n_for_significance": required_n if required_n != float("inf") else ">10000",
        "your_test_set_n":       n_test_samples,
        "statistically_detectable": is_significant_hint,
        "recommended_tests": [
            "McNemar's test — for comparing two classifiers on same test set (paired)",
            "Bootstrap CI — resample test set 10K times, report 95% CI on difference",
            "5×2 CV paired t-test — most robust when you have training budget",
        ],
        "bootstrap_recipe": (
            "from sklearn.utils import resample\n"
            "diffs = []\n"
            "for _ in range(10000):\n"
            "    idx = resample(range(n), replace=True)\n"
            "    d = metric_fn(y_true[idx], pred_A[idx]) - metric_fn(y_true[idx], pred_B[idx])\n"
            "    diffs.append(d)\n"
            "ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])\n"
            "p_value = (sum(d <= 0 for d in diffs) / len(diffs))  # one-sided"
        ),
        "interpretation": (
            f"With {n_test_samples} test samples and delta={delta:.3f}, "
            f"the improvement is {'LIKELY statistically significant' if is_significant_hint else 'POSSIBLY NOT significant'}. "
            f"Effect size is {effect_size}. "
            f"For paper submission, report 95% CI and p-value. "
            f"p < 0.05 with multiple experiments (n={n_experiments}) is considered significant."
        ),
    }, indent=2)
