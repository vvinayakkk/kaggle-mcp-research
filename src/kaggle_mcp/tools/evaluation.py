"""
Evaluation, iteration, and review tools — the "red team" for your ML approach.

Design:  These tools provide DATA + FRAMEWORKS that guide Copilot's intelligent reasoning.
         They fetch real SOTA scores, embed actual venue rubrics, and apply domain heuristics.

Pipeline position (compulsory before notebook generation):
    arch_v1 → brutal_evaluate → reiterate_architecture ×3 → roast_approach
            → reviewer_perspective → paper_worthiness → [if strong] q1_journal_analysis
            → [only then] generate_notebook
"""
from __future__ import annotations

import json
import re
from typing import Optional

import requests

from kaggle_mcp.config import PAPERS_WITH_CODE_API, SEMANTIC_SCHOLAR_API

# ════════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASES
# ════════════════════════════════════════════════════════════════════════════════

_ARCH_PARAMS_M: dict[str, int] = {
    "alexnet": 61, "vgg16": 138, "vgg19": 144,
    "resnet18": 11, "resnet34": 21, "resnet50": 25, "resnet101": 44, "resnet152": 60,
    "resnext50": 25, "wide_resnet50": 69,
    "densenet121": 8, "densenet201": 20,
    "mobilenet_v2": 3, "mobilenet_v3_large": 5,
    "efficientnet_b0": 5, "efficientnet_b1": 7, "efficientnet_b4": 19,
    "efficientnet_v2_s": 21, "efficientnet_v2_m": 54, "efficientnet_v2_l": 120,
    "convnext_tiny": 28, "convnext_small": 50, "convnext_base": 88, "convnext_large": 197,
    "convnext_v2_tiny": 28, "convnext_v2_base": 89, "convnext_v2_large": 198,
    "vit_b_16": 86, "vit_l_16": 307,
    "swin_t": 28, "swin_s": 50, "swin_b": 88, "swin_l": 197,
    "deit_small": 22, "deit_base": 86,
    "eva_base": 86, "eva_large": 304,
}

_GOOD_PRACTICES: dict[str, list[str]] = {
    "augmentation":   ["mixup", "cutmix", "randaugment", "augment", "albumentation",
                       "flip", "rotate", "crop", "jitter", "autoaugment", "mosaic"],
    "regularization": ["dropout", "weight_decay", "label_smooth", "stochastic_depth",
                       "droppath", "ema", "swa", "mixout"],
    "scheduler":      ["onecycle", "cosine", "step_lr", "plateau", "warmup", "scheduler"],
    "precision":      ["amp", "fp16", "mixed_precision", "autocast", "bfloat16"],
    "stability":      ["early_stopping", "checkpoint", "gradient_clip", "clip_grad"],
    "inference":      ["tta", "test_time", "ensemble", "swa", "sliding_window"],
    "validation":     ["cross_valid", "stratified", "kfold", "oof", "5-fold", "3-fold"],
}

_TASK_FAILURE_MODES: dict[str, list[str]] = {
    "image_classification": [
        "Data leakage: augmented test frames from training video",
        "Ignoring class imbalance — macro F1 will be misleading",
        "Not normalising with ImageNet stats for pretrained backbone",
        "Starting with high LR destroys pretrained ImageNet features",
        "BatchNorm unstable with batch_size < 8 per GPU",
        "TTA not used at inference — free +0.5-2% F1 left on table",
        "Only one random seed — high-variance performance estimate",
        "No stratified K-fold — val set may not represent all classes",
        "Frozen backbone throughout — fine-grained tasks require unfreezing",
    ],
    "nlp_classification": [
        "Learning rate too high (> 5e-5) catastrophically forgets pretrained weights",
        "Truncating from the wrong side for long documents",
        "Batch size < 16 leads to noisy gradient updates for transformers",
        "Not using gradient accumulation to simulate larger batches",
        "Missing [CLS] pooling strategy (mean vs first token vs last layer)",
    ],
    "tabular": [
        "Target encoding without cross-validation causes data leakage",
        "Missing value strategy differs between train and test",
        "High-cardinality features blow up GPU memory",
        "Temporal data needs time-aware splits — no random shuffle",
        "Not checking duplicate rows between train/test",
    ],
    "object_detection": [
        "Wrong anchor sizes for target object scales",
        "NMS threshold too aggressive — misses overlapping objects",
        "Not converting bbox format correctly (xyxy vs xywh vs normalized)",
        "Confidence threshold too high at inference — recall collapses",
    ],
}

_VENUE_RUBRICS: dict = {
    "ICLR": {
        "full_name": "International Conference on Learning Representations",
        "tier": "S",
        "acceptance_rate": 0.31,
        "review_criteria": {
            "soundness":     {"max": 4, "desc": "Technical correctness and rigor of methodology"},
            "presentation":  {"max": 4, "desc": "Clarity, organisation, and reproducibility"},
            "contribution":  {"max": 4, "desc": "Significance and originality of contribution"},
            "rating":        {"max": 10, "desc": "Overall recommendation"},
        },
        "score_map": {10: "Strong Accept", 9: "Strong Accept", 8: "Accept",
                      7: "Accept", 6: "Weak Accept", 5: "Borderline",
                      4: "Weak Reject", 3: "Reject", 2: "Strong Reject", 1: "Strong Reject"},
        "must_have": [
            "Novel contribution clearly differentiated from related work",
            "Ablation isolating each proposed component's contribution",
            "Reproducible: code/seeds/hyperparameters provided or promised",
            "Tested on at least 2 independent benchmarks/datasets",
            "Statistical significance or error bars on key results",
            "Failure cases and limitations explicitly discussed",
            "Computational cost analysis (FLOPs, parameters, wall-clock)",
        ],
        "instant_reject": [
            "Results only on a single small dataset < 10K samples",
            "No comparison to any SOTA baseline",
            "Theoretical claims asserted without proof",
            "Reproducibility impossible: no code, no hyperparameters",
            "Abstract overclaims not supported by any experiment",
        ],
    },
    "NeurIPS": {
        "full_name": "Conference on Neural Information Processing Systems",
        "tier": "S",
        "acceptance_rate": 0.26,
        "review_criteria": {
            "originality":   {"max": 4, "desc": "How novel is the contribution?"},
            "quality":       {"max": 4, "desc": "Technical quality and correctness"},
            "clarity":       {"max": 4, "desc": "Is the paper clearly written?"},
            "significance":  {"max": 4, "desc": "Long-term significance to the field"},
        },
        "must_have": [
            "Theoretical analysis (proofs, bounds, or convergence guarantees) preferred",
            "Extensive experimental validation across multiple settings",
            "Comparison to ALL relevant recent baselines (last 2 years)",
            "Ablation study demonstrating each design choice",
            "Broader impact + societal implications addressed",
            "Code and dataset availability statement",
        ],
        "instant_reject": [
            "Purely incremental (<1% improvement on all benchmarks)",
            "No theoretical motivation for empirical choices",
            "NeurIPS checklist not completed",
            "Experiments cannot be reproduced without author assistance",
        ],
    },
    "CVPR": {
        "full_name": "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        "tier": "S",
        "acceptance_rate": 0.25,
        "review_criteria": {
            "novelty":           {"max": 5, "desc": "Originality of the proposed approach"},
            "technical_quality": {"max": 5, "desc": "Technical soundness and correctness"},
            "potential_impact":  {"max": 5, "desc": "Expected impact on CV research"},
            "presentation":      {"max": 5, "desc": "Clarity of writing and figures"},
        },
        "must_have": [
            "State-of-the-art or competitive results on widely-used CV benchmarks",
            "Both qualitative AND quantitative results",
            "Ablation on all key architectural choices",
            "Runtime / efficiency comparison",
            "Clear insight — NOT just an engineering combination",
        ],
        "instant_reject": [
            "Benchmarks not standard in the CV community",
            "Only marginal numerical improvement without new insight",
            "Cherry-picked qualitative examples hinting at failures",
        ],
    },
    "AAAI": {
        "full_name": "AAAI Conference on Artificial Intelligence",
        "tier": "A",
        "acceptance_rate": 0.23,
        "review_criteria": {
            "significance":    {"max": 5},
            "novelty":         {"max": 5},
            "correctness":     {"max": 5},
            "clarity":         {"max": 5},
        },
        "must_have": [
            "Clear problem definition and motivation",
            "Solid experimental comparison",
            "Discussion of assumptions and limitations",
        ],
        "instant_reject": ["No clear contribution beyond existing methods"],
    },
    "Neurips_workshop": {
        "full_name": "NeurIPS Workshop (lower bar, good starting point)",
        "tier": "C",
        "acceptance_rate": 0.50,
        "review_criteria": {"overall": {"max": 5}},
        "must_have": ["Interesting idea + preliminary results"],
        "instant_reject": ["Zero experimental validation"],
    },
}

_Q1_JOURNALS: dict = {
    "computer_vision": [
        {"name": "IEEE TPAMI", "impact_factor": 23.6, "tier": "S",
         "review_months": 6, "acceptance_rate": 0.15,
         "url": "https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34",
         "requirements": [
             "≥30% new content vs any prior conference version",
             "Comprehensive related work (40+ references)",
             "3+ diverse benchmark datasets",
             "Full ablation and computational analysis",
             "Code publicly available",
         ]},
        {"name": "IJCV", "impact_factor": 13.2, "tier": "A", "review_months": 5,
         "url": "https://www.springer.com/journal/11263",
         "requirements": ["Strong novel hypothesis", "2+ datasets", "All SOTA baselines"]},
        {"name": "IEEE TIP", "impact_factor": 10.6, "tier": "A", "review_months": 4,
         "url": "https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=83",
         "requirements": ["Image/video signal processing angle", "Quantitative metrics", "PSNR/SSIM if applicable"]},
        {"name": "Pattern Recognition", "impact_factor": 7.9, "tier": "B", "review_months": 3,
         "url": "https://www.journals.elsevier.com/pattern-recognition",
         "requirements": ["Pattern recognition method or analysis", "Comparative study"]},
    ],
    "nlp": [
        {"name": "TACL", "impact_factor": 7.0, "tier": "S", "review_months": 3,
         "requirements": ["Novel NLP contribution", "Human evaluation if generative", "Multiple datasets"]},
        {"name": "IEEE TKDE", "impact_factor": 8.9, "tier": "A", "review_months": 4,
         "requirements": ["Knowledge representation or data engineering focus"]},
    ],
    "general_ml": [
        {"name": "JMLR", "impact_factor": 6.0, "tier": "S", "review_months": 8,
         "url": "https://jmlr.org",
         "requirements": ["Open-source code required", "Theoretical guarantees OR extensive study", "No page limit — be thorough"]},
        {"name": "IEEE TNNLS", "impact_factor": 14.3, "tier": "A", "review_months": 5,
         "requirements": ["Neural network/learning systems angle", "Theoretical contribution preferred"]},
        {"name": "Machine Learning (Springer)", "impact_factor": 5.4, "tier": "B", "review_months": 4,
         "requirements": ["General ML method or survey", "Multiple benchmarks"]},
    ],
}

# Improvement prescriptions keyed by what's MISSING
_REITERATION_PRESCRIPTIONS: dict[str, dict] = {
    "augmentation": {
        "image_classification": "Add MixUp (α=0.4) + CutMix (α=1.0) + RandAugment(num_ops=2, magnitude=9). "
                                 "Expected F1 gain: +2-5% on small datasets (<20K images).",
        "object_detection":     "Add Mosaic augmentation + Copy-Paste + multi-scale training. "
                                 "Expected mAP gain: +3-6%.",
        "nlp_classification":   "Add back-translation (EN→DE→EN) + synonym replacement (WordNet). "
                                 "Expected F1 gain: +1-3% on low-resource tasks.",
        "tabular":              "Add Gaussian noise injection + feature dropout during training. Not applicable to all tabular tasks.",
    },
    "regularization": {
        "image_classification": "Add Label Smoothing (ε=0.1) + DropPath (p=0.1) + EMA (decay=0.9999). "
                                 "Expected F1 gain: +1-3%.",
        "nlp_classification":   "Add Mixout (p=0.1) + weight decay (0.01) to ALL layers. "
                                 "Expected F1 gain: +0.5-2%.",
        "tabular":              "Add L1/L2 regularisation to neural components. Use feature subsampling in tree models.",
    },
    "scheduler": {
        "image_classification": "Replace StepLR with OneCycleLR (max_lr=config_lr, pct_start=warmup/total_epochs). "
                                 "Expected convergence: 20% fewer epochs.",
        "nlp_classification":   "Use linear warmup for 10% of steps, then linear decay. "
                                 "Critical for transformer fine-tuning stability.",
    },
    "validation": {
        "image_classification": "Use 5-fold StratifiedKFold. Train all 5 folds, ensemble predictions. "
                                 "Expected F1 gain: +1-4% vs single fold.",
        "tabular":              "5-fold GroupKFold if temporal data, else StratifiedKFold. "
                                 "OOF predictions for stacking.",
    },
    "inference": {
        "image_classification": "Add TTA: HorizontalFlip + 5-crop + original = 7 predictions averaged. "
                                 "Expected F1 gain: +0.5-2%.",
        "object_detection":     "Add multi-scale TTA (0.5×, 0.75×, 1.0×, 1.25× input sizes) + WBF ensemble. "
                                 "Expected mAP gain: +2-4%.",
    },
    "backbone_upgrade": {
        "image_classification": {
            "upgrade_path": [
                ("resnet50", "convnext_tiny",  "+8-12% F1"),
                ("resnet101", "convnext_small", "+6-9% F1"),
                ("efficientnet_b0", "efficientnet_v2_s", "+4-7% F1"),
                ("efficientnet_v2_s", "convnext_v2_base", "+3-5% F1 (2× compute)"),
                ("convnext_base", "swin_b", "+1-3% F1, better fine-grained"),
                ("swin_b", "vit_l_16 (ImageNet-21k)", "+2-5% F1, requires 16GB VRAM"),
            ],
        },
    },
}


# ════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _detect_arch_and_params(text: str) -> tuple[str, int]:
    text_l = text.lower().replace("-", "_").replace(" ", "_")
    for arch, params in _ARCH_PARAMS_M.items():
        if arch.replace("_", "") in text_l.replace("_", ""):
            return arch, params
    return "custom", -1


def _audit_practices(text: str) -> dict[str, list[str]]:
    """Return found and missing good practices per category."""
    text_l = text.lower()
    audit = {}
    for category, keywords in _GOOD_PRACTICES.items():
        found   = [k for k in keywords if k in text_l]
        missing = [k for k in keywords[:2] if k not in text_l]
        audit[category] = {"found": found, "missing": missing, "score": len(found) / max(len(keywords), 1)}
    return audit


def _fetch_sota_quick(task_type: str) -> list[dict]:
    """Pull top-3 SOTA entries for a task from Papers With Code."""
    task_slug_map = {
        "image_classification": "image-classification",
        "nlp_classification":   "text-classification",
        "object_detection":     "object-detection",
        "tabular":              "tabular-classification",
        "image_segmentation":   "semantic-segmentation",
    }
    slug = task_slug_map.get(task_type, task_type.replace("_", "-"))
    try:
        r = requests.get(f"{PAPERS_WITH_CODE_API}/sota/",
                         params={"task": slug}, timeout=10)
        if r.status_code != 200:
            return []
        benchmarks = r.json().get("results", [])
        for bm in benchmarks[:1]:
            rows = bm.get("sota", {}).get("rows", [])
            return [{"rank": i+1, "model": row.get("model_name", ""),
                     "score": row.get("metrics", {})}
                    for i, row in enumerate(rows[:3])]
    except Exception:
        return []
    return []


# ════════════════════════════════════════════════════════════════════════════════
# PUBLIC TOOLS
# ════════════════════════════════════════════════════════════════════════════════

def brutal_evaluate(
    arch_description: str,
    task_type:        str = "image_classification",
    dataset_info:     str = "",
    target_metric:    str = "f1",
    competition_slug: str = "",
) -> str:
    """
    Brutally honest technical evaluation of a proposed ML approach.

    Returns structured JSON with:
      • FATAL_FLAWS      — will definitely fail/underperform
      • SERIOUS_CONCERNS — likely significant performance impact
      • MINOR_ISSUES     — nice-to-have improvements
      • PRACTICE_AUDIT   — missing good practices by category
      • SOTA_COMPARISON  — top-3 SOTA for context
      • OVERALL_SCORE    — 0-10 readiness rating
      • VERDICT          — proceed / reiterate / scrap
    """
    text_l  = arch_description.lower()
    arch, params = _detect_arch_and_params(arch_description)
    audit   = _audit_practices(arch_description)
    sota    = _fetch_sota_quick(task_type)
    failures = _TASK_FAILURE_MODES.get(task_type, [])

    fatal, serious, minor = [], [], []

    # ── Dataset-size vs model-params overfitting check ────────────────────────
    size_match = re.search(r"(\d[\d,_]*)\s*(image|sample|row|example|record)", text_l)
    if size_match:
        n = int(size_match.group(1).replace(",", "").replace("_", ""))
        if params > 0 and n < params * 100:
            fatal.append(
                f"OVERFITTING RISK: {arch} has {params}M params but dataset has only ~{n:,} samples. "
                f"Rule of thumb: need ≥{params * 100:,} samples. Switch to a smaller backbone "
                f"or use heavy regularisation + augmentation."
            )

    # ── Augmentation check for vision ────────────────────────────────────────
    if task_type in ("image_classification", "object_detection", "image_segmentation"):
        if not audit["augmentation"]["found"]:
            fatal.append(
                "NO AUGMENTATION: Vision task with no data augmentation will severely overfit. "
                "Minimum: RandomHorizontalFlip + ColorJitter + RandomCrop. "
                "Better: MixUp + CutMix + RandAugment."
            )
        elif len(audit["augmentation"]["found"]) < 2:
            serious.append(
                f"WEAK AUGMENTATION: Only {audit['augmentation']['found']} detected. "
                "Add MixUp + CutMix for +2-5% F1 on typical competition datasets."
            )

    # ── Learning rate and scheduler check ────────────────────────────────────
    if not audit["scheduler"]["found"]:
        serious.append(
            "NO LR SCHEDULER: Flat learning rate hurts convergence and final performance. "
            "Add OneCycleLR or CosineAnnealingLR. Expected +1-3% F1."
        )

    # ── Label smoothing / regularisation ──────────────────────────────────────
    if not audit["regularization"]["found"]:
        serious.append(
            "NO REGULARISATION: No dropout/weight_decay/label_smoothing detected. "
            "Training will likely overfit especially on small datasets. "
            "Add weight_decay=1e-4 + label_smoothing=0.1 minimum."
        )

    # ── Mixed precision ───────────────────────────────────────────────────────
    if not audit["precision"]["found"]:
        minor.append(
            "NO MIXED PRECISION: Without FP16/AMP, training is ~2× slower and uses 2× GPU memory. "
            "Add torch.cuda.amp.autocast() — free performance gain."
        )

    # ── Cross-validation / multiple seeds ─────────────────────────────────────
    if not audit["validation"]["found"]:
        serious.append(
            "SINGLE FOLD: Single train/val split is a high-variance performance estimate. "
            "Add 5-fold StratifiedKFold. OOF predictions also improve final ensembling."
        )

    # ── TTA at inference ──────────────────────────────────────────────────────
    if not audit["inference"]["found"]:
        minor.append(
            "NO TEST-TIME AUGMENTATION: TTA is a free +0.5-2% F1 gain at inference time. "
            "Add horizontal flip + multi-crop at minimum."
        )

    # ── Task-specific failure modes ───────────────────────────────────────────
    for fm in failures[:3]:
        minor.append(f"KNOWN FAILURE MODE for {task_type}: {fm}")

    # ── SOTA comparison ──────────────────────────────────────────────────────
    sota_context = sota if sota else [{"note": "SOTA data unavailable — check Papers With Code manually"}]

    # ── Score calculation ─────────────────────────────────────────────────────
    deductions = len(fatal) * 3 + len(serious) * 1.5 + len(minor) * 0.3
    overall_score = max(0, min(10, round(10 - deductions, 1)))
    verdict       = ("PROCEED — ready for notebook" if overall_score >= 7
                     else "REITERATE — fix fatal/serious issues first" if overall_score >= 4
                     else "SCRAP AND REDESIGN — too many fundamental problems")

    return json.dumps({
        "architecture": arch,
        "params_M":     params,
        "task_type":    task_type,
        "FATAL_FLAWS":      fatal,
        "SERIOUS_CONCERNS": serious,
        "MINOR_ISSUES":     minor,
        "PRACTICE_AUDIT":   audit,
        "KNOWN_FAILURE_MODES_FOR_TASK": failures,
        "SOTA_TOP3":        sota_context,
        "OVERALL_SCORE":    overall_score,
        "VERDICT":          verdict,
    }, indent=2)


def reiterate_architecture(
    arch_description:    str,
    task_type:           str = "image_classification",
    evaluation_feedback: str = "",
    iteration_num:       int = 1,
) -> str:
    """
    Systematically hardens an architecture based on evaluation feedback.
    Applies iteration-specific improvements: each round tackles different layers.

    Iteration 1 — Backbone & augmentation (biggest wins)
    Iteration 2 — Training recipe & regularisation (stability)
    Iteration 3 — Inference & ensembling (final squeeze)

    Returns: { original, changes_applied, improved_description, expected_delta, next_iteration_focus }
    """
    text_l  = arch_description.lower()
    arch, _ = _detect_arch_and_params(arch_description)
    audit   = _audit_practices(arch_description)

    changes      = []
    improvements = {}

    if iteration_num == 1:
        # Backbone upgrade
        backbones = _REITERATION_PRESCRIPTIONS.get("backbone_upgrade", {}).get(task_type, {})
        for old_b, new_b, gain in backbones.get("upgrade_path", []):
            if old_b.replace("_", "") in text_l.replace("_", "").replace("-", ""):
                changes.append({"type": "BACKBONE_UPGRADE", "from": old_b, "to": new_b, "expected_gain": gain})
                improvements["backbone"] = f"Replace {old_b} → {new_b} ({gain})"
                break

        # Augmentation
        if not audit["augmentation"]["found"]:
            aug = _REITERATION_PRESCRIPTIONS.get("augmentation", {})
            aug_fix = aug.get(task_type, aug.get("image_classification", "Add strong augmentation"))
            changes.append({"type": "ADD_AUGMENTATION", "prescription": aug_fix})
            improvements["augmentation"] = aug_fix

        focus_next = "Training recipe: scheduler, optimiser, regularisation"

    elif iteration_num == 2:
        # Scheduling
        if not audit["scheduler"]["found"]:
            sched = _REITERATION_PRESCRIPTIONS.get("scheduler", {})
            sched_fix = sched.get(task_type, "Add OneCycleLR scheduler")
            changes.append({"type": "ADD_SCHEDULER", "prescription": sched_fix})

        # Regularisation
        if not audit["regularization"]["found"]:
            reg = _REITERATION_PRESCRIPTIONS.get("regularization", {})
            reg_fix = reg.get(task_type, "Add dropout + weight_decay + label_smoothing")
            changes.append({"type": "ADD_REGULARISATION", "prescription": reg_fix})

        # Mixed precision
        if not audit["precision"]["found"]:
            changes.append({"type": "ADD_AMP",
                             "prescription": "Enable torch.cuda.amp.autocast() + GradScaler for FP16. "
                                             "2× training speed, 2× memory efficiency."})

        # Cross-validation
        if not audit["validation"]["found"]:
            val = _REITERATION_PRESCRIPTIONS.get("validation", {})
            val_fix = val.get(task_type, "Use 5-fold StratifiedKFold")
            changes.append({"type": "ADD_CV", "prescription": val_fix})

        focus_next = "Inference: TTA, ensembling, post-processing"

    elif iteration_num >= 3:
        # Inference and final polish
        if not audit["inference"]["found"]:
            tta = _REITERATION_PRESCRIPTIONS.get("inference", {})
            tta_fix = tta.get(task_type, "Add TTA: flip + multi-crop averaging")
            changes.append({"type": "ADD_TTA", "prescription": tta_fix})

        changes.append({
            "type": "ADD_ENSEMBLE_PLAN",
            "prescription": (
                "Train 3 models: (1) current arch, (2) different backbone, (3) different image_size. "
                "Average softmax probabilities. Expected +2-5% F1 over single model."
            ),
        })
        changes.append({
            "type": "ADD_POST_PROCESSING",
            "prescription": (
                "Apply temperature scaling on logits before final predictions. "
                "Use val set to find optimal threshold per class for imbalanced data."
            ),
        })

        focus_next = "COMPLETE — architecture is hardened. Proceed to notebook generation."

    # Build improved description
    improved = arch_description
    for change in changes:
        improved += f"\n[ITER-{iteration_num} ADD] {change.get('prescription', change.get('type',''))}"

    total_expected_gain = {
        1: "Expected cumulative F1 gain after iteration 1: +8-18%",
        2: "Expected cumulative F1 gain after iteration 2: +12-25%",
        3: "Expected cumulative F1 gain after iteration 3: +15-30% vs baseline",
    }.get(iteration_num, "")

    return json.dumps({
        "iteration":             iteration_num,
        "original_description":  arch_description,
        "changes_applied":       changes,
        "improved_description":  improved,
        "expected_delta":        total_expected_gain,
        "next_iteration_focus":  focus_next,
        "ready_for_notebook":    iteration_num >= 3,
    }, indent=2)


def roast_approach(
    arch_description: str,
    task_type:        str = "image_classification",
    competition_context: str = "",
) -> str:
    """
    Simulates a senior ML researcher's blunt critique of a proposed approach.
    Returns technically grounded, specific criticisms — not generic advice.

    Output: { punchline, brutal_observations, what_a_winner_does_instead,
              technical_debt_list, redemption_arc, roast_score }
    """
    text_l  = arch_description.lower()
    arch, params = _detect_arch_and_params(arch_description)
    audit   = _audit_practices(arch_description)

    observations = []
    tech_debt     = []

    # ── Architecture-specific roasts ─────────────────────────────────────────
    if "resnet50" in text_l and "2015" not in text_l:
        observations.append(
            "ResNet50 in 2025. Bold choice. Why not use a DCGAN while you're at it? "
            "ConvNeXt-Tiny beats ResNet50 by 8% F1 and has fewer parameters."
        )
        tech_debt.append("Backbone from 2015 — upgrade to ConvNeXt or EfficientNetV2")

    if "vgg" in text_l:
        observations.append(
            "VGG? Are we doing a history lesson? This backbone was deprecated before "
            "half your training data was photographed."
        )
        tech_debt.append("VGG — replace immediately with anything from 2020+")

    if "adam" in text_l and "weight_decay" not in text_l and "adamw" not in text_l:
        observations.append(
            "Using Adam without weight_decay. Classic. You're paying the AdamW tax "
            "in overfitting penalties. Switch to AdamW — it's literally Adam + one number."
        )
        tech_debt.append("Adam → AdamW (add weight_decay=1e-4)")

    if "sgd" in text_l and "momentum" not in text_l:
        observations.append(
            "SGD without momentum is just rolling a ball down a mountain and hoping "
            "it lands in the right valley. Add momentum=0.9."
        )
        tech_debt.append("SGD needs momentum=0.9 minimum")

    if not audit["augmentation"]["found"] and task_type in ("image_classification", "object_detection"):
        observations.append(
            "No augmentation on a vision task. Your model will memorise the exact pixel "
            "arrangement of each training image and confidently predict garbage on test. "
            "This isn't deep learning, it's deep memorisation."
        )
        tech_debt.append("Zero augmentation on vision task — critical gap")

    if not audit["scheduler"]["found"]:
        observations.append(
            "Fixed learning rate throughout training. You're asking the model to take "
            "the same size steps when it's a kilometre away AND when it's a millimetre away. "
            "Add OneCycleLR. This is 2015 called, they want their training recipe back."
        )
        tech_debt.append("No LR scheduler — convergence will be suboptimal")

    if not audit["validation"]["found"]:
        observations.append(
            "Single train/val split. Congratulations, you've built a model that performs "
            "excellently on one specific random seed. That's not a model, that's luck. "
            "Add 5-fold CV."
        )
        tech_debt.append("Single fold — unreliable performance estimate")

    # Competition-specific observations
    if competition_context:
        observations.append(
            f"Competition context: '{competition_context[:100]}'. "
            "Have you actually looked at the top notebooks on this competition? "
            "Past winners on similar tasks share their approaches in the discussion tab."
        )

    # What winners do instead
    winner_strategies = {
        "image_classification": [
            "Ensemble 3+ models: ConvNeXt + EfficientNetV2 + ViT",
            "5-fold CV with OOF predictions for robust estimate",
            "MixUp + CutMix + aggressive augmentation pipeline",
            "Pseudo-labelling on test set in round 2",
            "Progressive resizing: train at 128→192→224→288",
            "TTA with 7+ transforms at inference",
        ],
        "nlp_classification": [
            "DeBERTa-v3-large or DeBERTa-v3-base fine-tuning",
            "Multi-fold training with ensemble",
            "Mixout + AWP (Adversarial Weight Perturbation)",
            "Back-translation augmentation",
            "Custom pooling: mean-max-attention hybrid",
        ],
        "tabular": [
            "LGBM + CatBoost + XGBoost 3-level stacking",
            "Optuna-tuned with 200+ trials",
            "Feature engineering: rolling stats, lag features, interactions",
            "Target-encoded features with strict CV",
            "Neural network (MLP/TabNet) as 4th stack member",
        ],
    }

    punchline = (
        f"This looks like a first draft written at 2am before a deadline. "
        f"It uses {arch} with {len(tech_debt)} critical improvements needed. "
        f"A Kaggle grandmaster would rewrite this in 20 minutes."
        if tech_debt else
        f"Actually... not terrible. {arch} is reasonable and has {10 - len(observations)} good practices. "
        f"But it still needs polishing."
    )

    roast_score = max(0, min(10, 10 - len(tech_debt) * 1.5 - len(observations) * 0.5))

    return json.dumps({
        "punchline":                  punchline,
        "brutal_observations":        observations,
        "technical_debt_list":        tech_debt,
        "what_a_kaggle_grandmaster_does_instead": winner_strategies.get(task_type, []),
        "redemption_arc":             (
            "Fix the tech debt list above in order of severity. "
            "After iteration 3, this could be a solid competition solution. "
            "The architecture is not the limiting factor — the training recipe is."
            if tech_debt else
            "Solid foundation. Iterate on the training recipe and add ensembling."
        ),
        "roast_score_10":             roast_score,
        "is_kaggle_medal_worthy":     roast_score >= 7,
    }, indent=2)


def reviewer_perspective(
    arch_description: str,
    results_summary:  str = "",
    target_venue:     str = "ICLR",
) -> str:
    """
    Simulate a rigorous peer reviewer at a top ML venue.

    Returns: { venue_context, scores, review_text, verdict,
               strengths, weaknesses, questions_for_authors,
               required_changes, confidence, recommendation }
    """
    rubric  = _VENUE_RUBRICS.get(target_venue.upper(), _VENUE_RUBRICS["ICLR"])
    text_l  = arch_description.lower()
    audit   = _audit_practices(arch_description)
    arch, params = _detect_arch_and_params(arch_description)

    strengths   = []
    weaknesses  = []
    questions   = []
    required    = []
    scores      = {}

    # ── Assess each criterion ────────────────────────────────────────────────
    for criterion, info in rubric["review_criteria"].items():
        max_score = info["max"]
        score     = max_score  # start full, deduct

        if criterion in ("novelty", "contribution", "originality"):
            if arch in ("resnet50", "resnet101", "vgg16"):
                score = max(1, score - max_score // 2)
                weaknesses.append(f"NOVELTY: Using {arch} without modification is not a novel contribution.")
            elif any(k in text_l for k in ["novel", "propose", "new", "introduce"]):
                strengths.append(f"Claims novelty — verify this is substantiated by related work discussion.")
            else:
                score = max(1, score - 1)

        if criterion in ("quality", "soundness", "technical_quality"):
            if not audit["validation"]["found"]:
                score = max(1, score - 1)
                weaknesses.append("No cross-validation: single split result is statistically unreliable.")
            if not results_summary:
                score = max(1, score - 1)
                weaknesses.append("No experimental results provided to evaluate.")
            else:
                strengths.append("Experimental results provided — reviewer can assess claim validity.")

        if criterion in ("presentation", "clarity"):
            if len(arch_description) < 100:
                score = max(1, score - 1)
                weaknesses.append("Architecture description is too brief to assess reproducibility.")
            else:
                strengths.append("Sufficient detail to evaluate technical approach.")

        if criterion in ("significance", "potential_impact"):
            sota_available = bool(_fetch_sota_quick(
                "image_classification" if "image" in text_l else
                "text-classification"  if "bert" in text_l else "general"
            ))
            if not sota_available:
                score = max(1, score - 1)
            else:
                strengths.append("SOTA context available — impact can be quantified.")

        scores[criterion] = {"score": score, "max": max_score}

    # ── Check must-haves ─────────────────────────────────────────────────────
    for item in rubric.get("must_have", []):
        keywords = item.lower().split()[:3]
        if not any(kw in text_l for kw in keywords):
            required.append(f"REQUIRED BY {target_venue}: {item}")
            questions.append(f"How does your work address: '{item}'?")

    # ── Check instant reject triggers ─────────────────────────────────────────
    instant_reject_triggered = []
    for trigger in rubric.get("instant_reject", []):
        kws = trigger.lower().split()[:3]
        if any(kw in text_l for kw in kws):
            instant_reject_triggered.append(trigger)

    # ── Compute overall recommendation ───────────────────────────────────────
    avg_score    = sum(s["score"] for s in scores.values()) / len(scores)
    max_avg      = max(s["max"] for s in scores.values())
    pct          = avg_score / max_avg
    if instant_reject_triggered:
        verdict, recommendation = "REJECT", "Strong Reject"
    elif pct >= 0.85:
        verdict, recommendation = "ACCEPT", "Accept / Strong Accept"
    elif pct >= 0.70:
        verdict, recommendation = "WEAK ACCEPT", "Weak Accept — minor revisions required"
    elif pct >= 0.55:
        verdict, recommendation = "BORDERLINE", "Borderline — major revisions required"
    else:
        verdict, recommendation = "REJECT", "Reject — fundamental issues"

    return json.dumps({
        "venue":                   target_venue,
        "venue_full_name":         rubric["full_name"],
        "venue_acceptance_rate":   f"{rubric['acceptance_rate']*100:.0f}%",
        "scores":                  scores,
        "strengths":               strengths,
        "weaknesses":              weaknesses,
        "instant_reject_triggers": instant_reject_triggered,
        "questions_for_authors":   questions[:6],
        "required_changes":        required,
        "verdict":                 verdict,
        "recommendation":          recommendation,
        "confidence":              "3/5 — based on abstract/description only (no full paper)",
        "review_note":             (
            f"This review is based on the architecture description and available results. "
            f"A full paper review would cover related work citations, methodology detail, "
            f"and supplementary material. Acceptance rate at {target_venue}: "
            f"{rubric['acceptance_rate']*100:.0f}%."
        ),
    }, indent=2)


def paper_worthiness(
    arch_description:  str,
    results_summary:   str = "",
    target_venue:      str = "CVPR",
    ablation_done:     bool = False,
    code_available:    bool = False,
    num_datasets:      int  = 1,
) -> str:
    """
    Publication readiness assessment.

    Returns: { readiness_score, acceptance_probability, missing_for_submission,
               what_makes_it_publishable, minimum_venue, stretch_venue,
               estimated_weeks_to_ready }
    """
    rubric = _VENUE_RUBRICS.get(target_venue.upper(), _VENUE_RUBRICS["CVPR"])
    audit  = _audit_practices(arch_description)
    arch, _ = _detect_arch_and_params(arch_description)

    score        = 0
    max_score    = 100
    missing      = []
    present      = []

    # ── Scoring rubric ────────────────────────────────────────────────────────
    # 1. Novel contribution (30 pts)
    novel_keywords = ["propose", "novel", "new", "introduce", "design", "framework"]
    if any(k in arch_description.lower() for k in novel_keywords):
        score += 20
        present.append("Novel contribution claimed (20/30 pts — verify vs related work)")
    else:
        missing.append("CRITICAL: State the novel contribution explicitly. What does nobody else do?")

    # 2. Experimental rigor (25 pts)
    if results_summary:
        score += 10
        present.append("Results provided (10 pts)")
    else:
        missing.append("CRITICAL: No results. A submission without results = instant reject.")

    if num_datasets >= 2:
        score += 10
        present.append(f"Multiple datasets: {num_datasets} (10 pts)")
    elif num_datasets == 1:
        score += 4
        missing.append(f"IMPORTANT: Only 1 dataset. {target_venue} expects ≥2 for credibility.")
    else:
        missing.append("CRITICAL: No dataset evaluation.")

    if ablation_done:
        score += 5
        present.append("Ablation study present (5 pts)")
    else:
        missing.append("IMPORTANT: No ablation study. Reviewers will ask: 'How do you know each component helps?'")

    # 3. Technical quality (20 pts)
    if audit["validation"]["found"]:
        score += 10
        present.append("Cross-validation / rigorous evaluation (10 pts)")
    else:
        missing.append("IMPORTANT: Use K-fold CV or multiple seeds for reliable estimates.")

    if audit["precision"]["found"] or audit["stability"]["found"]:
        score += 5
        present.append("Training best practices (5 pts)")
    else:
        missing.append("MINOR: Mixed precision training not mentioned — raises reproducibility questions.")

    # 4. Reproducibility (15 pts)
    if code_available:
        score += 15
        present.append("Code available (15 pts) — major credibility boost")
    else:
        score += 5
        missing.append(
            "IMPORTANT: No code mentioned. Post code on GitHub — reviewers increasingly expect this. "
            "NeurIPS, ICLR require reproducibility checklist."
        )

    # 5. Communication (10 pts) — hard to assess without full text
    score += 5
    present.append("Communication: partial credit (5/10 — full paper needed for full assessment)")

    pct = score / max_score

    # ── Venue matching ────────────────────────────────────────────────────────
    if pct >= 0.85:
        min_venue, stretch_venue = "AAAI / ECCV", f"{target_venue} (ambitious but possible)"
    elif pct >= 0.70:
        min_venue, stretch_venue = "AAAI", "CVPR / ICCV (with polishing)"
    elif pct >= 0.55:
        min_venue, stretch_venue = "NeurIPS Workshop", "AAAI (with 3-4 weeks work)"
    elif pct >= 0.40:
        min_venue, stretch_venue = "arXiv preprint", "NeurIPS Workshop (with major work)"
    else:
        min_venue, stretch_venue = "Blog post / arXiv", "Need major research contributions first"

    weeks_to_ready = max(0, int((0.8 - pct) * 20))

    return json.dumps({
        "target_venue":        target_venue,
        "readiness_score":     f"{score}/{max_score}",
        "readiness_pct":       f"{pct*100:.0f}%",
        "acceptance_probability": (
            "~35%"  if pct >= 0.85 else
            "~15%"  if pct >= 0.70 else
            "~5%"   if pct >= 0.50 else
            "<2%"
        ),
        "what_is_present":     present,
        "missing_for_submission": missing,
        "minimum_realistic_venue": min_venue,
        "stretch_venue":       stretch_venue,
        "estimated_weeks_to_ready": weeks_to_ready,
        "priority_actions": [m for m in missing if m.startswith("CRITICAL")][:3],
    }, indent=2)


def q1_journal_analysis(
    arch_description: str,
    results_summary:  str = "",
    domain:           str = "computer_vision",
    has_theory:       bool = False,
    num_datasets:     int  = 1,
    num_baselines:    int  = 0,
) -> str:
    """
    Full Q1 journal submission analysis.

    Returns: { recommended_journals_ranked, per_journal_analysis,
               current_gap_score, required_experiments, expected_acceptance_prob }
    """
    journals = _Q1_JOURNALS.get(domain, _Q1_JOURNALS["computer_vision"])

    audit = _audit_practices(arch_description)
    arch, params = _detect_arch_and_params(arch_description)

    # Overall submission readiness score (journal bar is much higher than conference)
    base_score = 0
    if results_summary:   base_score += 20
    if has_theory:        base_score += 20
    if num_datasets >= 3: base_score += 20
    elif num_datasets == 2: base_score += 10
    if num_baselines >= 5: base_score += 15
    elif num_baselines >= 3: base_score += 8
    if audit["validation"]["found"]: base_score += 10
    if any(k in arch_description.lower() for k in ["novel", "propose"]): base_score += 15

    per_journal = []
    for j in journals:
        met       = []
        unmet     = []
        for req in j.get("requirements", []):
            kws = req.lower().split()[:4]
            if any(kw in arch_description.lower() or kw in results_summary.lower()
                   for kw in kws):
                met.append(req)
            else:
                unmet.append(req)

        match_pct = len(met) / max(len(j.get("requirements", [])), 1) * 100

        per_journal.append({
            "journal":               j["name"],
            "impact_factor":         j.get("impact_factor"),
            "tier":                  j.get("tier"),
            "review_months":         j.get("review_months"),
            "url":                   j.get("url", ""),
            "requirements_met":      met,
            "requirements_unmet":    unmet,
            "match_score":           f"{match_pct:.0f}%",
            "estimated_acceptance":  (
                f"~{j.get('acceptance_rate', 0.2)*100:.0f}%"
                if match_pct >= 80 else
                f"~{j.get('acceptance_rate', 0.2)*40:.0f}%"
            ),
            "recommendation": (
                "SUBMIT NOW" if match_pct >= 80 and base_score >= 70 else
                f"NEEDS WORK — address {len(unmet)} requirement(s) first"
            ),
        })

    required_experiments = []
    if num_datasets < 3:
        required_experiments.append(
            f"Evaluate on {3 - num_datasets} more benchmark dataset(s). "
            "Journals expect ≥3 diverse datasets."
        )
    if not has_theory:
        required_experiments.append(
            "Add theoretical motivation: convergence analysis, complexity bounds, or formal guarantees. "
            "Q1 journals expect more than pure empirics."
        )
    if num_baselines < 5:
        required_experiments.append(
            f"Compare to {5 - num_baselines} more baselines. "
            "Specifically include the last 2 years of published SOTA."
        )
    if not audit["validation"]["found"]:
        required_experiments.append(
            "Add rigorous statistical evaluation: error bars, significance tests, multiple seeds."
        )

    return json.dumps({
        "domain":                  domain,
        "submission_readiness":    f"{base_score}/100",
        "recommended_journals":    per_journal,
        "required_experiments":    required_experiments,
        "journal_vs_conference":   (
            "Journal submission requires: 30%+ new content vs any prior conference paper, "
            "3+ datasets, full ablation, theory where possible, and typically 6-12 month review. "
            "Expected rejection rate: 70-85% even for good work."
        ),
        "top_recommendation": per_journal[0]["journal"] if per_journal else "Insufficient data",
        "time_estimate_months": (
            2  if base_score >= 80 else
            4  if base_score >= 60 else
            8  if base_score >= 40 else 12
        ),
    }, indent=2)
