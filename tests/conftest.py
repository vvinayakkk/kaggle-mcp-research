"""
Shared pytest fixtures for the Kaggle Research MCP test suite.
"""
import json
import pytest


# ────────────────────────────────────────────────────────────────────────────────
# Architecture descriptions
# ────────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_arch():
    """Minimal arch — many missing practices (should score low on brutal_evaluate)."""
    return "ResNet50 fine-tuned on ImageNet, 6 classes, CrossEntropyLoss, Adam optimizer, 30 epochs."


@pytest.fixture
def strong_arch():
    """Well-specified arch — covers most good practices."""
    return (
        "EfficientNetV2-S fine-tuned on ImageNet-21K. "
        "MixUp (alpha=0.4) + CutMix (alpha=1.0) + RandAugment(2, 9) augmentation. "
        "AdamW optimizer (lr=3e-4, weight_decay=1e-4). "
        "OneCycleLR scheduler with warmup (5 epochs out of 30). "
        "Label smoothing (eps=0.1) + DropPath (p=0.1) regularisation. "
        "Mixed precision (AMP fp16). "
        "5-fold StratifiedKFold cross-validation. "
        "TTA at inference (HFlip + 5-Crop). "
        "EMA of weights (decay=0.9999). "
        "Gradient clipping (max_norm=1.0)."
    )


@pytest.fixture
def vgg_arch():
    """Ancient arch — should trigger roast."""
    return "VGG16 with SGD optimizer and no augmentation."


@pytest.fixture
def arch_with_overfitting_risk():
    """Large model on tiny dataset — should trigger FATAL overfitting warning."""
    return (
        "ViT-L/16 fine-tuned on 500 images, 6 classes. "
        "CrossEntropyLoss, AdamW, 100 epochs."
    )


# ────────────────────────────────────────────────────────────────────────────────
# Results summaries
# ────────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def weak_results():
    return json.dumps({"val_f1": 0.62, "val_acc": 0.66, "best_epoch": 28})


@pytest.fixture
def strong_results():
    return json.dumps({"val_f1": 0.91, "val_acc": 0.93, "best_epoch": 18,
                       "test_f1": 0.89, "leaderboard_score": 0.89})


# ────────────────────────────────────────────────────────────────────────────────
# Training logs
# ────────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def overfitting_log():
    lines = [
        f"Epoch {e:2d} | train_loss={1.3 - e*0.04:.3f} | val_loss={0.7 + e*0.03:.3f} | f1={0.5 + e*0.01:.3f}"
        for e in range(1, 26)
    ]
    return "\n".join(lines)


@pytest.fixture
def healthy_log():
    lines = [
        f"Epoch {e:2d} | train_loss={1.3 - e*0.035:.3f} | val_loss={1.1 - e*0.025:.3f} | f1={0.5 + e*0.015:.3f}"
        for e in range(1, 26)
    ]
    return "\n".join(lines)


@pytest.fixture
def plateau_log():
    """Training improves then plateaus."""
    lines = (
        [f"Epoch {e:2d} | train_loss={1.3 - e*0.04:.3f} | val_loss={1.0 - e*0.02:.3f} | f1={0.5 + e*0.02:.3f}"
         for e in range(1, 11)]
        +
        [f"Epoch {e:2d} | train_loss={0.87 - (e-10)*0.003:.3f} | val_loss=0.800 | f1=0.702"
         for e in range(11, 26)]
    )
    return "\n".join(lines)


@pytest.fixture
def early_convergence_log():
    """Best result reached at epoch 5 of 30."""
    lines = (
        [f"Epoch {e:2d} | train_loss={1.2 - e*0.06:.3f} | val_loss={1.0 - e*0.04:.3f} | f1={0.55 + e*0.04:.3f}"
         for e in range(1, 6)]
        +
        [f"Epoch {e:2d} | train_loss={0.85 - e*0.01:.3f} | val_loss={0.82 + e*0.01:.3f} | f1={0.73 - e*0.005:.3f}"
         for e in range(6, 31)]
    )
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────────
# Shared constants
# ────────────────────────────────────────────────────────────────────────────────

TASK_TYPES = [
    "image_classification",
    "nlp_classification",
    "object_detection",
    "tabular",
]

VENUES = ["ICLR", "NeurIPS", "CVPR", "ICML", "AAAI"]
