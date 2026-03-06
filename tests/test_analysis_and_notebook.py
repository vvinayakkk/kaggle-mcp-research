"""
Tests for analysis tools: design_ablation_study, interpret_training_log,
estimate_kaggle_feasibility, suggest_ensemble_strategy,
identify_hard_samples, generate_hypothesis_test_plan.

Also includes notebook edge cases.
"""
import json
import pytest

from kaggle_mcp.tools.analysis import (
    design_ablation_study,
    interpret_training_log,
    estimate_kaggle_feasibility,
    suggest_ensemble_strategy,
    identify_hard_samples,
    generate_hypothesis_test_plan,
)
from kaggle_mcp.tools.notebook import generate_kaggle_notebook


# ════════════════════════════════════════════════════════════════════════════════
# design_ablation_study
# ════════════════════════════════════════════════════════════════════════════════

class TestDesignAblationStudy:
    def test_returns_valid_json(self, strong_arch):
        result = json.loads(design_ablation_study(strong_arch, "image_classification"))
        assert "ablation_components" in result
        assert "paper_table_markdown" in result
        assert "priority_order" in result

    def test_at_least_one_component_essential(self, strong_arch):
        result = json.loads(design_ablation_study(strong_arch, "image_classification"))
        components = result["ablation_components"]
        assert any(c["essential"] for c in components)

    def test_paper_table_is_markdown(self, strong_arch):
        result = json.loads(design_ablation_study(strong_arch))
        table = result["paper_table_markdown"]
        assert "|" in table, "Paper table should be markdown table"
        assert "Component" in table

    def test_minimal_runs_lte_full_runs(self, minimal_arch):
        result = json.loads(design_ablation_study(minimal_arch))
        assert result["minimal_ablation_runs"] <= result["full_ablation_runs"]

    def test_works_with_all_task_types(self, minimal_arch):
        for task in ("image_classification", "object_detection", "tabular", "nlp_classification"):
            result = json.loads(design_ablation_study(minimal_arch, task))
            assert "ablation_components" in result


# ════════════════════════════════════════════════════════════════════════════════
# interpret_training_log
# ════════════════════════════════════════════════════════════════════════════════

class TestInterpretTrainingLog:
    def test_returns_valid_json_healthy(self, healthy_log):
        result = json.loads(interpret_training_log(healthy_log))
        assert "diagnosis" in result
        assert "issues_detected" in result
        assert "recommendations" in result

    def test_detects_overfitting(self, overfitting_log):
        result = json.loads(interpret_training_log(overfitting_log))
        assert "OVERFIT" in result["diagnosis"].upper(), \
            f"Expected overfitting diagnosis, got: {result['diagnosis']}"
        assert len(result["issues_detected"]) > 0

    def test_overfitting_triggers_regularisation_advice(self, overfitting_log):
        result = json.loads(interpret_training_log(overfitting_log))
        recs = " ".join(result["recommendations"]).lower()
        assert any(k in recs for k in ["dropout", "weight_decay", "regularis", "augment"])

    def test_detects_plateau(self, plateau_log):
        result = json.loads(interpret_training_log(plateau_log))
        assert "PLATEAU" in result["diagnosis"].upper() or \
               any("plateau" in i.lower() for i in result["issues_detected"]), \
            "Plateau should be detected"

    def test_detects_early_convergence(self, early_convergence_log):
        result = json.loads(interpret_training_log(early_convergence_log))
        has_early = any("early" in i.lower() or "best" in i.lower()
                        for i in result["issues_detected"])
        assert has_early or "early_stop" in " ".join(result["recommendations"]).lower()

    def test_invalid_log_returns_error(self):
        result = json.loads(interpret_training_log("completely garbage no numbers here xyz"))
        assert "error" in result

    def test_epochs_parsed_count(self, healthy_log):
        result = json.loads(interpret_training_log(healthy_log))
        assert result["epochs_parsed"] >= 20  # fixture has 25 epochs

    def test_best_val_metric_sensible(self, healthy_log):
        result = json.loads(interpret_training_log(healthy_log))
        if result["best_val_metric"] is not None:
            assert 0 <= result["best_val_metric"] <= 1.5


# ════════════════════════════════════════════════════════════════════════════════
# estimate_kaggle_feasibility
# ════════════════════════════════════════════════════════════════════════════════

class TestEstimateKaggleFeasibility:
    def test_returns_valid_json(self, minimal_arch):
        result = json.loads(estimate_kaggle_feasibility(minimal_arch))
        assert "estimated_memory_gb" in result
        assert "estimated_hours" in result
        assert "risk_level" in result
        assert "recommended_env" in result

    def test_memory_positive(self, strong_arch):
        result = json.loads(estimate_kaggle_feasibility(strong_arch))
        assert result["estimated_memory_gb"] > 0

    def test_hours_positive(self, strong_arch):
        result = json.loads(estimate_kaggle_feasibility(
            strong_arch, dataset_info="50000 images", num_epochs=30
        ))
        assert result["estimated_hours"] > 0

    def test_large_model_triggers_high_risk_or_oom(self):
        huge_arch = "ViT-L/16 at 512px, batch_size=64, no fp16, 50 epochs"
        result = json.loads(estimate_kaggle_feasibility(
            huge_arch, dataset_info="100000 images", num_epochs=50,
            batch_size=64, image_size=512
        ))
        # Should flag memory or time issue
        assert result["estimated_memory_gb"] > 5  # ViT-L is massive

    def test_fp16_arch_lower_memory(self):
        fp16_arch  = "efficientnet_v2_s with mixed_precision amp fp16"
        fp32_arch  = "efficientnet_v2_s training"
        r_fp16 = json.loads(estimate_kaggle_feasibility(fp16_arch, batch_size=32))
        r_fp32 = json.loads(estimate_kaggle_feasibility(fp32_arch, batch_size=32))
        # fp16 should use less or equal memory
        assert r_fp16["estimated_memory_gb"] <= r_fp32["estimated_memory_gb"]

    def test_risk_level_is_known_value(self, minimal_arch):
        result = json.loads(estimate_kaggle_feasibility(minimal_arch))
        assert any(level in result["risk_level"] for level in ["LOW", "MEDIUM", "HIGH"])


# ════════════════════════════════════════════════════════════════════════════════
# suggest_ensemble_strategy
# ════════════════════════════════════════════════════════════════════════════════

class TestSuggestEnsembleStrategy:
    def test_returns_valid_json(self):
        result = json.loads(suggest_ensemble_strategy(
            "image_classification",
            ["efficientnet_v2_s", "convnext_base", "swin_transformer_base"]
        ))
        assert "strategy" in result
        assert "diversity_score" in result
        assert "expected_gain" in result
        assert "implementation_hint" in result

    def test_diverse_models_higher_score(self):
        diverse_result = json.loads(suggest_ensemble_strategy(
            "image_classification",
            ["efficientnet_v2_s fp-16 augment", "convnext_base mixup", "vit_b_16 swin deberta"]
        ))
        same_result = json.loads(suggest_ensemble_strategy(
            "image_classification",
            ["resnet50", "resnet50 v2", "resnet50 v3"]
        ))
        assert diverse_result["diversity_score"] >= same_result["diversity_score"]

    @pytest.mark.parametrize("task", ["image_classification", "nlp_classification",
                                       "object_detection", "tabular"])
    def test_all_tasks_have_strategy(self, task):
        result = json.loads(suggest_ensemble_strategy(task, ["model_a", "model_b"]))
        assert "strategy" in result
        assert result["strategy"]["simple"] != ""

    def test_empty_model_list_handled(self):
        result = json.loads(suggest_ensemble_strategy("image_classification", []))
        assert "strategy" in result


# ════════════════════════════════════════════════════════════════════════════════
# identify_hard_samples
# ════════════════════════════════════════════════════════════════════════════════

class TestIdentifyHardSamples:
    def test_returns_valid_json(self):
        result = json.loads(identify_hard_samples("image_classification"))
        assert "known_hard_cases" in result
        assert "targeted_augmentations" in result
        assert "manual_inspection_checklist" in result

    def test_known_hard_cases_non_empty_for_vision(self):
        result = json.loads(identify_hard_samples("image_classification"))
        assert len(result["known_hard_cases"]) > 0

    def test_known_hard_cases_non_empty_for_nlp(self):
        result = json.loads(identify_hard_samples("nlp_classification"))
        assert len(result["known_hard_cases"]) > 0

    def test_class_names_captured(self):
        result = json.loads(identify_hard_samples("image_classification",
                                                   class_names=["cat", "dog", "car"]))
        assert result["classes"] == ["cat", "dog", "car"]

    def test_known_confusion_pairs_detected(self):
        result = json.loads(identify_hard_samples("image_classification",
                                                   class_names=["cat", "dog"]))
        # cat+dog pair is in known confusion list
        patterns = " ".join(result["confusion_matrix_patterns"]).lower()
        assert "cat" in patterns or "confusion" in patterns

    def test_targeted_augs_present_for_all_tasks(self):
        for task in ("image_classification", "object_detection", "nlp_classification", "tabular"):
            result = json.loads(identify_hard_samples(task))
            assert len(result["targeted_augmentations"]) > 0


# ════════════════════════════════════════════════════════════════════════════════
# generate_hypothesis_test_plan
# ════════════════════════════════════════════════════════════════════════════════

class TestGenerateHypothesisTestPlan:
    def test_returns_valid_json(self):
        result = json.loads(generate_hypothesis_test_plan("f1", 0.80, 0.83, 1000))
        assert "effect_size" in result
        assert "recommended_tests" in result
        assert "statistically_detectable" in result
        assert "bootstrap_recipe" in result

    def test_large_improvement_effect_size_large(self):
        result = json.loads(generate_hypothesis_test_plan("f1", 0.50, 0.90, 5000))
        assert result["effect_size"] in ("large", "medium")

    def test_tiny_improvement_effect_negligible(self):
        result = json.loads(generate_hypothesis_test_plan("f1", 0.800, 0.801, 100))
        assert result["effect_size"] in ("negligible", "small")

    def test_large_n_detectable(self):
        result = json.loads(generate_hypothesis_test_plan("f1", 0.80, 0.85, 10000))
        assert result["statistically_detectable"] is True

    def test_small_n_may_not_be_detectable(self):
        result = json.loads(generate_hypothesis_test_plan("f1", 0.80, 0.81, 30))
        # Very small sample vs small improvement — might not be detectable
        # Just verify it returns a boolean
        assert isinstance(result["statistically_detectable"], bool)

    def test_recommended_tests_non_empty(self):
        result = json.loads(generate_hypothesis_test_plan("accuracy", 0.75, 0.78, 500))
        assert len(result["recommended_tests"]) > 0


# ════════════════════════════════════════════════════════════════════════════════
# Notebook edge cases
# ════════════════════════════════════════════════════════════════════════════════

class TestNotebookEdgeCases:
    """Edge-case tests for generate_kaggle_notebook."""

    BASE_KWARGS = dict(
        task_description="6-class scene classification",
        dataset_info="5000 images across 6 classes",
        architecture_description="EfficientNetV2-S fine-tuned, MixUp, AdamW",
        competition_slug="test-comp",
        use_gpu=True,
        num_epochs=30,
        batch_size=32,
        image_size=224,
    )

    def test_all_task_types_return_valid_json(self):
        task_types = [
            "image_classification", "nlp_classification",
            "tabular", "object_detection", "general",
        ]
        for tt in task_types:
            nb_str = generate_kaggle_notebook(task_type=tt, **self.BASE_KWARGS)
            nb = json.loads(nb_str)
            assert "cells" in nb, f"Task type '{tt}' returned invalid notebook"
            assert "metadata" in nb

    def test_notebook_has_code_cells(self):
        nb = json.loads(generate_kaggle_notebook(
            task_type="image_classification", **self.BASE_KWARGS
        ))
        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        assert len(code_cells) >= 3

    def test_notebook_has_no_walrus_operator_bug(self):
        """Regression: ensure {VAR := value} pattern is absent from f-strings."""
        nb_str = generate_kaggle_notebook(
            task_type="image_classification", **self.BASE_KWARGS
        )
        assert ":=" not in nb_str, "Walrus operator in f-string would break at runtime"

    def test_competition_slug_in_notebook(self):
        nb_str = generate_kaggle_notebook(
            task_type="image_classification",
            competition_slug="my-competition-slug",
            **{k: v for k, v in self.BASE_KWARGS.items() if k != "competition_slug"}
        )
        assert "my-competition-slug" in nb_str

    def test_large_batch_size_accepted(self):
        kw = {k: v for k, v in self.BASE_KWARGS.items() if k != "batch_size"}
        nb_str = generate_kaggle_notebook(
            task_type="image_classification", batch_size=128, **kw
        )
        assert json.loads(nb_str)  # valid JSON

    def test_small_image_size_accepted(self):
        kw = {k: v for k, v in self.BASE_KWARGS.items() if k != "image_size"}
        nb_str = generate_kaggle_notebook(
            task_type="image_classification", image_size=64, **kw
        )
        assert json.loads(nb_str)

    def test_large_image_size_accepted(self):
        kw = {k: v for k, v in self.BASE_KWARGS.items() if k != "image_size"}
        nb_str = generate_kaggle_notebook(
            task_type="image_classification", image_size=512, **kw
        )
        assert json.loads(nb_str)

    def test_extra_notes_included(self):
        custom_note = "USE_SPECIAL_TOKEN_XYZ_99999"
        nb_str = generate_kaggle_notebook(
            task_type="image_classification",
            extra_notes=custom_note,
            **self.BASE_KWARGS
        )
        assert custom_note in nb_str

    def test_no_gpu_flag_still_generates(self):
        kw = {k: v for k, v in self.BASE_KWARGS.items() if k != "use_gpu"}
        nb_str = generate_kaggle_notebook(
            task_type="tabular", use_gpu=False, **kw
        )
        assert json.loads(nb_str)

    def test_nbformat_version_set(self):
        nb = json.loads(generate_kaggle_notebook(
            task_type="image_classification", **self.BASE_KWARGS
        ))
        assert "nbformat" in nb
