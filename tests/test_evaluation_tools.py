"""
Tests for evaluation tools: brutal_evaluate, reiterate_architecture,
roast_approach, reviewer_perspective, paper_worthiness, q1_journal_analysis.
"""
import json
import pytest

from kaggle_mcp.tools.evaluation import (
    brutal_evaluate,
    reiterate_architecture,
    roast_approach,
    reviewer_perspective,
    paper_worthiness,
    q1_journal_analysis,
)
from tests.conftest import VENUES


# ════════════════════════════════════════════════════════════════════════════════
# brutal_evaluate
# ════════════════════════════════════════════════════════════════════════════════

class TestBrutalEvaluate:
    def test_returns_valid_json(self, minimal_arch):
        result = json.loads(brutal_evaluate(minimal_arch, "image_classification"))
        assert "FATAL_FLAWS" in result
        assert "SERIOUS_CONCERNS" in result
        assert "MINOR_ISSUES" in result
        assert "OVERALL_SCORE" in result
        assert "VERDICT" in result

    def test_detects_missing_augmentation(self, minimal_arch):
        result = json.loads(brutal_evaluate(minimal_arch, "image_classification"))
        all_issues = result["FATAL_FLAWS"] + result["SERIOUS_CONCERNS"]
        assert any("augment" in i.lower() or "augmentation" in i.lower()
                   for i in all_issues), "Missing augmentation not flagged"

    def test_detects_missing_scheduler(self, minimal_arch):
        result = json.loads(brutal_evaluate(minimal_arch, "image_classification"))
        all_issues = result["SERIOUS_CONCERNS"] + result["FATAL_FLAWS"] + result["MINOR_ISSUES"]
        assert any("scheduler" in i.lower() or "lr" in i.lower() for i in all_issues)

    def test_detects_overfitting_risk_large_model_small_data(self, arch_with_overfitting_risk):
        result = json.loads(brutal_evaluate(
            arch_with_overfitting_risk,
            task_type="image_classification",
            dataset_info="500 images of 6 scene classes"
        ))
        assert any("overfit" in f.lower() for f in result["FATAL_FLAWS"]), \
            "Overfitting risk on ViT-L + 500 images should be FATAL"

    def test_strong_arch_scores_higher(self, minimal_arch, strong_arch):
        weak_score  = json.loads(brutal_evaluate(minimal_arch,  "image_classification"))["OVERALL_SCORE"]
        strong_score = json.loads(brutal_evaluate(strong_arch, "image_classification"))["OVERALL_SCORE"]
        assert strong_score > weak_score, "Strong arch should score better than minimal arch"

    def test_overall_score_in_range(self, minimal_arch):
        result = json.loads(brutal_evaluate(minimal_arch, "image_classification"))
        assert 0 <= result["OVERALL_SCORE"] <= 10

    def test_verdict_is_string(self, minimal_arch):
        result = json.loads(brutal_evaluate(minimal_arch, "image_classification"))
        assert isinstance(result["VERDICT"], str)
        assert len(result["VERDICT"]) > 0

    def test_all_task_types(self, minimal_arch):
        for task in ("image_classification", "nlp_classification", "object_detection", "tabular"):
            result = json.loads(brutal_evaluate(minimal_arch, task))
            assert "OVERALL_SCORE" in result

    def test_practice_audit_present(self, minimal_arch):
        result = json.loads(brutal_evaluate(minimal_arch, "image_classification"))
        assert "PRACTICE_AUDIT" in result
        audit = result["PRACTICE_AUDIT"]
        assert "augmentation" in audit
        assert "scheduler" in audit


# ════════════════════════════════════════════════════════════════════════════════
# reiterate_architecture
# ════════════════════════════════════════════════════════════════════════════════

class TestReterateArchitecture:
    def test_returns_valid_json(self, minimal_arch):
        result = json.loads(reiterate_architecture(minimal_arch, "image_classification", iteration_num=1))
        assert "iteration" in result
        assert "changes_applied" in result
        assert "improved_description" in result
        assert "expected_delta" in result

    def test_iteration_1_targets_backbone_augmentation(self, minimal_arch):
        result = json.loads(reiterate_architecture(minimal_arch, "image_classification", iteration_num=1))
        types = [c["type"] for c in result["changes_applied"]]
        assert any("AUGMENT" in t or "BACKBONE" in t for t in types), \
            "Iteration 1 should target backbone or augmentation"

    def test_iteration_2_targets_training_recipe(self, minimal_arch):
        result = json.loads(reiterate_architecture(minimal_arch, "image_classification", iteration_num=2))
        types = [c["type"] for c in result["changes_applied"]]
        assert len(types) > 0, "Iteration 2 should apply at least one change"

    def test_iteration_3_marks_ready_for_notebook(self, minimal_arch):
        result = json.loads(reiterate_architecture(minimal_arch, "image_classification", iteration_num=3))
        assert result["ready_for_notebook"] is True

    def test_improved_description_longer_than_original(self, minimal_arch):
        result = json.loads(reiterate_architecture(minimal_arch, iteration_num=1))
        assert len(result["improved_description"]) >= len(result["original_description"])

    def test_iteration_number_preserved(self, minimal_arch):
        for i in (1, 2, 3):
            result = json.loads(reiterate_architecture(minimal_arch, iteration_num=i))
            assert result["iteration"] == i


# ════════════════════════════════════════════════════════════════════════════════
# roast_approach
# ════════════════════════════════════════════════════════════════════════════════

class TestRoastApproach:
    def test_returns_valid_json(self, minimal_arch):
        result = json.loads(roast_approach(minimal_arch, "image_classification"))
        assert "punchline" in result
        assert "brutal_observations" in result
        assert "technical_debt_list" in result
        assert "redemption_arc" in result
        assert "roast_score_10" in result

    def test_vgg_triggers_roast(self, vgg_arch):
        result = json.loads(roast_approach(vgg_arch, "image_classification"))
        all_obs = " ".join(result["brutal_observations"])
        assert "vgg" in all_obs.lower() or len(result["technical_debt_list"]) > 0, \
            "VGG should trigger at least one roast observation"

    def test_resnet50_roast_mentions_upgrade(self, minimal_arch):
        result = json.loads(roast_approach(minimal_arch, "image_classification"))
        all_text = result["punchline"] + " ".join(result["brutal_observations"])
        assert "resnet50" in all_text.lower() or len(result["technical_debt_list"]) > 0

    def test_score_in_range(self, minimal_arch):
        result = json.loads(roast_approach(minimal_arch, "image_classification"))
        assert 0 <= result["roast_score_10"] <= 10

    def test_strong_arch_scores_higher_than_weak(self, minimal_arch, strong_arch):
        weak  = json.loads(roast_approach(minimal_arch, "image_classification"))["roast_score_10"]
        strong = json.loads(roast_approach(strong_arch, "image_classification"))["roast_score_10"]
        assert strong >= weak

    def test_winner_strategies_present(self, minimal_arch):
        result = json.loads(roast_approach(minimal_arch, "image_classification"))
        assert "what_a_kaggle_grandmaster_does_instead" in result
        assert len(result["what_a_kaggle_grandmaster_does_instead"]) > 0

    def test_is_medal_worthy_boolean(self, minimal_arch):
        result = json.loads(roast_approach(minimal_arch, "image_classification"))
        assert isinstance(result["is_kaggle_medal_worthy"], bool)


# ════════════════════════════════════════════════════════════════════════════════
# reviewer_perspective
# ════════════════════════════════════════════════════════════════════════════════

class TestReviewerPerspective:
    def test_returns_valid_json(self, minimal_arch):
        result = json.loads(reviewer_perspective(minimal_arch, target_venue="ICLR"))
        assert "scores" in result
        assert "verdict" in result
        assert "recommendation" in result
        assert "required_changes" in result

    @pytest.mark.parametrize("venue", ["ICLR", "NeurIPS", "CVPR", "ICML", "AAAI"])
    def test_all_venues_return_scores(self, minimal_arch, venue):
        result = json.loads(reviewer_perspective(minimal_arch, target_venue=venue))
        assert "scores" in result
        assert len(result["scores"]) > 0

    def test_venue_acceptance_rate_present(self, minimal_arch):
        result = json.loads(reviewer_perspective(minimal_arch, target_venue="CVPR"))
        assert "venue_acceptance_rate" in result

    def test_weaknesses_filled_for_minimal_arch(self, minimal_arch):
        result = json.loads(reviewer_perspective(minimal_arch, target_venue="ICLR"))
        assert len(result["weaknesses"]) > 0

    def test_verdict_is_one_of_known_values(self, minimal_arch):
        result = json.loads(reviewer_perspective(minimal_arch, target_venue="NeurIPS"))
        valid = {"ACCEPT", "WEAK ACCEPT", "BORDERLINE", "REJECT"}
        assert result["verdict"] in valid, f"Unknown verdict: {result['verdict']}"

    def test_unknown_venue_falls_back(self, minimal_arch):
        # Should not crash on unknown venue — falls back to ICLR
        result = json.loads(reviewer_perspective(minimal_arch, target_venue="UNKNOWN_CONF"))
        assert "scores" in result


# ════════════════════════════════════════════════════════════════════════════════
# paper_worthiness
# ════════════════════════════════════════════════════════════════════════════════

class TestPaperWorthiness:
    def test_returns_valid_json(self, minimal_arch, weak_results):
        result = json.loads(paper_worthiness(minimal_arch, weak_results))
        assert "readiness_score" in result
        assert "readiness_pct" in result
        assert "missing_for_submission" in result
        assert "acceptance_probability" in result

    def test_missing_critical_flagged_without_results(self, minimal_arch):
        result = json.loads(paper_worthiness(minimal_arch, results_summary=""))
        assert any("CRITICAL" in m for m in result["missing_for_submission"])

    def test_code_available_increases_score(self, minimal_arch, weak_results):
        no_code  = json.loads(paper_worthiness(minimal_arch, weak_results, code_available=False))
        yes_code = json.loads(paper_worthiness(minimal_arch, weak_results, code_available=True))
        score_a  = int(no_code["readiness_score"].split("/")[0])
        score_b  = int(yes_code["readiness_score"].split("/")[0])
        assert score_b > score_a

    def test_more_datasets_better_score(self, minimal_arch, weak_results):
        one = json.loads(paper_worthiness(minimal_arch, weak_results, num_datasets=1))
        two = json.loads(paper_worthiness(minimal_arch, weak_results, num_datasets=2))
        assert int(two["readiness_score"].split("/")[0]) >= int(one["readiness_score"].split("/")[0])

    def test_priority_actions_are_subset_of_missing(self, minimal_arch, weak_results):
        result = json.loads(paper_worthiness(minimal_arch, weak_results))
        for action in result["priority_actions"]:
            assert action in result["missing_for_submission"]


# ════════════════════════════════════════════════════════════════════════════════
# q1_journal_analysis
# ════════════════════════════════════════════════════════════════════════════════

class TestQ1JournalAnalysis:
    def test_returns_valid_json(self, minimal_arch, weak_results):
        result = json.loads(q1_journal_analysis(minimal_arch, weak_results))
        assert "recommended_journals" in result
        assert "required_experiments" in result
        assert "top_recommendation" in result

    @pytest.mark.parametrize("domain", ["computer_vision", "nlp", "general_ml"])
    def test_all_domains_return_journals(self, minimal_arch, weak_results, domain):
        result = json.loads(q1_journal_analysis(minimal_arch, weak_results, domain=domain))
        assert len(result["recommended_journals"]) > 0

    def test_journals_have_impact_factor(self, minimal_arch, strong_results):
        result = json.loads(q1_journal_analysis(minimal_arch, strong_results))
        for j in result["recommended_journals"]:
            assert "impact_factor" in j
            assert j["impact_factor"] > 0

    def test_theory_increases_readiness(self, minimal_arch, strong_results):
        no_theory   = json.loads(q1_journal_analysis(minimal_arch, strong_results, has_theory=False))
        yes_theory  = json.loads(q1_journal_analysis(minimal_arch, strong_results, has_theory=True))
        assert int(yes_theory["submission_readiness"].split("/")[0]) >= \
               int(no_theory["submission_readiness"].split("/")[0])

    def test_more_datasets_required_when_few(self, minimal_arch, weak_results):
        result = json.loads(q1_journal_analysis(minimal_arch, weak_results, num_datasets=1))
        assert any("dataset" in r.lower() for r in result["required_experiments"])
