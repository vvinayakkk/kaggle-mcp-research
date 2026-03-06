"""
Tests for deep research tools: analyse_method_evolution,
find_competition_winning_solutions, compare_sota_methods,
identify_research_gaps, fetch_paper_implementation,
papers_with_negative_results, deep_dive_single_paper,
cross_dataset_analysis.
"""
import json
import pytest

from kaggle_mcp.tools.deep_research import (
    analyse_method_evolution,
    find_competition_winning_solutions,
    compare_sota_methods,
    identify_research_gaps,
    fetch_paper_implementation,
    papers_with_negative_results,
    deep_dive_single_paper,
    cross_dataset_analysis,
)


# ════════════════════════════════════════════════════════════════════════════════
# analyse_method_evolution
# ════════════════════════════════════════════════════════════════════════════════

class TestAnalyseMethodEvolution:
    def test_returns_valid_json(self):
        result = json.loads(analyse_method_evolution("image classification", start_year=2021))
        assert "topic" in result
        assert "timeline" in result
        assert "suggested_next_step" in result

    def test_topic_preserved(self):
        result = json.loads(analyse_method_evolution("vision transformer", start_year=2020))
        assert result["topic"] == "vision transformer"

    def test_years_covered_includes_start(self):
        result = json.loads(analyse_method_evolution("object detection", start_year=2020))
        assert "2020" in result["years_covered"]

    def test_timeline_is_dict(self):
        result = json.loads(analyse_method_evolution("convnext", start_year=2021))
        assert isinstance(result["timeline"], dict)

    def test_handles_obscure_topic_gracefully(self):
        """Should not crash on a niche/obscure topic."""
        result = json.loads(analyse_method_evolution(
            "adversarial robustness on medical imaging xray", start_year=2022
        ))
        assert "topic" in result


# ════════════════════════════════════════════════════════════════════════════════
# find_competition_winning_solutions
# ════════════════════════════════════════════════════════════════════════════════

class TestFindCompetitionWinningSolutions:
    def test_returns_valid_json(self):
        result = json.loads(find_competition_winning_solutions("titanic"))
        assert "competition" in result
        assert "top_kernels" in result

    def test_competition_slug_preserved(self):
        result = json.loads(find_competition_winning_solutions("some-competition"))
        assert result["competition"] == "some-competition"

    def test_handles_no_results_gracefully(self):
        """Non-existent competition should not crash — returns empty list."""
        result = json.loads(find_competition_winning_solutions("xyzzzz-nonexistent-99"))
        assert "top_kernels" in result
        # May be empty but should be a list
        assert isinstance(result["top_kernels"], list)

    def test_suggested_combinations_present(self):
        result = json.loads(find_competition_winning_solutions("titanic"))
        assert "suggested_combinations" in result


# ════════════════════════════════════════════════════════════════════════════════
# compare_sota_methods
# ════════════════════════════════════════════════════════════════════════════════

class TestCompareSotaMethods:
    def test_returns_valid_json_image_classification(self):
        result = json.loads(compare_sota_methods("image-classification"))
        assert "task" in result
        assert "interpretation" in result or "note" in result

    def test_returns_valid_json_text_classification(self):
        result = json.loads(compare_sota_methods("text-classification"))
        assert "task" in result

    def test_does_not_crash_on_unknown_task(self):
        result = json.loads(compare_sota_methods("unknown-bogus-task-xyz"))
        assert "task" in result

    def test_metric_filter_applied(self):
        result = json.loads(compare_sota_methods("image-classification", metric_filter="top-1"))
        assert "task" in result


# ════════════════════════════════════════════════════════════════════════════════
# identify_research_gaps
# ════════════════════════════════════════════════════════════════════════════════

class TestIdentifyResearchGaps:
    def test_returns_valid_json(self):
        result = json.loads(identify_research_gaps("scene classification"))
        assert "topic" in result
        assert "unexplored_combinations" in result
        assert "domain_open_problems" in result
        assert "recommendation" in result

    def test_domain_open_problems_non_empty(self):
        result = json.loads(identify_research_gaps("image segmentation", domain="computer_vision"))
        assert len(result["domain_open_problems"]) > 0

    @pytest.mark.parametrize("domain", ["computer_vision", "nlp", "tabular"])
    def test_all_domains_return_problems(self, domain):
        result = json.loads(identify_research_gaps("deep learning", domain=domain))
        assert len(result["domain_open_problems"]) > 0

    def test_papers_analysed_count_non_negative(self):
        result = json.loads(identify_research_gaps("contrastive learning"))
        assert result["papers_analysed"] >= 0


# ════════════════════════════════════════════════════════════════════════════════
# fetch_paper_implementation
# ════════════════════════════════════════════════════════════════════════════════

class TestFetchPaperImplementation:
    def test_returns_valid_json(self):
        result = json.loads(fetch_paper_implementation("ConvNeXt"))
        assert "query" in result
        assert "implementation_quality" in result
        assert "quickstart_hints" in result

    def test_arxiv_id_query(self):
        """arXiv ID format should work."""
        result = json.loads(fetch_paper_implementation("2201.03545"))
        assert "query" in result

    def test_unknown_paper_returns_gracefully(self):
        result = json.loads(fetch_paper_implementation("completely-made-up-paper-title-xyz123"))
        assert "implementation_quality" in result


# ════════════════════════════════════════════════════════════════════════════════
# papers_with_negative_results
# ════════════════════════════════════════════════════════════════════════════════

class TestPapersWithNegativeResults:
    def test_returns_valid_json(self):
        result = json.loads(papers_with_negative_results("image classification"))
        assert "topic" in result
        assert "negative_papers_found" in result
        assert "known_domain_pitfalls" in result
        assert "time_saved" in result

    def test_returns_domain_pitfalls_for_classification(self):
        result = json.loads(papers_with_negative_results("image classification"))
        assert len(result["known_domain_pitfalls"]) > 0

    def test_returns_domain_pitfalls_for_nlp(self):
        result = json.loads(papers_with_negative_results("text classification nlp"))
        # nlp domain pitfalls should be populated
        assert isinstance(result["known_domain_pitfalls"], list)

    def test_papers_found_non_negative(self):
        result = json.loads(papers_with_negative_results("object detection"))
        assert result["negative_papers_found"] >= 0

    def test_limit_respected(self):
        result = json.loads(papers_with_negative_results("image classification", limit=3))
        assert len(result["negative_findings"]) <= 3


# ════════════════════════════════════════════════════════════════════════════════
# deep_dive_single_paper
# ════════════════════════════════════════════════════════════════════════════════

class TestDeepDiveSinglePaper:
    def test_returns_valid_json_with_arxiv_id(self):
        result = json.loads(deep_dive_single_paper("2010.11929"))  # ViT paper
        assert "paper" in result
        assert "reproduce_difficulty" in result

    def test_returns_valid_json_with_title_query(self):
        result = json.loads(deep_dive_single_paper("EfficientNet"))
        assert "paper" in result

    def test_reproduce_difficulty_one_of_known(self):
        result = json.loads(deep_dive_single_paper("2010.11929"))
        assert any(level in result["reproduce_difficulty"]
                   for level in ["HARD", "MEDIUM", "EASY", "UNKNOWN"])

    def test_citing_papers_is_list(self):
        result = json.loads(deep_dive_single_paper("2010.11929"))
        assert isinstance(result["top_citing_papers"], list)


# ════════════════════════════════════════════════════════════════════════════════
# cross_dataset_analysis
# ════════════════════════════════════════════════════════════════════════════════

class TestCrossDatasetAnalysis:
    def test_returns_valid_json(self):
        result = json.loads(cross_dataset_analysis("image_classification"))
        assert "task_type" in result
        assert "datasets_checked" in result
        assert "transfer_advice" in result

    def test_datasets_listed_for_classification(self):
        result = json.loads(cross_dataset_analysis("image_classification"))
        assert len(result["datasets_checked"]) > 0

    def test_handles_unknown_task_gracefully(self):
        result = json.loads(cross_dataset_analysis("completely_unknown_task_type"))
        assert "task_type" in result

    @pytest.mark.parametrize("task", [
        "image_classification", "object_detection",
        "text_classification", "semantic_segmentation",
    ])
    def test_all_common_tasks(self, task):
        result = json.loads(cross_dataset_analysis(task))
        assert "datasets_checked" in result
