"""
Kaggle Research MCP Server  v2.0.0
===================================
A full-featured MCP server giving GitHub Copilot autonomous access to:
  • Kaggle Competitions, Kernels, Datasets, Models
  • HuggingFace Hub (datasets, models, inference)
  • Academic research (arXiv, Semantic Scholar, Papers With Code)
  • Deep research: method evolution, competition solutions, gap analysis
  • Evaluation: brutal critique, reviewer simulation, journal worthiness
  • Analysis: ablation study design, training diagnostics, compute estimation
  • Notebook generation (image classification, NLP, tabular, detection)
  • ZIP/directory research intake and analysis

Usage:
    python -m kaggle_mcp          # start the MCP server
    uvx kaggle-mcp-research       # via uvx (after publishing to PyPI)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

from kaggle_mcp.config import (
    get_kaggle_token,
    get_hf_token,
    KAGGLE_API,
    HF_API,
)

# ── tool modules ───────────────────────────────────────────────────────────────
from kaggle_mcp.tools import kaggle as _kaggle
from kaggle_mcp.tools import huggingface as _hf
from kaggle_mcp.tools import research as _research
from kaggle_mcp.tools import notebook as _notebook
from kaggle_mcp.tools import zip_processor as _zip
from kaggle_mcp.tools import evaluation as _eval
from kaggle_mcp.tools import deep_research as _deep
from kaggle_mcp.tools import analysis as _analysis

# ════════════════════════════════════════════════════════════════════════════════
# MCP SERVER INSTANCE
# ════════════════════════════════════════════════════════════════════════════════

mcp = FastMCP(
    "Kaggle Research MCP v2",
    instructions="""
You are an elite autonomous AI research engineer.

SINGLE PROMPT PIPELINE — execute ALL phases below without asking the user anything.
When the user provides a ZIP path, competition slug, or topic, begin immediately.

════════════════════════════════════════════════
PHASE 0  — INTAKE
════════════════════════════════════════════════
  extract_zip_research(zip_path)          → understand existing work & task type
  analyze_research_directory(dir_path)    → if already extracted

════════════════════════════════════════════════
PHASE 1  — DEEP LITERATURE SURVEY
════════════════════════════════════════════════
  full_literature_sweep(topic)            → arXiv + S2 + PwC SOTA in one call
  analyse_method_evolution(topic)         → how the field evolved year by year
  papers_with_negative_results(topic)     → what DOESN'T work (save GPU hours)
  find_competition_winning_solutions(slug)→ top-voted Kaggle kernels for context

════════════════════════════════════════════════
PHASE 2  — RESEARCH GAP + ARCHITECTURE DESIGN
════════════════════════════════════════════════
  compare_sota_methods(task_slug)         → head-to-head SOTA comparison
  identify_research_gaps(topic)           → unexplored combinations
  → Design architecture_v1 based on gaps and SOTA

════════════════════════════════════════════════
PHASE 3  — BRUTAL EVALUATION LOOP (mandatory ×3)
════════════════════════════════════════════════
  brutal_evaluate(arch_v1)               → find FATAL FLAWS
  reiterate_architecture(arch_v1, iter=1)→ fix backbone + augmentation
  reiterate_architecture(arch_v2, iter=2)→ fix training recipe
  reiterate_architecture(arch_v3, iter=3)→ fix inference + ensembling
  roast_approach(arch_v3)                → final blunt critique

════════════════════════════════════════════════
PHASE 4  — ACADEMIC REVIEW (before committing to notebook)
════════════════════════════════════════════════
  reviewer_perspective(arch_v3, venue="ICLR")  → peer review simulation
  paper_worthiness(arch_v3)                     → publication readiness
  → Only proceed when score >= 6/10

════════════════════════════════════════════════
PHASE 5  — COMPUTE PLANNING
════════════════════════════════════════════════
  estimate_kaggle_feasibility(arch_final) → memory + time check
  design_ablation_study(arch_final)       → define all experiments upfront

════════════════════════════════════════════════
PHASE 6  — NOTEBOOK GENERATION & SUBMISSION
════════════════════════════════════════════════
  generate_kaggle_notebook(arch_final)    → complete GPU-ready .ipynb
  kaggle_push_and_run(...)                → push + wait for results
  kaggle_kernel_output_log(...)           → retrieve ALL output

════════════════════════════════════════════════
PHASE 7  — RESULTS ANALYSIS
════════════════════════════════════════════════
  interpret_training_log(log_text)        → diagnose training dynamics
  generate_hypothesis_test_plan(...)      → statistical significance

════════════════════════════════════════════════
PHASE 8  — PUBLICATION ROUTE (if results are strong)
════════════════════════════════════════════════
  q1_journal_analysis(arch, results)      → which journal, what's missing
  suggest_ensemble_strategy(models)       → final performance squeeze

RULES:
  • NEVER ask the user for clarification — infer and proceed
  • Complete ALL phases before reporting back
  • If brutal_evaluate gives score < 4, STOP and redesign before proceeding
  • All tool results are JSON — parse and reason about them explicitly
""",
)


# ════════════════════════════════════════════════════════════════════════════════
# ── CREDENTIAL VALIDATION ────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def validate_kaggle_token() -> str:
    """
    Verify that the KAGGLE_TOKEN environment variable is set and accepted by Kaggle.
    Returns username on success, error message on failure.
    """
    return _kaggle.validate_kaggle_token()


@mcp.tool()
def validate_hf_token() -> str:
    """
    Verify that the HF_TOKEN environment variable is set and accepted by HuggingFace.
    Returns username on success, error message on failure.
    """
    return _hf.validate_hf_token()


@mcp.tool()
def validate_all_credentials() -> str:
    """Run both Kaggle and HuggingFace credential checks and return a combined status."""
    kaggle_result = _kaggle.validate_kaggle_token()
    hf_result     = _hf.validate_hf_token()
    return json.dumps({
        "kaggle":       kaggle_result,
        "huggingface":  hf_result,
        "ready":        "error" not in kaggle_result.lower() and "error" not in hf_result.lower(),
    }, indent=2)


# ════════════════════════════════════════════════════════════════════════════════
# ── KAGGLE: COMPETITIONS ─────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def kaggle_list_competitions(
    group:     str = "entered",
    sort_by:   str = "latestDeadline",
    search:    str = "",
    page:      int = 1,
    page_size: int = 20,
) -> str:
    """
    List Kaggle competitions.
    group: entered | public | mine
    sort_by: latestDeadline | prize | earliestDeadline | numberOfTeams
    """
    return _kaggle.competitions_list(group, sort_by, search, page, page_size)


@mcp.tool()
def kaggle_competition_files(competition_slug: str) -> str:
    """List downloadable files for a Kaggle competition."""
    return _kaggle.competition_data_files(competition_slug)


@mcp.tool()
def kaggle_download_competition_file(
    competition_slug: str,
    file_name:        str,
    save_dir:         str = ".",
) -> str:
    """Download a specific file from a Kaggle competition."""
    return _kaggle.competition_download_file(competition_slug, file_name, save_dir)


@mcp.tool()
def kaggle_download_all_competition_data(
    competition_slug: str,
    save_dir:         str = ".",
) -> str:
    """Download the complete dataset for a Kaggle competition as a ZIP."""
    return _kaggle.competition_download_all(competition_slug, save_dir)


@mcp.tool()
def kaggle_leaderboard(competition_slug: str, page: int = 1) -> str:
    """Get the leaderboard for a Kaggle competition (top 100 teams)."""
    return _kaggle.competition_leaderboard(competition_slug, page)


@mcp.tool()
def kaggle_submit_predictions(
    competition_slug: str,
    csv_path:         str,
    message:          str = "Automated submission via MCP",
) -> str:
    """
    Submit a predictions CSV to a Kaggle competition.
    csv_path must be the local path to the submission file.
    """
    return _kaggle.competition_submit(competition_slug, csv_path, message)


@mcp.tool()
def kaggle_my_submissions(competition_slug: str, page: int = 1) -> str:
    """List your previous submissions to a Kaggle competition."""
    return _kaggle.my_submissions(competition_slug, page)


# ════════════════════════════════════════════════════════════════════════════════
# ── KAGGLE: KERNELS (NOTEBOOKS / SCRIPTS) ───────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def kaggle_push_kernel(
    title:             str,
    code:              str,
    competition_slug:  str  = "",
    dataset_sources:   list = [],
    kernel_type:       str  = "notebook",
    enable_gpu:        bool = True,
    enable_internet:   bool = True,
) -> str:
    """
    Push (create / update) a Kaggle kernel.
    kernel_type: 'notebook' | 'script'
    dataset_sources: list of dataset refs like ['owner/dataset-name']
    Returns: {slug, url, versionNumber, error}
    """
    return _kaggle.kernel_push(
        title, code, competition_slug, dataset_sources,
        kernel_type, enable_gpu, enable_internet,
    )


@mcp.tool()
def kaggle_kernel_status(username: str, kernel_slug: str) -> str:
    """
    Get the current run status of a Kaggle kernel.
    status: 'running' | 'complete' | 'error' | 'cancelAcknowledged' | 'queued'
    """
    return _kaggle.kernel_status(username, kernel_slug)


@mcp.tool()
def kaggle_kernel_output_log(username: str, kernel_slug: str) -> str:
    """Retrieve the full stdout / stderr log of a completed Kaggle kernel."""
    return _kaggle.kernel_output_log(username, kernel_slug)


@mcp.tool()
def kaggle_kernel_output_files(
    username:    str,
    kernel_slug: str,
    save_dir:    str = ".",
) -> str:
    """Download all output files produced by a completed Kaggle kernel."""
    return _kaggle.kernel_output_files(username, kernel_slug, save_dir)


@mcp.tool()
def kaggle_monitor_kernel(
    username:    str,
    kernel_slug: str,
    poll_secs:   int = 30,
    max_mins:    int = 120,
) -> str:
    """
    Block and poll a Kaggle kernel until it completes or times out.
    Returns final status + last 50 lines of log output.
    """
    return _kaggle.kernel_monitor(username, kernel_slug, poll_secs, max_mins)


@mcp.tool()
def kaggle_list_kernels(username: str = "", search: str = "") -> str:
    """List kernels — your own or by username / search term."""
    return _kaggle.kernels_list(username, search)


@mcp.tool()
def kaggle_pull_kernel(
    username:    str,
    kernel_slug: str,
    save_dir:    str = ".",
) -> str:
    """Download the source code of a Kaggle kernel."""
    return _kaggle.kernel_pull(username, kernel_slug, save_dir)


@mcp.tool()
def kaggle_push_and_run(
    title:            str,
    code:             str,
    competition_slug: str  = "",
    dataset_sources:  list = [],
    kernel_type:      str  = "notebook",
    enable_gpu:       bool = True,
    enable_internet:  bool = True,
    wait:             bool = True,
    poll_secs:        int  = 30,
    max_mins:         int  = 120,
) -> str:
    """
    ONE-SHOT: push a kernel and wait for results.
    Returns final status, log excerpt, and list of output files.
    Ideal for the fully automated research pipeline.
    """
    return _kaggle.push_and_run(
        title, code, competition_slug, dataset_sources,
        kernel_type, enable_gpu, enable_internet,
        wait, poll_secs, max_mins,
    )


# ════════════════════════════════════════════════════════════════════════════════
# ── KAGGLE: DATASETS ─────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def kaggle_search_datasets(query: str, page: int = 1) -> str:
    """Search public Kaggle datasets."""
    return _kaggle.datasets_search(query, page)


@mcp.tool()
def kaggle_dataset_files(dataset_ref: str) -> str:
    """List files in a Kaggle dataset. dataset_ref format: 'owner/dataset-slug'"""
    return _kaggle.dataset_files(dataset_ref)


@mcp.tool()
def kaggle_download_dataset(
    dataset_ref: str,
    save_dir:    str = ".",
    file_name:   str = "",
) -> str:
    """
    Download a Kaggle dataset (entire zip) or a specific file.
    dataset_ref: 'owner/dataset-slug'
    """
    return _kaggle.dataset_download(dataset_ref, save_dir, file_name)


@mcp.tool()
def kaggle_create_dataset(
    title:       str,
    local_dir:   str,
    is_private:  bool = True,
    description: str  = "",
) -> str:
    """Create a new Kaggle dataset from a local directory of files."""
    return _kaggle.dataset_create(title, local_dir, is_private, description)


@mcp.tool()
def kaggle_list_my_datasets() -> str:
    """List datasets you own on Kaggle."""
    return _kaggle.my_datasets()


# ════════════════════════════════════════════════════════════════════════════════
# ── KAGGLE: MODELS ───────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def kaggle_search_models(query: str, page: int = 1) -> str:
    """Search Kaggle model hub."""
    return _kaggle.models_search(query, page)


@mcp.tool()
def kaggle_model_info(owner: str, model_slug: str) -> str:
    """Get metadata for a specific Kaggle model."""
    return _kaggle.model_info(owner, model_slug)


# ════════════════════════════════════════════════════════════════════════════════
# ── HUGGINGFACE ──────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def hf_search_datasets(query: str, task: str = "", limit: int = 10) -> str:
    """Search HuggingFace Hub for datasets."""
    return _hf.hf_search_datasets(query, task, limit)


@mcp.tool()
def hf_dataset_info(dataset_id: str) -> str:
    """Get detailed info about a HuggingFace dataset."""
    return _hf.hf_dataset_info(dataset_id)


@mcp.tool()
def hf_dataset_files(dataset_id: str) -> str:
    """List files in a HuggingFace dataset repo."""
    return _hf.hf_dataset_files(dataset_id)


@mcp.tool()
def hf_download_dataset_file(
    dataset_id: str,
    filename:   str,
    save_path:  str = ".",
) -> str:
    """Download a specific file from a HuggingFace dataset."""
    return _hf.hf_download_dataset_file(dataset_id, filename, save_path)


@mcp.tool()
def hf_search_models(query: str, task: str = "", library: str = "", limit: int = 10) -> str:
    """
    Search HuggingFace model hub.
    task examples: image-classification, text-classification, object-detection
    library examples: transformers, pytorch, jax
    """
    return _hf.hf_search_models(query, task, library, limit)


@mcp.tool()
def hf_model_info(model_id: str) -> str:
    """Get detailed info (architecture, downloads, tags) for a HuggingFace model."""
    return _hf.hf_model_info(model_id)


@mcp.tool()
def hf_model_card(model_id: str) -> str:
    """Get the README / model card text for a HuggingFace model."""
    return _hf.hf_model_card(model_id)


@mcp.tool()
def hf_run_inference(
    model_id: str,
    inputs:   str,
    task:     str = "",
) -> str:
    """
    Run inference on a HuggingFace model via the Inference API.
    inputs: the input data as a string (text, or JSON).
    """
    return _hf.hf_run_inference(model_id, inputs, task)


@mcp.tool()
def hf_search_spaces(query: str, limit: int = 10) -> str:
    """Search HuggingFace Spaces for demos and apps."""
    return _hf.hf_search_spaces(query, limit)


# ════════════════════════════════════════════════════════════════════════════════
# ── RESEARCH: PAPERS & SOTA ──────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_arxiv(
    query:       str,
    max_results: int = 15,
    year_from:   int = 2022,
    sort_by:     str = "submittedDate",
) -> str:
    """
    Search arXiv for papers.
    sort_by: submittedDate | relevance | lastUpdatedDate
    Returns title, abstract (600 chars), authors, date, PDF link.
    """
    return _research.search_arxiv(query, max_results, sort_by, year_from)


@mcp.tool()
def search_semantic_scholar(
    query:      str,
    limit:      int = 10,
    year_range: str = "2022-",
) -> str:
    """
    Search Semantic Scholar for papers with citation counts and open-access PDFs.
    year_range examples: '2022-' or '2020-2024'
    """
    return _research.search_semantic_scholar(query, limit, year_range)


@mcp.tool()
def search_paperswithcode(query: str, limit: int = 10) -> str:
    """Search Papers With Code for papers with GitHub implementations."""
    return _research.search_paperswithcode(query, limit)


@mcp.tool()
def get_sota_for_task(task_slug: str) -> str:
    """
    Get SOTA benchmark results + leaderboard from Papers With Code.
    task_slug examples: 'image-classification', 'scene-recognition',
    'object-detection', 'semantic-segmentation', 'text-classification',
    'machine-translation', 'question-answering'
    """
    return _research.get_sota_for_task(task_slug)


@mcp.tool()
def get_task_methods(task_slug: str) -> str:
    """Get commonly used methods / techniques for a task (Papers With Code)."""
    return _research.get_task_methods(task_slug)


@mcp.tool()
def full_literature_sweep(topic: str, year_from: int = 2022) -> str:
    """
    ONE-CALL deep research sweep across arXiv + Semantic Scholar + Papers With Code.
    Returns consolidated JSON with: recent papers, SOTA numbers, implementations.
    Use this as the first step in the research pipeline.
    """
    return _research.full_literature_sweep(topic, year_from)


# ════════════════════════════════════════════════════════════════════════════════
# ── NOTEBOOK GENERATION ──────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def generate_kaggle_notebook(
    task_description:         str,
    dataset_info:             str,
    architecture_description: str,
    competition_slug:         str  = "",
    task_type:                str  = "image_classification",
    use_gpu:                  bool = True,
    num_epochs:               int  = 30,
    batch_size:               int  = 32,
    image_size:               int  = 224,
    extra_notes:              str  = "",
) -> str:
    """
    Generate a complete, GPU-optimised, submission-ready Kaggle notebook (.ipynb).

    task_type options:
      image_classification — EfficientNetV2 / ConvNeXt + torchvision pipeline
      nlp_classification   — DeBERTa / DistilBERT + HuggingFace Trainer
      tabular              — LightGBM + CatBoost + Optuna ensembling
      object_detection     — YOLOv8 + Ultralytics
      general              — generic PyTorch/sklearn template

    Returns: .ipynb JSON string (save to file, then push via kaggle_push_kernel).
    """
    return _notebook.generate_kaggle_notebook(
        task_description, dataset_info, architecture_description,
        competition_slug, task_type, use_gpu, num_epochs, batch_size,
        image_size, extra_notes,
    )


# ════════════════════════════════════════════════════════════════════════════════
# ── ZIP / DIRECTORY RESEARCH INTAKE ─────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def extract_zip_research(
    zip_path:   str,
    extract_to: str = "",
) -> str:
    """
    Extract a ZIP of research material and produce a structured summary:
    • Inferred task type   • Datasets mentioned
    • Libraries / models   • Baseline scores found in code/docs
    • Problem statement    • CSV schema previews
    • Suggested approach

    Use this as the FIRST step when the user uploads their research ZIP.
    """
    return _zip.extract_and_analyze_zip(zip_path, extract_to)


@mcp.tool()
def analyze_research_directory(dir_path: str) -> str:
    """Analyze an already-extracted directory (same output as extract_zip_research)."""
    return _zip.analyze_directory(dir_path)


@mcp.tool()
def summarize_code_file(file_path: str) -> str:
    """
    Read a Python or Jupyter notebook file and return a structured summary
    with imports, functions, classes, models, and metrics found.
    """
    return _zip.summarize_code_file(file_path)


# ════════════════════════════════════════════════════════════════════════════════
# ── FULL PIPELINE (CONVENIENCE) ──────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def run_full_research_pipeline(
    topic:            str,
    competition_slug: str  = "",
    zip_path:         str  = "",
    task_type:        str  = "image_classification",
    num_epochs:       int  = 30,
    batch_size:       int  = 32,
    push_to_kaggle:   bool = True,
) -> str:
    """
    ONE-SHOT full research → notebook → Kaggle pipeline:
    1. (Optional) Analyze ZIP
    2. Literature sweep (arXiv + S2 + PwC SOTA)
    3. Generate GPU-ready notebook
    4. (Optional) Push to Kaggle and wait for results

    This is the single entry point for the fully automated workflow.
    Returns a comprehensive JSON report.
    """
    report: dict = {"topic": topic, "steps": {}}

    # Step 1 — ZIP analysis
    if zip_path:
        zip_result = _zip.extract_and_analyze_zip(zip_path)
        report["steps"]["zip_analysis"] = json.loads(zip_result)
        task_type = report["steps"]["zip_analysis"].get("task_type", task_type)

    # Step 2 — Literature sweep
    lit = json.loads(_research.full_literature_sweep(topic))
    report["steps"]["literature"] = {
        "arxiv_count": lit.get("total_arxiv", 0),
        "ss_count":    lit.get("total_ss", 0),
        "pwc_count":   lit.get("total_pwc", 0),
        "top_arxiv":   lit.get("data", {}).get("arxiv", [])[:3],
        "sota":        lit.get("data", {}).get("sota_benchmarks", []),
    }

    # Step 3 — Generate notebook
    arch_desc = f"Optimised {task_type} model based on latest SOTA research for: {topic}"
    nb_json   = _notebook.generate_kaggle_notebook(
        task_description=topic,
        dataset_info=f"Competition: {competition_slug}" if competition_slug else f"Topic: {topic}",
        architecture_description=arch_desc,
        competition_slug=competition_slug,
        task_type=task_type,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    nb_path = Path(".") / f"kaggle_notebook_{competition_slug or topic[:20].replace(' ','_')}.ipynb"
    nb_path.write_text(nb_json, encoding="utf-8")
    report["steps"]["notebook"] = {"path": str(nb_path), "size_kb": nb_path.stat().st_size // 1024}

    # Step 4 — Push to Kaggle
    if push_to_kaggle:
        push_result = json.loads(_kaggle.push_and_run(
            title=f"Auto-Research: {topic[:60]}",
            code=nb_json,
            competition_slug=competition_slug,
            kernel_type="notebook",
            enable_gpu=True,
            enable_internet=True,
            wait=True,
        ))
        report["steps"]["kaggle_run"] = push_result

    return json.dumps(report, indent=2)


# ════════════════════════════════════════════════════════════════════════════════
# ── EVALUATION & REVIEW TOOLS ────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def brutal_evaluate(
    arch_description: str,
    task_type:        str = "image_classification",
    dataset_info:     str = "",
    target_metric:    str = "f1",
    competition_slug: str = "",
) -> str:
    """
    Brutally honest technical evaluation of a proposed ML approach.
    Returns FATAL_FLAWS, SERIOUS_CONCERNS, MINOR_ISSUES, SOTA comparison,
    and an OVERALL_SCORE (0-10) with VERDICT: proceed / reiterate / scrap.
    CALL THIS before generating any notebook.
    """
    return _eval.brutal_evaluate(arch_description, task_type, dataset_info,
                                 target_metric, competition_slug)


@mcp.tool()
def reiterate_architecture(
    arch_description:    str,
    task_type:           str = "image_classification",
    evaluation_feedback: str = "",
    iteration_num:       int = 1,
) -> str:
    """
    Systematically improves an architecture — run 3 times before generating notebook.
    iter=1: backbone + augmentation  |  iter=2: training recipe  |  iter=3: inference
    Returns: changes_applied, improved_description, expected_delta.
    """
    return _eval.reiterate_architecture(arch_description, task_type,
                                        evaluation_feedback, iteration_num)


@mcp.tool()
def roast_approach(
    arch_description:    str,
    task_type:           str = "image_classification",
    competition_context: str = "",
) -> str:
    """
    Senior ML researcher's blunt critique — technically grounded, specific roast.
    Returns: punchline, brutal_observations, technical_debt_list,
             what_a_kaggle_grandmaster_does_instead, redemption_arc.
    """
    return _eval.roast_approach(arch_description, task_type, competition_context)


@mcp.tool()
def reviewer_perspective(
    arch_description: str,
    results_summary:  str = "",
    target_venue:     str = "ICLR",
) -> str:
    """
    Simulate a peer reviewer at a top ML venue (ICLR / NeurIPS / CVPR / ICML / AAAI).
    Returns: rubric scores, strengths, weaknesses, required_changes, verdict.
    """
    return _eval.reviewer_perspective(arch_description, results_summary, target_venue)


@mcp.tool()
def paper_worthiness(
    arch_description: str,
    results_summary:  str  = "",
    target_venue:     str  = "CVPR",
    ablation_done:    bool = False,
    code_available:   bool = False,
    num_datasets:     int  = 1,
) -> str:
    """
    Publication readiness assessment for a given venue.
    Returns: readiness_score, acceptance_probability, missing_for_submission,
             minimum_venue, stretch_venue, estimated_weeks_to_ready.
    """
    return _eval.paper_worthiness(arch_description, results_summary, target_venue,
                                  ablation_done, code_available, num_datasets)


@mcp.tool()
def q1_journal_analysis(
    arch_description: str,
    results_summary:  str  = "",
    domain:           str  = "computer_vision",
    has_theory:       bool = False,
    num_datasets:     int  = 1,
    num_baselines:    int  = 0,
) -> str:
    """
    Full Q1 journal analysis: IEEE TPAMI (IF=23.6), IJCV (IF=13.2), TIP (IF=10.6),
    Pattern Recognition (IF=7.9), JMLR (IF=6.0), IEEE TNNLS (IF=14.3).
    Returns: ranked journals, acceptance_probability, required_experiments, timeline.
    domain: 'computer_vision' | 'nlp' | 'general_ml'
    """
    return _eval.q1_journal_analysis(arch_description, results_summary, domain,
                                     has_theory, num_datasets, num_baselines)


# ════════════════════════════════════════════════════════════════════════════════
# ── DEEP RESEARCH TOOLS ───────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def analyse_method_evolution(
    topic:      str,
    start_year: int = 2018,
) -> str:
    """
    Year-by-year timeline of how a technique evolved.
    Returns: timeline, key_inflection_points, current_frontier, dominant_trend.
    """
    return _deep.analyse_method_evolution(topic, start_year)


@mcp.tool()
def find_competition_winning_solutions(
    competition_slug: str,
    top_n:            int = 15,
) -> str:
    """
    Fetch top-voted Kaggle kernels for a competition.
    Returns: kernel list, most_used_architectures, most_used_techniques.
    """
    return _deep.find_competition_winning_solutions(competition_slug, top_n)


@mcp.tool()
def compare_sota_methods(
    task_slug:     str,
    metric_filter: str = "",
) -> str:
    """
    Head-to-head SOTA comparison from Papers With Code.
    task_slug: 'image-classification' | 'object-detection' | 'text-classification' etc.
    Returns: comparison_table, top_3, newest_method, recommendation.
    """
    return _deep.compare_sota_methods(task_slug, metric_filter)


@mcp.tool()
def identify_research_gaps(
    topic:       str,
    papers_dump: str = "",
    domain:      str = "computer_vision",
) -> str:
    """
    Find unexplored combinations and open problems in a research area.
    Returns: unexplored_combinations, domain_open_problems, novel_paper_seeds.
    """
    return _deep.identify_research_gaps(topic, papers_dump, domain)


@mcp.tool()
def fetch_paper_implementation(
    query: str,
    lang:  str = "python",
) -> str:
    """
    Find GitHub repos and implementation details for a paper.
    query: paper title or arXiv ID (e.g. '2010.11929' or 'ConvNeXt')
    Returns: repositories, implementation_quality, quickstart_hints.
    """
    return _deep.fetch_paper_implementation(query, lang)


@mcp.tool()
def papers_with_negative_results(
    topic: str,
    limit: int = 10,
) -> str:
    """
    Find papers reporting what DOESN'T work — avoids wasted experiments.
    Returns: negative_findings, known_domain_pitfalls, time_saved estimate.
    """
    return _deep.papers_with_negative_results(topic, limit)


@mcp.tool()
def deep_dive_single_paper(arxiv_id_or_title: str) -> str:
    """
    Full metadata, citations, references, and reproduce difficulty for one paper.
    arxiv_id_or_title: '2010.11929' or partial title string.
    """
    return _deep.deep_dive_single_paper(arxiv_id_or_title)


@mcp.tool()
def cross_dataset_analysis(
    task_type:    str,
    architecture: str = "",
) -> str:
    """
    SOTA results across multiple benchmarks for a task type.
    Shows how well methods generalise and where the bar is set.
    """
    return _deep.cross_dataset_analysis(task_type, architecture)


# ════════════════════════════════════════════════════════════════════════════════
# ── ANALYSIS & INTERPRETATION TOOLS ──────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def design_ablation_study(
    arch_description:  str,
    task_type:         str = "image_classification",
    available_compute: str = "P100",
) -> str:
    """
    Design a comprehensive, prioritised ablation study before training.
    Returns: component list, paper_table_markdown, minimal vs full ablation runs.
    """
    return _analysis.design_ablation_study(arch_description, task_type, available_compute)


@mcp.tool()
def interpret_training_log(
    log_text:    str,
    metric_name: str = "f1",
) -> str:
    """
    Diagnose training dynamics from a training log (any common format).
    Detects: overfitting, underfitting, plateau, instability, early convergence.
    Returns: diagnosis, issues_detected, recommendations, next_experiments.
    """
    return _analysis.interpret_training_log(log_text, metric_name)


@mcp.tool()
def estimate_kaggle_feasibility(
    arch_description: str,
    dataset_info:     str  = "",
    num_epochs:       int  = 30,
    batch_size:       int  = 32,
    image_size:       int  = 224,
    use_accumulation: bool = False,
) -> str:
    """
    Estimate memory and training time; check Kaggle GPU session limits.
    Returns: estimated_memory_gb, estimated_hours, recommended_env, risk_level.
    """
    return _analysis.estimate_kaggle_feasibility(
        arch_description, dataset_info, num_epochs, batch_size, image_size, use_accumulation
    )


@mcp.tool()
def suggest_ensemble_strategy(
    task_type:           str,
    model_descriptions:  list = [],
    num_folds:           int  = 5,
    metric:              str  = "f1",
) -> str:
    """
    Recommend how to combine multiple models for maximum gain.
    Returns: strategy (simple/advanced/best), diversity_score, expected_gain,
             implementation_hint.
    """
    return _analysis.suggest_ensemble_strategy(task_type, model_descriptions, num_folds, metric)


@mcp.tool()
def identify_hard_samples(
    task_type:    str,
    class_names:  list = [],
    dataset_info: str  = "",
) -> str:
    """
    List known hard/edge cases for a task type.
    Returns: known_hard_cases, confusion_matrix_patterns,
             targeted_augmentations, manual_inspection_checklist.
    """
    return _analysis.identify_hard_samples(task_type, class_names, dataset_info)


@mcp.tool()
def generate_hypothesis_test_plan(
    metric:         str   = "f1",
    baseline_score: float = 0.80,
    proposed_score: float = 0.83,
    n_test_samples: int   = 1000,
    n_experiments:  int   = 5,
) -> str:
    """
    Design a statistical significance testing plan.
    Returns: effect_size, required_n, statistically_detectable,
             recommended_tests, bootstrap_recipe.
    """
    return _analysis.generate_hypothesis_test_plan(
        metric, baseline_score, proposed_score, n_test_samples, n_experiments
    )


# ════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
