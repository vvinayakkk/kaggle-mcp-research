"""
Kaggle MCP Research Server
==========================
A full-featured MCP server giving GitHub Copilot access to:
  • Kaggle Competitions, Kernels, Datasets, Models
  • HuggingFace Hub (datasets, models, inference)
  • Academic research (arXiv, Semantic Scholar, Papers With Code)
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

# ════════════════════════════════════════════════════════════════════════════════
# MCP SERVER INSTANCE
# ════════════════════════════════════════════════════════════════════════════════

mcp = FastMCP(
    "Kaggle Research MCP",
    instructions="""
You are an autonomous AI research engineer with full access to Kaggle, HuggingFace,
and academic paper databases (arXiv, Semantic Scholar, Papers With Code).

When a user provides research material (a ZIP file path and/or a text description),
immediately execute the full research pipeline WITHOUT asking clarifying questions:

1. extract_zip_research      — understand the user's existing work
2. full_literature_sweep     — survey SOTA papers + benchmarks
3. get_sota_for_task         — get exact leaderboard numbers
4. search_arxiv / search_semantic_scholar — find the latest methods
5. generate_kaggle_notebook  — create a complete, GPU-ready notebook
6. kaggle_push_and_run       — upload to Kaggle, wait for results
7. kaggle_kernel_output_log  — retrieve ALL output / cell results
8. Summarise findings        — architecture decisions, final metrics, next steps

Never ask the user for clarification; infer and proceed.
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
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
