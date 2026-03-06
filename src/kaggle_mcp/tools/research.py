"""
Research tools — ArXiv, Semantic Scholar, Papers With Code, web search.
Provides real-time literature search to support novel architecture design.
"""
from __future__ import annotations
import json
import re
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.parse import quote_plus

import requests

from kaggle_mcp.config import ARXIV_API, SEMANTIC_SCHOLAR_API, PAPERS_WITH_CODE_API


# ═══════════════════════════════════════════════════════════════════════════════
# ARXIV
# ═══════════════════════════════════════════════════════════════════════════════

def search_arxiv(
    query: str,
    max_results: int = 15,
    sort_by: str = "submittedDate",
    year_from: int = 2020,
) -> str:
    """
    Search arXiv for papers. Returns titles, abstracts, authors, dates.
    sort_by: submittedDate | relevance | lastUpdatedDate
    """
    params = {
        "search_query": f"all:{query}",
        "start":        0,
        "max_results":  max_results,
        "sortBy":       sort_by,
        "sortOrder":    "descending",
    }
    r = requests.get(ARXIV_API, params=params, timeout=30)
    if r.status_code != 200:
        return f"ArXiv ERROR {r.status_code}"

    ns = "http://www.w3.org/2005/Atom"
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return f"ArXiv parse error: {r.text[:500]}"

    papers = []
    for entry in root.findall(f"{{{ns}}}entry"):
        def t(tag):
            el = entry.find(f"{{{ns}}}{tag}")
            return el.text.strip() if el is not None and el.text else ""

        published = t("published")[:10]  # YYYY-MM-DD
        year = int(published[:4]) if published else 0
        if year_from and year < year_from:
            continue

        authors = [a.find(f"{{{ns}}}name").text for a in entry.findall(f"{{{ns}}}author")
                   if a.find(f"{{{ns}}}name") is not None]
        arxiv_id = t("id").split("/abs/")[-1]
        papers.append({
            "id":        arxiv_id,
            "title":     t("title").replace("\n", " "),
            "abstract":  t("summary")[:600].replace("\n", " "),
            "authors":   authors[:4],
            "published": published,
            "url":       f"https://arxiv.org/abs/{arxiv_id}",
            "pdf":       f"https://arxiv.org/pdf/{arxiv_id}",
        })

    return json.dumps({"query": query, "count": len(papers), "papers": papers}, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC SCHOLAR
# ═══════════════════════════════════════════════════════════════════════════════

def search_semantic_scholar(
    query: str,
    limit: int = 10,
    year_range: str = "2020-",
    fields: str = "title,abstract,year,citationCount,authors,externalIds,openAccessPdf",
) -> str:
    """
    Search Semantic Scholar for papers. Returns citation counts, PDFs.
    year_range: e.g. '2022-' or '2020-2024'
    """
    params = {
        "query":       query,
        "limit":       limit,
        "fields":      fields,
        "year":        year_range,
    }
    r = requests.get(f"{SEMANTIC_SCHOLAR_API}/paper/search", params=params, timeout=30)
    if r.status_code != 200:
        return f"Semantic Scholar ERROR {r.status_code}: {r.text[:300]}"

    data = r.json()
    papers = []
    for p in data.get("data", []):
        pdf_url = None
        oap = p.get("openAccessPdf")
        if oap and isinstance(oap, dict):
            pdf_url = oap.get("url")
        arxiv_id = p.get("externalIds", {}).get("ArXiv")
        authors = [a.get("name", "") for a in (p.get("authors") or [])[:4]]
        papers.append({
            "paperId":      p.get("paperId", ""),
            "title":        p.get("title", ""),
            "abstract":     (p.get("abstract") or "")[:500],
            "year":         p.get("year"),
            "citations":    p.get("citationCount", 0),
            "authors":      authors,
            "arxiv_id":     arxiv_id,
            "arxiv_url":    f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
            "pdf_url":      pdf_url,
            "s2_url":       f"https://www.semanticscholar.org/paper/{p.get('paperId','')}",
        })

    return json.dumps({"query": query, "total": data.get("total", 0), "papers": papers}, indent=2)


def get_paper_citations(paper_id: str, limit: int = 10) -> str:
    """Get papers that cite a given Semantic Scholar paper ID."""
    r = requests.get(
        f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}/citations",
        params={"limit": limit, "fields": "title,year,citationCount"},
        timeout=20,
    )
    if r.status_code != 200:
        return f"ERROR {r.status_code}: {r.text[:300]}"
    return json.dumps(r.json(), indent=2)


def get_paper_references(paper_id: str, limit: int = 20) -> str:
    """Get papers referenced by a given paper."""
    r = requests.get(
        f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}/references",
        params={"limit": limit, "fields": "title,year,citationCount"},
        timeout=20,
    )
    if r.status_code != 200:
        return f"ERROR {r.status_code}: {r.text[:300]}"
    return json.dumps(r.json(), indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# PAPERS WITH CODE — SOTA BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════

def search_paperswithcode(query: str, items_per_page: int = 10) -> str:
    """Search Papers With Code for papers with GitHub implementations."""
    r = requests.get(
        f"{PAPERS_WITH_CODE_API}/papers/",
        params={"q": query, "items_per_page": items_per_page, "ordering": "-published"},
        timeout=20,
    )
    if r.status_code != 200:
        if r.status_code == 422:
            return f"PapersWithCode search error (check query): {r.text[:300]}"
        return f"PapersWithCode ERROR {r.status_code}: {r.text[:300]}"
    data = r.json()
    out = []
    for p in data.get("results", []):
        repos = [{"url": rp.get("url", ""), "stars": rp.get("stars", 0)}
                 for rp in (p.get("repositories") or [])[:2]]
        out.append({
            "title":    p.get("title", ""),
            "abstract": (p.get("abstract") or "")[:400],
            "date":     p.get("published", ""),
            "arxiv":    p.get("arxiv_id", ""),
            "url":      p.get("url_pdf", ""),
            "repos":    repos,
        })
    return json.dumps({"query": query, "count": len(out), "papers": out}, indent=2)


def get_sota_for_task(task_slug: str) -> str:
    """
    Get SOTA benchmark results for a task from Papers With Code.
    task_slug examples: 'image-classification', 'scene-recognition',
    'object-detection', 'image-segmentation', 'text-classification'
    """
    r = requests.get(f"{PAPERS_WITH_CODE_API}/sota/", params={"task": task_slug}, timeout=20)
    if r.status_code != 200:
        return f"PwC SOTA ERROR {r.status_code}: {r.text[:300]}"
    data = r.json()
    benchmarks = []
    for b in data.get("results", [])[:5]:
        rows = []
        for row in (b.get("sota", {}).get("rows") or [])[:10]:
            rows.append({
                "rank":   row.get("rank"),
                "model":  row.get("model_name", ""),
                "paper":  row.get("paper_title", ""),
                "metric": row.get("evaluated_on", ""),
                "score":  row.get("metrics", {}).get(row.get("best_metric", ""), ""),
            })
        benchmarks.append({
            "benchmark": b.get("task", {}).get("task_name", ""),
            "dataset":   b.get("dataset", {}).get("name", ""),
            "top10":     rows,
        })
    return json.dumps({"task": task_slug, "benchmarks": benchmarks}, indent=2)


def get_task_methods(task_slug: str) -> str:
    """Get methods/techniques commonly used for a task (Papers With Code)."""
    r = requests.get(f"{PAPERS_WITH_CODE_API}/methods/", params={"task": task_slug, "items_per_page": 15}, timeout=20)
    if r.status_code != 200:
        return f"ERROR {r.status_code}: {r.text[:200]}"
    data = r.json()
    out = [{
        "name":        m.get("name", ""),
        "description": (m.get("description") or "")[:300],
        "paper_title": m.get("paper", {}).get("title", "") if isinstance(m.get("paper"), dict) else "",
    } for m in data.get("results", [])[:10]]
    return json.dumps(out, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED RESEARCH SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def full_literature_sweep(topic: str, year_from: int = 2022) -> str:
    """
    One-call deep sweep across ArXiv + Semantic Scholar + Papers With Code.
    Returns consolidated JSON with baseline numbers, SOTA, and recent papers.
    """
    results = {}

    # ArXiv — most recent papers
    try:
        arxiv_data = json.loads(search_arxiv(topic, max_results=10, year_from=year_from))
        results["arxiv"] = arxiv_data.get("papers", [])
    except Exception as e:
        results["arxiv"] = [{"error": str(e)}]

    # Semantic Scholar — highly cited papers
    try:
        ss_data = json.loads(search_semantic_scholar(topic, limit=8, year_range=f"{year_from}-"))
        results["semantic_scholar"] = ss_data.get("papers", [])
    except Exception as e:
        results["semantic_scholar"] = [{"error": str(e)}]

    # Papers With Code — implementations + SOTA
    try:
        pwc_data = json.loads(search_paperswithcode(topic, items_per_page=8))
        results["paperswithcode"] = pwc_data.get("papers", [])
    except Exception as e:
        results["paperswithcode"] = [{"error": str(e)}]

    # SOTA benchmarks — infer task slug from topic
    task_slug = _infer_task_slug(topic)
    if task_slug:
        try:
            sota_data = json.loads(get_sota_for_task(task_slug))
            results["sota_benchmarks"] = sota_data.get("benchmarks", [])
            results["inferred_task"] = task_slug
        except Exception as e:
            results["sota_benchmarks"] = [{"error": str(e)}]

    summary = {
        "topic":             topic,
        "year_from":         year_from,
        "total_arxiv":       len(results.get("arxiv", [])),
        "total_ss":          len(results.get("semantic_scholar", [])),
        "total_pwc":         len(results.get("paperswithcode", [])),
        "has_sota":          bool(results.get("sota_benchmarks")),
        "data":              results,
    }
    return json.dumps(summary, indent=2)


def _infer_task_slug(topic: str) -> str:
    """Map a free-form research topic to a Papers With Code task slug."""
    t = topic.lower()
    mapping = {
        ("image classification", "scene classification", "scene recognition"): "scene-recognition",
        ("object detection",):                                                  "object-detection",
        ("image segmentation", "semantic segmentation", "instance segmentation"): "semantic-segmentation",
        ("natural language processing", "text classification", "sentiment"):    "text-classification",
        ("machine translation", "translation"):                                 "machine-translation",
        ("image generation", "generative", "diffusion"):                        "image-generation",
        ("question answering", "reading comprehension"):                        "question-answering",
        ("named entity recognition", "ner"):                                    "named-entity-recognition",
        ("image captioning", "captioning"):                                     "image-captioning",
        ("depth estimation", "monocular depth"):                                "monocular-depth-estimation",
        ("pose estimation", "keypoint"):                                        "pose-estimation",
        ("speech recognition", "asr"):                                          "speech-recognition",
        ("medical image", "pathology", "radiology"):                            "medical-image-classification",
        ("tabular", "structured data"):                                         "tabular-classification",
    }
    for keys, slug in mapping.items():
        if any(k in t for k in keys):
            return slug
    if "classif" in t:
        return "image-classification"
    if "detect" in t:
        return "object-detection"
    return ""
