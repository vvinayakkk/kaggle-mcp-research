"""
Deep research tools — go beyond surface-level literature search.

Tools:
  analyse_method_evolution       — timeline of how a technique evolved (S2 + arXiv)
  find_competition_winning_solutions — top voted Kaggle kernels for a competition
  compare_sota_methods           — structured head-to-head comparison from Papers With Code
  identify_research_gaps         — what combinations / settings nobody has tried
  fetch_paper_implementation     — find GitHub repos + issue trackers for a paper
  papers_with_negative_results   — what approaches are known NOT to work
  deep_dive_single_paper         — full metadata + citation context for one paper
  cross_dataset_analysis         — how methods transfer across benchmarks
"""
from __future__ import annotations

import json
import re
import urllib.parse
from typing import Optional

import requests

from kaggle_mcp.config import (
    ARXIV_API, SEMANTIC_SCHOLAR_API, PAPERS_WITH_CODE_API,
    KAGGLE_API, kaggle_headers,
)


# ════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _s2_search(query: str, limit: int = 10, year_range: str = "") -> list[dict]:
    """Return papers from Semantic Scholar, sorted by citation count."""
    params: dict = {
        "query":  query,
        "limit":  limit,
        "fields": "title,authors,year,citationCount,externalIds,abstract,url,influentialCitationCount",
    }
    if year_range:
        params["year"] = year_range
    try:
        r = requests.get(f"{SEMANTIC_SCHOLAR_API}/paper/search", params=params, timeout=15)
        if r.status_code == 200:
            return r.json().get("data", [])
    except Exception:
        pass
    return []


def _arxiv_search(query: str, max_results: int = 8, sort_by: str = "submittedDate") -> list[dict]:
    """Return papers from arXiv."""
    q = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "max_results":   max_results,
        "sortBy":        sort_by,
        "sortOrder":     "descending",
    })
    try:
        r = requests.get(f"{ARXIV_API}?{q}", timeout=15)
        if r.status_code != 200:
            return []
        entries = re.findall(r"<entry>(.*?)</entry>", r.text, re.DOTALL)
        papers  = []
        for e in entries:
            title   = re.search(r"<title>(.*?)</title>", e, re.DOTALL)
            summary = re.search(r"<summary>(.*?)</summary>", e, re.DOTALL)
            pub     = re.search(r"<published>(.*?)</published>", e)
            arxiv_id = re.search(r"<id>.*?abs/([\w.]+)</id>", e)
            papers.append({
                "title":    title.group(1).strip().replace("\n", " ") if title else "",
                "abstract": summary.group(1).strip()[:300] if summary else "",
                "year":     pub.group(1)[:4] if pub else "",
                "arxiv_id": arxiv_id.group(1) if arxiv_id else "",
            })
        return papers
    except Exception:
        return []


def _pwc_task_rows(task_slug: str) -> list[dict]:
    """Get SOTA rows for a task from Papers With Code."""
    try:
        r = requests.get(f"{PAPERS_WITH_CODE_API}/sota/",
                         params={"task": task_slug}, timeout=10)
        if r.status_code == 200:
            bms = r.json().get("results", [])
            rows = []
            for bm in bms[:2]:
                for row in bm.get("sota", {}).get("rows", [])[:5]:
                    rows.append({
                        "benchmark":  bm.get("benchmark", ""),
                        "model":      row.get("model_name", ""),
                        "paper_date": row.get("paper_date", ""),
                        "metrics":    row.get("metrics", {}),
                        "uses_extra_data": row.get("uses_extra_data", False),
                    })
            return rows
    except Exception:
        pass
    return []


# ════════════════════════════════════════════════════════════════════════════════
# PUBLIC TOOLS
# ════════════════════════════════════════════════════════════════════════════════

def analyse_method_evolution(
    topic:      str,
    start_year: int = 2018,
) -> str:
    """
    Traces how a technique / architecture evolved over time.

    For each year from start_year to present: key papers, their contribution,
    citation count, and the chain of improvements.

    Returns: { topic, timeline, key_inflection_points, current_frontier,
               dominant_trend, suggested_next_step }
    """
    # Fetch from Semantic Scholar sorted by year
    papers = _s2_search(query=topic, limit=20, year_range=f"{start_year}-")
    # Also grab arxiv for most recent
    recent = _arxiv_search(query=topic, max_results=10, sort_by="submittedDate")

    # Group by year
    by_year: dict[str, list] = {}
    for p in papers:
        yr = str(p.get("year") or "unknown")
        by_year.setdefault(yr, []).append({
            "title":       p.get("title", ""),
            "citations":   p.get("citationCount", 0),
            "influential": p.get("influentialCitationCount", 0),
            "abstract":    (p.get("abstract") or "")[:200],
            "source":      "SemanticScholar",
        })

    for p in recent:
        yr = p.get("year") or "recent"
        by_year.setdefault(yr, []).append({
            "title":    p["title"],
            "arxiv_id": p["arxiv_id"],
            "abstract": p["abstract"],
            "source":   "arXiv",
        })

    # Sort each year bucket by citation count
    timeline = {}
    for yr in sorted(by_year.keys()):
        bucket = sorted(by_year[yr], key=lambda x: x.get("citations", 0), reverse=True)
        timeline[yr] = bucket[:3]

    # Identify inflection points: papers with disproportionately high citations
    all_papers = [p for bucket in by_year.values() for p in bucket]
    citations  = [p.get("citations", 0) for p in all_papers]
    mean_cit   = sum(citations) / max(len(citations), 1)
    inflection = [
        p["title"] for p in all_papers
        if p.get("citations", 0) > mean_cit * 3 and p["title"]
    ][:5]

    # Current frontier = highest avg citations in most recent 2 years
    recent_years  = sorted(timeline.keys())[-2:]
    recent_papers = [p for yr in recent_years for p in timeline.get(yr, [])]
    frontier      = [p["title"] for p in recent_papers if p["title"]][:3]

    return json.dumps({
        "topic":                  topic,
        "years_covered":          f"{start_year}–present",
        "total_papers_found":     len(all_papers),
        "timeline":               timeline,
        "key_inflection_points":  inflection,
        "current_frontier_papers": frontier,
        "dominant_trend":         (
            f"Based on {len(all_papers)} papers, the field is moving toward "
            "larger pretrained models with better fine-tuning recipes and "
            "self-supervised pretraining. Check the 2023-2024 bucket for the latest."
        ),
        "suggested_next_step": (
            "Read the top-cited papers in the most recent year. "
            "Find one technique from 2020-2021 that hasn't been applied to your specific domain yet — "
            "that combination gap is often where novel papers come from."
        ),
    }, indent=2)


def find_competition_winning_solutions(
    competition_slug: str,
    top_n:            int = 15,
) -> str:
    """
    Fetches top-voted Kaggle kernels for a competition.
    These represent the community's best public solutions.

    Returns: { competition, top_kernels, common_patterns, suggested_combinations }
    """
    headers = kaggle_headers()
    results = []

    # Try kernels API with competition filter
    try:
        r = requests.get(
            f"{KAGGLE_API}/kernels/list",
            params={
                "competitionDataSources": competition_slug,
                "orderBy":                "voteCount",
                "pageSize":               top_n,
            },
            headers=headers,
            timeout=15,
        )
        if r.status_code == 200:
            for k in r.json():
                results.append({
                    "title":         k.get("title", ""),
                    "author":        k.get("author", ""),
                    "votes":         k.get("totalVotes", 0),
                    "language":      k.get("language", ""),
                    "url":          f"https://www.kaggle.com/code/{k.get('author','')}/{k.get('slug','')}",
                    "is_notebook":   k.get("isNotebook", True),
                })
    except Exception as e:
        results = [{"error": str(e)}]

    # Pattern analysis from titles
    all_titles  = " ".join(r.get("title", "") for r in results).lower()
    arch_hits = {
        arch: all_titles.count(arch.lower())
        for arch in ["efficientnet", "convnext", "swin", "vit", "resnet", "deberta",
                     "lgbm", "catboost", "xgboost", "yolo", "unet", "timm"]
        if arch.lower() in all_titles
    }
    technique_hits = {
        tech: all_titles.count(tech.lower())
        for tech in ["tta", "ensemble", "mixup", "kfold", "pseudo", "augment",
                     "optuna", "stacking", "blend"]
        if tech.lower() in all_titles
    }

    top_arch     = sorted(arch_hits.items(),      key=lambda x: x[1], reverse=True)[:5]
    top_tech     = sorted(technique_hits.items(), key=lambda x: x[1], reverse=True)[:5]

    return json.dumps({
        "competition":           competition_slug,
        "kernels_found":         len(results),
        "top_kernels":           results[:top_n],
        "most_used_architectures": top_arch,
        "most_used_techniques":    top_tech,
        "suggested_combinations": (
            f"Based on top solutions: combine {top_arch[0][0] if top_arch else 'strong backbone'} "
            f"with {top_tech[0][0] if top_tech else 'TTA/ensemble'}. "
            "The winning pattern is usually: 3-5 diverse models + TTA + ensembling."
        ),
        "_note": (
            "These are PUBLIC kernels. Gold/Silver solutions are often kept private. "
            "Read discussion tab for post-competition write-ups by medal winners."
        ),
    }, indent=2)


def compare_sota_methods(
    task_slug: str,
    metric_filter: str = "",
) -> str:
    """
    Structured head-to-head comparison of SOTA methods from Papers With Code.

    Args:
        task_slug: e.g. 'image-classification', 'object-detection', 'text-classification'
        metric_filter: focus on a specific metric e.g. 'top-1-accuracy', 'f1', 'map'

    Returns: { benchmark, comparison_table, efficiency_analysis, recommendation }
    """
    rows = _pwc_task_rows(task_slug)

    if not rows:
        # Fallback: try searching for SOTA papers directly
        papers = _s2_search(f"state of the art {task_slug}", limit=10)
        return json.dumps({
            "task": task_slug,
            "note": "Papers With Code returned no data. Fell back to paper search.",
            "top_papers": [{"title": p.get("title"), "citations": p.get("citationCount", 0)}
                           for p in papers[:5]],
        }, indent=2)

    # Build comparison table
    table = []
    for i, row in enumerate(rows):
        table.append({
            "rank":           i + 1,
            "model":          row["model"],
            "benchmark":      row["benchmark"],
            "metrics":        row["metrics"],
            "paper_date":     row["paper_date"],
            "uses_extra_data": row["uses_extra_data"],
        })

    # Filter by metric if requested
    if metric_filter and table:
        mc = metric_filter.lower().replace("-", "").replace("_", "")
        for row in table:
            row["target_metric"] = {
                k: v for k, v in row["metrics"].items()
                if mc in k.lower().replace("-", "").replace("_", "")
            }

    # Efficiency estimate (param count from known models)
    from kaggle_mcp.tools.evaluation import _ARCH_PARAMS_M
    for row in table:
        model_l = row["model"].lower().replace("-", "_").replace(" ", "_")
        for arch, params in _ARCH_PARAMS_M.items():
            if arch.replace("_", "") in model_l.replace("_", ""):
                row["params_M"] = params
                break

    top3        = table[:3]
    newest      = max(table, key=lambda x: x.get("paper_date", ""), default={})

    return json.dumps({
        "task":          task_slug,
        "methods_found": len(table),
        "comparison_table": table,
        "top_3":         top3,
        "newest_method": newest,
        "interpretation": (
            f"Top performer: {top3[0]['model'] if top3 else 'N/A'}. "
            f"Newest entry: {newest.get('model', 'N/A')} ({newest.get('paper_date', '')[:7]}). "
            "For competition, prefer the newest method with available code."
        ),
        "recommendation": (
            "Use the newest method published in the last 12 months that has a public implementation. "
            "Older top-1 methods are often impractical (huge compute, proprietary data, no code)."
        ),
    }, indent=2)


def identify_research_gaps(
    topic:       str,
    papers_dump: str = "",
    domain:      str = "computer_vision",
) -> str:
    """
    Analyses literature to find unexplored combinations and open problems.

    Args:
        topic: main research topic
        papers_dump: optional JSON/text of already-gathered papers
        domain: 'computer_vision', 'nlp', 'tabular', etc.

    Returns: { unexplored_combinations, open_problems, adjacent_fields,
               low_hanging_fruit, novel_paper_seeds }
    """
    papers = _s2_search(topic, limit=15)
    recent = _arxiv_search(topic,  max_results=8, sort_by="submittedDate")

    # Gather all abstracts
    all_text = " ".join((p.get("abstract") or "") for p in papers).lower()
    all_text += " " + " ".join(p.get("abstract", "") for p in recent).lower()

    # Look for what's been studied
    studied_datasets = re.findall(
        r"\b(imagenet|cifar|coco|pascal|celeba|cityscapes|voc|lfw|glue|squad|"
        r"ms-coco|openimages|places|sun|caltech|tiny-imagenet)\b", all_text
    )
    studied_archs = re.findall(
        r"\b(resnet|efficientnet|vit|swin|convnext|deit|bert|gpt|vgg|inception|"
        r"detr|yolo|maskrcnn|retinanet|dino|sam|clip)\b", all_text
    )
    studied_techniques = re.findall(
        r"\b(self.supervised|contrastive|knowledge.distil|transfer.learn|"
        r"few.shot|zero.shot|meta.learn|active.learn|semi.supervised)\b", all_text
    )

    # Frequent vs rare combinations
    frequent  = list(set(studied_archs[:10]))
    frequent_datasets = list(set(studied_datasets[:5]))

    # Unexplored = known good techniques not combined with this topic
    all_known_techniques = [
        "self-supervised pretraining", "knowledge distillation", "neural architecture search",
        "continual learning", "federated learning", "interpretability analysis",
        "uncertainty quantification", "multi-task learning", "domain adaptation",
        "data-efficient training",
    ]
    not_studied = [t for t in all_known_techniques if t.split("-")[0] not in all_text[:2000]]

    novel_seeds = []
    if not_studied:
        for t in not_studied[:3]:
            novel_seeds.append(
                f"'{t} for {topic}' — searched {len(papers)} papers, none specifically address this combination."
            )

    domain_gaps = {
        "computer_vision": [
            "Applying foundation models (SAM, DINO) to domain-specific datasets",
            "Efficient inference at edge (< 10ms latency) for real-world deployment",
            "Robustness under distribution shift and adversarial conditions",
            "Few-shot adaptation with only 5-10 labelled examples",
        ],
        "nlp": [
            "Long-context understanding beyond 8K tokens efficiently",
            "Multilingual zero-shot transfer to low-resource languages",
            "Faithful reasoning with verifiable intermediate steps",
            "Efficient fine-tuning with < 1% of parameters (LoRA, BitFit)",
        ],
        "tabular": [
            "Combining graph neural networks with tabular data",
            "Self-supervised pretraining for tabular (TabTransformer, SAINT)",
            "Uncertainty estimation for high-stakes decisions",
            "Feature selection under distribution shift",
        ],
    }

    return json.dumps({
        "topic":                     topic,
        "papers_analysed":           len(papers) + len(recent),
        "studied_architectures":     list(set(studied_archs))[:8],
        "studied_datasets":          list(set(studied_datasets))[:6],
        "studied_techniques":        list(set(studied_techniques))[:6],
        "unexplored_combinations":   not_studied[:5],
        "domain_open_problems":      domain_gaps.get(domain, domain_gaps["computer_vision"]),
        "novel_paper_seeds":         novel_seeds,
        "low_hanging_fruit": (
            f"Combine {frequent[0] if frequent else 'strong backbone'} + "
            f"{not_studied[0] if not_studied else 'better augmentation'} on "
            f"{frequent_datasets[0] if frequent_datasets else 'a new dataset'}. "
            "This combination appears missing from current literature."
        ),
        "recommendation": (
            "The clearest research gap: take the best-performing method and apply it to "
            "a dataset or domain where it hasn't been tested. Show it generalises (or doesn't) "
            "and explain why — that insight IS the contribution."
        ),
    }, indent=2)


def fetch_paper_implementation(
    query:    str,
    lang:     str = "python",
) -> str:
    """
    Finds GitHub repositories and implementation details for a paper.

    Args:
        query: paper title or arXiv ID (e.g. '2010.11929' or 'ConvNeXt A ConvNet')
        lang: preferred implementation language

    Returns: { paper_info, repositories, implementation_quality,
               known_issues, quickstart_hints }
    """
    repos     = []
    paper_info = {}

    # 1. Try Papers With Code API
    try:
        r = requests.get(f"{PAPERS_WITH_CODE_API}/papers/",
                         params={"q": query}, timeout=10)
        if r.status_code == 200:
            data = r.json().get("results", [])
            for p in data[:1]:
                paper_info = {
                    "title":       p.get("title", ""),
                    "arxiv_id":    p.get("arxiv_id", ""),
                    "published":   p.get("published", ""),
                    "pwc_url":     p.get("paper_page", ""),
                    "tasks":      [t.get("task", "") for t in p.get("tasks", [])],
                }
                # Get repos for this paper
                paper_id = p.get("id", "")
                if paper_id:
                    repo_r = requests.get(
                        f"{PAPERS_WITH_CODE_API}/papers/{paper_id}/repositories/",
                        timeout=10
                    )
                    if repo_r.status_code == 200:
                        for repo in repo_r.json().get("results", [])[:5]:
                            repos.append({
                                "url":        repo.get("url", ""),
                                "stars":      repo.get("stars", 0),
                                "framework":  repo.get("framework", ""),
                                "official":   repo.get("is_official", False),
                            })
    except Exception:
        pass

    # 2. Fallback: search Methods API
    if not repos:
        try:
            r2 = requests.get(f"{PAPERS_WITH_CODE_API}/methods/",
                              params={"q": query}, timeout=10)
            if r2.status_code == 200:
                methods = r2.json().get("results", [])
                for m in methods[:3]:
                    paper_info.setdefault("methods_found", []).append(m.get("full_name", ""))
        except Exception:
            pass

    # Implementation quality assessment
    quality = "UNKNOWN"
    if repos:
        top_repo = max(repos, key=lambda x: x.get("stars", 0), default={})
        stars    = top_repo.get("stars", 0)
        official = top_repo.get("official", False)
        quality = (
            "EXCELLENT — official + well-starred" if official and stars > 500
            else "GOOD — official implementation exists" if official
            else f"COMMUNITY — unofficial ({stars} stars)" if stars > 100
            else "FRAGILE — low-star community implementation, may have bugs"
        )

    repos_sorted = sorted(repos, key=lambda x: (x.get("official", False), x.get("stars", 0)), reverse=True)

    return json.dumps({
        "query":                query,
        "paper_info":           paper_info,
        "repositories_found":   len(repos_sorted),
        "repositories":         repos_sorted,
        "implementation_quality": quality,
        "quickstart_hints": [
            f"pip install timm  # likely needed for vision backbones",
            f"Check repo README for pretrained weight download links",
            f"Verify CUDA compatibility — many repos break with PyTorch 2.x+",
            f"Check Issues tab for known bugs before investing time",
        ],
        "fallback_advice": (
            "If no official repo exists: search GitHub for the paper title, "
            "or look for implementations in HuggingFace Hub, timm, torchvision, "
            "or the MMDetection/MMClassification ecosystem."
        ) if not repos else None,
    }, indent=2)


def papers_with_negative_results(
    topic: str,
    limit: int = 10,
) -> str:
    """
    Finds papers reporting what DOESN'T work — extremely valuable for avoiding wasted experiments.

    Searches for: 'do X not work', 'failure', 'challenges', 'pitfalls', 'limitation',
                  'negative results', 'why X fails', ablation showing components don't help.

    Returns: { topic, negative_findings, common_pitfalls, time_saved_estimate }
    """
    negative_queries = [
        f"{topic} failure analysis",
        f"{topic} limitations pitfalls",
        f"{topic} negative results challenges",
        f"why {topic} fails",
        f"{topic} ablation study does not help",
    ]

    all_negative = []
    for q in negative_queries[:2]:  # Limit API calls
        papers = _s2_search(q, limit=8)
        for p in papers:
            title = (p.get("title") or "").lower()
            abstr = (p.get("abstract") or "").lower()
            # Filter for actual negative results
            neg_keywords = ["fail", "limit", "challeng", "pitfall", "negative", "not help",
                           "does not", "cannot", "poor", "worse", "insufficient"]
            if any(k in title or k in abstr[:300] for k in neg_keywords):
                all_negative.append({
                    "title":     p.get("title", ""),
                    "year":      p.get("year", ""),
                    "citations": p.get("citationCount", 0),
                    "abstract":  (p.get("abstract") or "")[:250],
                    "key_finding": "Negative/critical — read carefully",
                })

    # Deduplicate by title
    seen   = set()
    unique = []
    for p in all_negative:
        if p["title"] not in seen:
            seen.add(p["title"])
            unique.append(p)

    unique.sort(key=lambda x: x.get("citations", 0), reverse=True)

    # Known domain-specific pitfalls (hardcoded knowledge)
    domain_pitfalls = {
        "image classification": [
            "Scaling data > scaling model for most downstream tasks (Kolesnikov et al. 2020)",
            "Self-attention in ViTs needs large pretraining — small datasets prefer convolutions (Dosovitskiy et al.)",
            "MixUp hurts performance on fine-grained tasks with subtle inter-class differences",
            "Knowledge distillation from large to small model often doesn't close the gap completely",
        ],
        "object detection": [
            "Anchor-free detectors don't always beat anchor-based on small objects",
            "Larger backbone ≠ better detection — neck/head design matters more at mobile scales",
        ],
        "tabular": [
            "Deep learning rarely beats gradient boosting on tabular data without massive tuning (Grinszztajn et al. 2022)",
            "AutoML often outperforms hand-tuned neural networks on tabular in under 1 hour",
        ],
        "nlp": [
            "Larger LLMs are not always better for domain-specific classification",
            "Fine-tuning on small datasets (<1K samples) often worse than zero-shot large LLM",
        ],
    }

    topic_lower    = topic.lower()
    relevant_pitfalls = []
    for domain, pitfalls in domain_pitfalls.items():
        if any(word in topic_lower for word in domain.split()):
            relevant_pitfalls = pitfalls
            break

    return json.dumps({
        "topic":               topic,
        "negative_papers_found": len(unique),
        "negative_findings":   unique[:limit],
        "known_domain_pitfalls": relevant_pitfalls,
        "time_saved": (
            f"Reading these {len(unique)} negative papers before experimenting "
            "could save 10-40 GPU hours by avoiding dead-end approaches."
        ),
        "researcher_advice": (
            "Negative results are gold. If a paper says X-method-on-Y-task doesn't work, "
            "and your competition IS that task, you have your answer. Don't repeat their mistake."
        ),
    }, indent=2)


def deep_dive_single_paper(
    arxiv_id_or_title: str,
) -> str:
    """
    Full metadata + citation context for a single paper.
    
    Args:
        arxiv_id_or_title: arXiv ID like '2010.11929' or partial title

    Returns: { paper, citation_context, influential_citations, future_work,
               implementation_hint, reproduce_difficulty }
    """
    paper_data = {}
    citations  = []
    references = []

    # Try Semantic Scholar by arXiv ID
    arxiv_pattern = re.search(r"(\d{4}\.\d{4,5})", arxiv_id_or_title)
    if arxiv_pattern:
        arxiv_id = arxiv_pattern.group(1)
        try:
            fields = "title,abstract,year,citationCount,influentialCitationCount,authors,externalIds,tldr,openAccessPdf"
            r = requests.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/arXiv:{arxiv_id}",
                params={"fields": fields},
                timeout=15,
            )
            if r.status_code == 200:
                paper_data = r.json()

            # Citations (who cites this)
            rc = requests.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/arXiv:{arxiv_id}/citations",
                params={"fields": "title,year,citationCount", "limit": 10},
                timeout=10,
            )
            if rc.status_code == 200:
                citations = [c.get("citingPaper", {}) for c in rc.json().get("data", [])]

            # References (what this paper cites)
            rr = requests.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/arXiv:{arxiv_id}/references",
                params={"fields": "title,year,citationCount", "limit": 10},
                timeout=10,
            )
            if rr.status_code == 200:
                references = [c.get("citedPaper", {}) for c in rr.json().get("data", [])]
        except Exception:
            pass
    else:
        # Fallback: search by title
        results = _s2_search(arxiv_id_or_title, limit=1)
        if results:
            paper_data = results[0]

    # Difficulty estimate
    abstract   = (paper_data.get("abstract") or "").lower()
    difficulty = (
        "HARD — theoretical paper with proofs"
        if any(k in abstract for k in ["theorem", "proof", "lemma", "convergence guarantee"])
        else "MEDIUM — methodological with implementation"
        if any(k in abstract for k in ["algorithm", "model", "network", "architecture"])
        else "EASY — empirical study or survey"
    )

    tldr = paper_data.get("tldr") or {}

    return json.dumps({
        "paper": {
            "title":        paper_data.get("title", ""),
            "year":         paper_data.get("year", ""),
            "citations":    paper_data.get("citationCount", 0),
            "influential":  paper_data.get("influentialCitationCount", 0),
            "abstract":     (paper_data.get("abstract") or "")[:400],
            "tldr":         tldr.get("text", "") if isinstance(tldr, dict) else str(tldr),
            "pdf":          (paper_data.get("openAccessPdf") or {}).get("url", "arXiv PDF may be available"),
        },
        "top_citing_papers":    [{"title": c.get("title"), "year": c.get("year"), "citations": c.get("citationCount")}
                                  for c in citations[:5]],
        "key_references":       [{"title": r.get("title"), "year": r.get("year")}
                                  for r in references[:5]],
        "reproduce_difficulty": difficulty,
        "implementation_hint": (
            "Search Papers With Code for the paper title to find official/unofficial implementations. "
            "Look for the link in the abstract or footer of the arXiv PDF."
        ),
    }, indent=2)


def cross_dataset_analysis(
    task_type:     str,
    architecture:  str = "",
) -> str:
    """
    Shows how methods generalise across benchmarks for a task.
    Provides SOTA scores on major datasets so you know where the bar is set.

    Returns: { datasets, sota_per_dataset, generalisation_analysis, transfer_advice }
    """
    task_datasets: dict[str, list] = {
        "image_classification": [
            {"name": "ImageNet-1K", "slug": "image-classification-on-imagenet",  "metric": "Top-1 Acc"},
            {"name": "CIFAR-100",   "slug": "image-classification-on-cifar-100", "metric": "Acc"},
            {"name": "iNaturalist","slug": "fine-grained-image-classification",  "metric": "Top-1 Acc"},
        ],
        "object_detection": [
            {"name": "MS COCO",   "slug": "object-detection-on-coco",           "metric": "AP"},
            {"name": "PASCAL VOC","slug": "object-detection-on-pascal-voc-2012","metric": "mAP"},
        ],
        "text_classification": [
            {"name": "SST-2",     "slug": "sentiment-analysis-on-sst-2",        "metric": "Acc"},
            {"name": "AG News",   "slug": "text-classification-on-ag-news",     "metric": "Acc"},
        ],
        "semantic_segmentation": [
            {"name": "ADE20K",    "slug": "semantic-segmentation-on-ade20k",    "metric": "mIoU"},
            {"name": "Cityscapes","slug": "semantic-segmentation-on-cityscapes","metric": "mIoU"},
        ],
    }

    datasets    = task_datasets.get(task_type, [])
    sota_lookup = {}
    for ds in datasets:
        rows = _pwc_task_rows(ds["slug"])
        if rows:
            sota_lookup[ds["name"]] = {
                "metric":    ds["metric"],
                "best_model": rows[0]["model"],
                "best_score": rows[0]["metrics"],
                "top_3":     [r["model"] for r in rows[:3]],
            }

    return json.dumps({
        "task_type":           task_type,
        "architecture_focus":  architecture,
        "datasets_checked":    [d["name"] for d in datasets],
        "sota_by_dataset":     sota_lookup,
        "generalisation_note": (
            "A model that trains ONLY on ImageNet-1K may not generalise to "
            "domain-specific datasets (medical, satellite, fine-grained). "
            "Pre-training corpus matters more than architecture at this scale."
        ),
        "transfer_advice": (
            f"For your task ({task_type}): "
            "Use ImageNet-21K or CLIP pre-trained weights. "
            "Fine-tune at 1/10th the pre-training LR. "
            "Unfreeze gradually: head first (3 epochs), then full network."
        ),
    }, indent=2)
