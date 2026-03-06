"""
HuggingFace Hub tools — datasets, models, spaces, inference.
"""
from __future__ import annotations
import json
from typing import Optional

import requests

from kaggle_mcp.config import HF_API, hf_headers


def _hget(path: str, params: dict = None) -> requests.Response:
    return requests.get(f"{HF_API}{path}", headers=hf_headers(), params=params or {}, timeout=30)


def _ok(r: requests.Response, label: str = "") -> str:
    if r.status_code == 200:
        try:
            return json.dumps(r.json(), indent=2)
        except Exception:
            return r.text[:3000]
    return f"ERROR {r.status_code} [{label}]: {r.text[:400]}"


# ── Credentials ───────────────────────────────────────────────────────────────

def validate_hf_token() -> dict:
    """Validate the HuggingFace token by calling the whoami endpoint."""
    r = requests.get("https://huggingface.co/api/whoami", headers=hf_headers(), timeout=15)
    if r.status_code == 200:
        data = r.json()
        return {"valid": True, "username": data.get("name", "unknown"), "message": "HuggingFace token is valid ✓"}
    if r.status_code == 401:
        return {"valid": False, "message": "HuggingFace token is invalid (401 Unauthorized)"}
    from kaggle_mcp.config import get_hf_token
    if not get_hf_token():
        return {"valid": True, "message": "No HF token set — public access only (fine for public datasets)"}
    return {"valid": False, "message": f"HuggingFace API error: HTTP {r.status_code}"}


# ── Datasets ──────────────────────────────────────────────────────────────────

def hf_search_datasets(
    query: str,
    task: str = "",
    full: bool = False,
    limit: int = 10,
) -> str:
    """Search HuggingFace datasets. task e.g. 'image-classification', 'text-classification'."""
    params: dict = {"search": query, "limit": limit}
    if task:
        params["task_categories"] = task
    if full:
        params["full"] = "true"
    r = _hget("/datasets", params)
    if r.status_code != 200:
        return _ok(r, "hf_search_datasets")
    datasets = r.json()
    out = []
    for d in datasets:
        out.append({
            "id":        d.get("id", ""),
            "downloads": d.get("downloads", 0),
            "likes":     d.get("likes", 0),
            "tags":      d.get("tags", [])[:8],
            "private":   d.get("private", False),
            "gated":     d.get("gated", False),
        })
    return json.dumps(out, indent=2)


def hf_dataset_info(dataset_id: str) -> str:
    """Get metadata about a HuggingFace dataset."""
    r = _hget(f"/datasets/{dataset_id}")
    return _ok(r, "hf_dataset_info")


def hf_dataset_files(dataset_id: str) -> str:
    """List files in a HuggingFace dataset repository."""
    r = requests.get(
        f"https://huggingface.co/api/datasets/{dataset_id}/tree/main",
        headers=hf_headers(), timeout=30,
    )
    if r.status_code == 200:
        files = r.json()
        out = [{"path": f.get("path"), "size": f.get("size", 0), "type": f.get("type")} for f in files]
        return json.dumps(out, indent=2)
    return _ok(r, "hf_dataset_files")


def hf_download_dataset_file(dataset_id: str, filename: str, save_path: str) -> str:
    """Download a single file from a HuggingFace dataset."""
    import os
    from pathlib import Path
    url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{filename}"
    r = requests.get(url, headers=hf_headers(), stream=True, timeout=120, allow_redirects=True)
    if r.status_code != 200:
        return f"ERROR {r.status_code}: {r.text[:300]}"
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(p, "wb") as fp:
        for chunk in r.iter_content(65536):
            fp.write(chunk)
            total += len(chunk)
    return f"✓ Downloaded {filename} → {save_path} ({total:,} bytes)"


# ── Models ────────────────────────────────────────────────────────────────────

def hf_search_models(
    query: str,
    task: str = "",
    library: str = "",
    limit: int = 10,
) -> str:
    """
    Search HuggingFace models.
    task e.g. 'image-classification', 'text-classification', 'token-classification'.
    library e.g. 'transformers', 'timm', 'diffusers'.
    """
    params: dict = {"search": query, "limit": limit}
    if task:
        params["pipeline_tag"] = task
    if library:
        params["library"] = library
    r = _hget("/models", params)
    if r.status_code != 200:
        return _ok(r, "hf_search_models")
    models = r.json()
    out = []
    for m in models:
        out.append({
            "id":           m.get("id", ""),
            "downloads":    m.get("downloads", 0),
            "likes":        m.get("likes", 0),
            "pipeline_tag": m.get("pipeline_tag", ""),
            "library_name": m.get("library_name", ""),
            "tags":         m.get("tags", [])[:5],
        })
    return json.dumps(out, indent=2)


def hf_model_info(model_id: str) -> str:
    """Get full metadata for a HuggingFace model."""
    r = _hget(f"/models/{model_id}")
    return _ok(r, "hf_model_info")


def hf_model_card(model_id: str) -> str:
    """Fetch the README/model card of a HuggingFace model."""
    url = f"https://huggingface.co/{model_id}/raw/main/README.md"
    r = requests.get(url, headers=hf_headers(), timeout=15)
    if r.status_code == 200:
        return r.text[:8000]
    return f"ERROR {r.status_code}: {r.text[:300]}"


# ── Inference API ─────────────────────────────────────────────────────────────

def hf_run_inference(model_id: str, inputs: str | list | dict) -> str:
    """
    Run the HuggingFace Inference API on a model.
    inputs: string or list of strings for text tasks.
    """
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    r = requests.post(url, headers=hf_headers(), json={"inputs": inputs}, timeout=60)
    return _ok(r, "hf_run_inference")


# ── Spaces ────────────────────────────────────────────────────────────────────

def hf_search_spaces(query: str, limit: int = 10) -> str:
    """Search HuggingFace Spaces (demos and apps)."""
    r = _hget("/spaces", {"search": query, "limit": limit})
    if r.status_code != 200:
        return _ok(r, "hf_search_spaces")
    spaces = r.json()
    out = [{"id": s.get("id"), "sdk": s.get("sdk"), "likes": s.get("likes", 0)} for s in spaces]
    return json.dumps(out, indent=2)
