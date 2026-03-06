"""
All Kaggle API tools — competitions, kernels, datasets, models.
Every operation the Kaggle public API exposes, wrapped as clean functions.
"""
from __future__ import annotations
import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests

from kaggle_mcp.config import KAGGLE_API, kaggle_headers


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _kget(path: str, params: dict = None, stream: bool = False) -> requests.Response:
    url = f"{KAGGLE_API}{path}"
    return requests.get(url, headers=kaggle_headers(), params=params or {}, stream=stream, timeout=60)


def _kpost(path: str, json_data: dict = None) -> requests.Response:
    url = f"{KAGGLE_API}{path}"
    return requests.post(url, headers=kaggle_headers(), json=json_data or {}, timeout=60)


def _kdelete(path: str) -> requests.Response:
    url = f"{KAGGLE_API}{path}"
    return requests.delete(url, headers=kaggle_headers(), timeout=60)


def _ok(r: requests.Response, label: str = "") -> str:
    """Return parsed JSON text or error string."""
    if r.status_code in (200, 201):
        try:
            return json.dumps(r.json(), indent=2)
        except Exception:
            return r.text[:2000]
    return f"ERROR {r.status_code} [{label}]: {r.text[:600]}"


def _save_stream(r: requests.Response, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(path, "wb") as fp:
        for chunk in r.iter_content(65536):
            fp.write(chunk)
            total += len(chunk)
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH / PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

def validate_kaggle_token() -> dict:
    """Validate token by listing entered competitions (fast smoke-test)."""
    r = _kget("/competitions/list", {"group": "entered", "page": 1, "pageSize": 1})
    if r.status_code == 200:
        return {"valid": True, "message": "Kaggle token is valid ✓"}
    return {"valid": False, "message": f"Kaggle token invalid — HTTP {r.status_code}: {r.text[:200]}"}


def get_username() -> str:
    """Return the Kaggle username by inspecting the kernel list for known user."""
    # Kernels list with user='me' doesn't exist; we infer from listing our kernels
    r = _kget("/kernels/list", {"pageSize": 1})
    if r.status_code == 200:
        items = r.json()
        if items:
            ref = items[0].get("ref", "")
            if "/" in ref:
                return ref.split("/")[0]
    # Fallback: push a tiny kernel and read back the username from ref
    return "_unknown_"


# ═══════════════════════════════════════════════════════════════════════════════
# COMPETITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def competitions_list(
    group: str = "general",
    sort_by: str = "latestDeadline",
    search: str = "",
    page: int = 1,
    page_size: int = 20,
) -> str:
    """List Kaggle competitions. group: general|entered|inClass."""
    r = _kget("/competitions/list", {
        "group": group, "sortBy": sort_by,
        "search": search, "page": page, "pageSize": page_size,
    })
    if r.status_code != 200:
        return _ok(r, "competitions_list")
    out = []
    for c in r.json():
        out.append({
            "title":       c.get("title", ""),
            "slug":        c.get("url", "").split("/")[-1] if c.get("url") else "",
            "description": c.get("description", ""),
            "category":    c.get("category", ""),
            "reward":      c.get("reward", ""),
            "deadline":    c.get("deadline", ""),
            "metric":      c.get("evaluationMetric", ""),
            "teams":       c.get("teamCount", 0),
            "userEntered": c.get("userHasEntered", False),
            "userRank":    c.get("userRank", 0),
        })
    return json.dumps(out, indent=2)


def competition_data_files(competition_slug: str) -> str:
    """List all data files in a competition dataset."""
    all_files = []
    token = ""
    while True:
        params = {"pageToken": token} if token else {}
        r = _kget(f"/competitions/data/list/{competition_slug}", params)
        if r.status_code != 200:
            return _ok(r, "competition_data_files")
        data = r.json()
        all_files.extend(data.get("files", []))
        token = data.get("nextPageToken", "")
        if not data.get("hasNextPageToken"):
            break
    summary = [{"name": f["name"], "bytes": f.get("totalBytes", 0)} for f in all_files]
    return json.dumps({"total": len(summary), "files": summary[:200]}, indent=2)


def competition_download_file(competition_slug: str, file_name: str, save_path: str) -> str:
    """Download a specific competition data file (e.g. train.csv)."""
    r = _kget(f"/competitions/data/download/{competition_slug}/{file_name}", stream=True)
    if r.status_code != 200:
        return _ok(r, "competition_download_file")
    total = _save_stream(r, Path(save_path))
    return f"✓ Saved {file_name} → {save_path} ({total:,} bytes)"


def competition_download_all(competition_slug: str, save_dir: str = "./kaggle_data") -> str:
    """Download all competition data as zip and extract."""
    r = _kget(f"/competitions/data/download-all/{competition_slug}", stream=True)
    if r.status_code != 200:
        return _ok(r, "competition_download_all")
    save_dir = Path(save_dir)
    zip_path = save_dir / f"{competition_slug}.zip"
    total = _save_stream(r, zip_path)
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(save_dir / competition_slug)
        return f"✓ Extracted {total:,} bytes → {save_dir}/{competition_slug}/"
    except Exception as e:
        return f"✓ Downloaded {total:,} bytes → {zip_path} (ZIP extraction failed: {e})"


def competition_leaderboard(competition_slug: str, page: int = 1) -> str:
    """Fetch the competition leaderboard (top 20 teams)."""
    r = _kget(f"/competitions/{competition_slug}/leaderboard/view", {"page": page})
    if r.status_code != 200:
        return _ok(r, "competition_leaderboard")
    data = r.json()
    subs = data.get("submissions", [])[:20]
    out = [{"rank": s.get("rank"), "team": s.get("teamName"), "score": s.get("score")} for s in subs]
    return json.dumps({"totalTeams": data.get("totalTeams", 0), "top20": out}, indent=2)


def competition_submit(competition_slug: str, csv_path: str, message: str = "submission") -> str:
    """Submit a predictions CSV to a competition."""
    p = Path(csv_path)
    if not p.exists():
        return f"ERROR: File not found: {csv_path}"
    # Step 1: get upload URL
    r1 = _kpost(f"/competitions/submissions/url/{competition_slug}", {"fileName": p.name})
    if r1.status_code not in (200, 201):
        return _ok(r1, "submit_get_url")
    info = r1.json()
    upload_url = info.get("createUrl") or info.get("url", "")
    blob_token = info.get("token", "")
    if not upload_url:
        return f"ERROR: No upload URL received: {json.dumps(info)}"
    # Step 2: upload file
    with open(p, "rb") as fp:
        r2 = requests.put(upload_url, data=fp, timeout=120)
    if r2.status_code not in (200, 201):
        return f"ERROR upload: HTTP {r2.status_code}: {r2.text[:200]}"
    # Step 3: confirm submission
    r3 = _kpost(
        f"/competitions/submissions/submit/{competition_slug}",
        {"submissionDescription": message, "blobFileTokens": [blob_token]},
    )
    return f"Submit status: {r3.status_code}\n{r3.text[:500]}"


def my_submissions(competition_slug: str, page: int = 1) -> str:
    """List your submissions for a competition."""
    r = _kget(f"/competitions/submissions/list/{competition_slug}", {"page": page, "pageSize": 20})
    return _ok(r, "my_submissions")


# ═══════════════════════════════════════════════════════════════════════════════
# KERNELS / NOTEBOOKS
# ═══════════════════════════════════════════════════════════════════════════════

def kernel_push(
    title: str,
    code: str,
    competition_slug: str = "",
    dataset_sources: list[str] = None,
    kernel_type: str = "notebook",
    enable_gpu: bool = True,
    enable_internet: bool = True,
    language: str = "python",
) -> dict:
    """
    Push a kernel to Kaggle for execution.
    Returns dict with {slug, url, versionNumber, kernelId, error}.
    kernel_type: 'notebook' | 'script'
    """
    payload = {
        "newTitle":           title,
        "text":               code,
        "language":           language,
        "kernelType":         kernel_type,
        "isPrivate":          True,
        "enableGpu":          enable_gpu,
        "enableTpu":          False,
        "enableInternet":     enable_internet,
        "datasetSources":     dataset_sources or [],
        "competitionSources": [competition_slug] if competition_slug else [],
        "kernelSources":      [],
    }
    r = requests.post(f"{KAGGLE_API}/kernels/push", headers=kaggle_headers(), json=payload, timeout=60)
    if r.status_code == 409:
        # Title conflict — append timestamp
        import re, time as _t
        new_title = f"{title} {int(_t.time())}"
        payload["newTitle"] = new_title
        r = requests.post(f"{KAGGLE_API}/kernels/push", headers=kaggle_headers(), json=payload, timeout=60)
    if r.status_code not in (200, 201):
        return {"error": f"HTTP {r.status_code}: {r.text[:400]}"}
    data = r.json()
    ref = data.get("ref", "")
    slug = ref.split("/")[-1] if "/" in ref else ""
    username = ref.split("/")[-2] if ref.count("/") >= 2 else ""
    return {
        "slug":          slug,
        "username":      username,
        "full_slug":     f"{username}/{slug}",
        "url":           data.get("url", ""),
        "versionNumber": data.get("versionNumber", 0),
        "kernelId":      data.get("kernelId", 0),
        "error":         data.get("error", ""),
    }


def kernel_status(username: str, slug: str) -> dict:
    """Poll kernel status. Returns {status, failureMessage}."""
    r = _kget("/kernels/status", {"userName": username, "kernelSlug": slug})
    if r.status_code != 200:
        return {"status": "unknown", "error": f"HTTP {r.status_code}"}
    data = r.json()
    return {
        "status":         data.get("status", "unknown"),
        "failureMessage": data.get("failureMessage", ""),
    }


def kernel_output_log(username: str, slug: str) -> str:
    """Fetch the execution log (stdout/stderr) from a completed kernel."""
    r = _kget("/kernels/output", {"userName": username, "kernelSlug": slug})
    if r.status_code != 200:
        return f"ERROR {r.status_code}: {r.text[:300]}"
    data = r.json()
    log = data.get("log") or data.get("logNullable") or ""
    if not log:
        return "(no log available)"
    try:
        entries = json.loads(log)
        lines = []
        for e in entries:
            stream = "[err]" if e.get("stream_name") == "stderr" else ""
            text = e.get("data", "")
            if text.strip():
                lines.append(f"{stream}{text}")
        return "".join(lines)[:15000]
    except Exception:
        return log[:15000]


def kernel_output_files(username: str, slug: str, save_dir: str = "./kernel_outputs") -> str:
    """Download output files (submission.csv, checkpoints, etc.) from a kernel."""
    r = _kget(f"/kernels/{username}/{slug}/output")
    if r.status_code != 200:
        # Try alternative endpoint
        r = _kget("/kernels/output", {"userName": username, "kernelSlug": slug})
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    content = r.content
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            z.extractall(save_dir)
            return f"✓ Extracted {len(z.namelist())} files to {save_dir}: {z.namelist()}"
    out = save_dir / "log.txt"
    out.write_bytes(content)
    return f"✓ Saved {len(content):,} bytes to {out}"


def kernel_monitor(username: str, slug: str, poll_secs: int = 20, max_mins: int = 120) -> str:
    """
    Poll kernel status until complete/error and return final status + log snippet.
    Blocks until done.
    """
    deadline = time.time() + max_mins * 60
    prev = ""
    while time.time() < deadline:
        info = kernel_status(username, slug)
        status = info.get("status", "unknown")
        if status != prev:
            prev = status
        if status in ("complete", "error", "cancelled", "failed"):
            log = kernel_output_log(username, slug)
            return json.dumps({
                "finalStatus":    status,
                "failureMessage": info.get("failureMessage", ""),
                "url":            f"https://www.kaggle.com/code/{username}/{slug}",
                "logTail":        log[-5000:],
            }, indent=2)
        time.sleep(poll_secs)
    return json.dumps({"finalStatus": "timeout", "url": f"https://www.kaggle.com/code/{username}/{slug}"})


def kernels_list(username: str = "", search: str = "", page: int = 1) -> str:
    """List public kernels (optionally filtered by user or search term)."""
    params = {"page": page, "pageSize": 20}
    if username:
        params["user"] = username
    if search:
        params["search"] = search
    r = _kget("/kernels/list", params)
    return _ok(r, "kernels_list")


def kernel_pull(username: str, slug: str, save_dir: str = "./pulled_kernels") -> str:
    """Download the source code of a kernel."""
    r = _kget("/kernels/pull", {"userName": username, "kernelSlug": slug})
    if r.status_code != 200:
        return _ok(r, "kernel_pull")
    data = r.json()
    source = data.get("blob", {}).get("source", "")
    save_path = Path(save_dir) / f"{slug}.py"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(source, encoding="utf-8")
    return f"✓ Pulled {len(source):,} chars → {save_path}"


# ═══════════════════════════════════════════════════════════════════════════════
# DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

def datasets_search(query: str, sort_by: str = "votes", page: int = 1) -> str:
    """Search Kaggle datasets. sort_by: hottest|votes|updated|active|published."""
    r = _kget("/datasets/list", {
        "search": query, "sortBy": sort_by, "page": page, "pageSize": 10,
    })
    if r.status_code != 200:
        return _ok(r, "datasets_search")
    out = []
    for d in r.json():
        out.append({
            "ref":       d.get("ref", ""),
            "title":     d.get("title", ""),
            "bytes":     d.get("totalBytes", 0),
            "votes":     d.get("voteCount", 0),
            "downloads": d.get("downloadCount", 0),
            "license":   d.get("licenseName", ""),
        })
    return json.dumps(out, indent=2)


def dataset_files(dataset_ref: str) -> str:
    """List files in a Kaggle dataset (owner/dataset-name)."""
    owner, name = dataset_ref.split("/", 1)
    r = _kget(f"/datasets/{owner}/{name}/files")
    return _ok(r, "dataset_files")


def dataset_download(dataset_ref: str, save_dir: str = "./kaggle_data") -> str:
    """Download and extract a Kaggle dataset."""
    owner, name = dataset_ref.split("/", 1)
    r = _kget(f"/datasets/download/{owner}/{name}", stream=True)
    if r.status_code != 200:
        return _ok(r, "dataset_download")
    save_dir = Path(save_dir)
    zip_path = save_dir / f"{name}.zip"
    total = _save_stream(r, zip_path)
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(save_dir / name)
        return f"✓ Extracted {total:,} bytes → {save_dir / name}/"
    except Exception:
        return f"✓ Downloaded {total:,} bytes → {zip_path}"


def dataset_create(
    title: str,
    files: list[str],
    is_private: bool = True,
    license_name: str = "CC0-1.0",
    description: str = "",
) -> str:
    """Create a new Kaggle dataset from local files."""
    import base64
    dataset_items = []
    for fpath in files:
        p = Path(fpath)
        if not p.exists():
            return f"ERROR: File not found: {fpath}"
        content = base64.b64encode(p.read_bytes()).decode()
        dataset_items.append({"name": p.name, "content": content})
    slug = title.lower().replace(" ", "-")
    payload = {
        "title":       title,
        "slug":        slug,
        "isPrivate":   is_private,
        "ownerSlug":   "",  # filled server-side
        "licenseName": license_name,
        "description": description,
        "files":       dataset_items,
    }
    r = requests.post(f"{KAGGLE_API}/datasets/create/new", headers=kaggle_headers(), json=payload, timeout=120)
    return _ok(r, "dataset_create")


def dataset_create_version(
    dataset_ref: str,
    files: list[str],
    version_notes: str = "Updated via MCP",
    delete_old_versions: bool = False,
) -> str:
    """Create a new version of an existing Kaggle dataset."""
    import base64
    owner, name = dataset_ref.split("/", 1)
    items = []
    for fpath in files:
        p = Path(fpath)
        if not p.exists():
            return f"ERROR: File not found: {fpath}"
        items.append({"name": p.name, "content": base64.b64encode(p.read_bytes()).decode()})
    payload = {
        "versionNotes":    version_notes,
        "deleteOldVersions": delete_old_versions,
        "files":           items,
    }
    r = requests.post(
        f"{KAGGLE_API}/datasets/create/version/{owner}/{name}",
        headers=kaggle_headers(), json=payload, timeout=120,
    )
    return _ok(r, "dataset_create_version")


def my_datasets(page: int = 1) -> str:
    """List your own Kaggle datasets."""
    r = _kget("/datasets/list", {"mine": True, "page": page, "pageSize": 20})
    return _ok(r, "my_datasets")


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def models_search(query: str, page: int = 1) -> str:
    """Search the Kaggle models hub."""
    r = _kget("/models/list", {"search": query, "page": page, "pageSize": 10})
    return _ok(r, "models_search")


def model_info(owner: str, model_slug: str) -> str:
    """Get details about a Kaggle model."""
    r = _kget(f"/models/{owner}/{model_slug}/get")
    return _ok(r, "model_info")


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def push_and_run(
    title: str,
    code: str,
    competition_slug: str = "",
    dataset_sources: list[str] = None,
    enable_gpu: bool = True,
    kernel_type: str = "script",
    wait: bool = False,
    poll_secs: int = 30,
    max_mins: int = 120,
) -> str:
    """
    One-shot: push kernel → optionally block until done → return log.
    Returns JSON with {slug, url, status, logTail}.
    """
    result = kernel_push(
        title=title, code=code,
        competition_slug=competition_slug,
        dataset_sources=dataset_sources,
        enable_gpu=enable_gpu,
        kernel_type=kernel_type,
    )
    if result.get("error"):
        return json.dumps({"error": result["error"]})

    username = result["username"]
    slug     = result["slug"]
    url      = result["url"]

    if not wait:
        return json.dumps({"slug": slug, "username": username, "url": url, "status": "queued"}, indent=2)

    monitor_result = kernel_monitor(username, slug, poll_secs=poll_secs, max_mins=max_mins)
    return monitor_result
