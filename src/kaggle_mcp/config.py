"""
Configuration management for Kaggle MCP Server.
Loads tokens from environment variables (preferred) or .env file.
"""
import os
from pathlib import Path
from typing import Optional

# Load .env if present
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).parent.parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
    else:
        load_dotenv()
except ImportError:
    pass


def _require(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise EnvironmentError(
            f"Missing required environment variable: {name}\n"
            "Set it in your .vscode/mcp.json env section, system env, or .env file.\n"
            "See README.md → Configuration for details."
        )
    return val


def get_kaggle_token() -> str:
    """Return Kaggle API token (Bearer KGAT_...)."""
    # Support both KAGGLE_TOKEN (Bearer) and KAGGLE_KEY (legacy)
    token = os.environ.get("KAGGLE_TOKEN", "").strip()
    if not token:
        token = os.environ.get("KAGGLE_KEY", "").strip()
    if not token:
        raise EnvironmentError(
            "Missing KAGGLE_TOKEN environment variable.\n"
            "Set KAGGLE_TOKEN=KGAT_... in your MCP configuration.\n"
            "Get your token at https://www.kaggle.com/settings/account"
        )
    return token


def get_hf_token() -> Optional[str]:
    """Return HuggingFace token (optional for public datasets)."""
    return (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_TOKEN", "").strip()
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
        or None
    )


def kaggle_headers() -> dict:
    return {
        "Authorization": f"Bearer {get_kaggle_token()}",
        "Content-Type": "application/json",
    }


def hf_headers() -> dict:
    token = get_hf_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


KAGGLE_API = "https://www.kaggle.com/api/v1"
HF_API = "https://huggingface.co/api"
ARXIV_API = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
PAPERS_WITH_CODE_API = "https://paperswithcode.com/api/v1"
