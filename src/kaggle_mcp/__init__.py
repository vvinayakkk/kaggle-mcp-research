"""
kaggle-mcp-research: Full Kaggle + HuggingFace MCP server for VS Code Copilot.

Research automation — upload a ZIP, get a complete ML solution.
"""

__version__ = "1.0.0"
__author__ = "Vinayak Bhatia"
__license__ = "MIT"

from kaggle_mcp.server import mcp, main

__all__ = ["mcp", "main"]
