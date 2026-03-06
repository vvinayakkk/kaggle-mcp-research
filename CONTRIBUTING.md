# Contributing to Kaggle Research MCP

Thank you for your interest in contributing! This project aims to be the definitive MCP server for Kaggle + research automation.

## Ways to Contribute

- **Bug reports** — open an issue with reproduction steps
- **Feature requests** — open an issue describing the use case
- **Pull requests** — see guidelines below
- **Documentation** — improve README, add examples, fix typos

## Development Setup

```bash
git clone https://github.com/vvinayakkk/kaggle-mcp-research.git
cd kaggle-mcp-research
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/Mac
pip install -e ".[dev]"
```

Set your credentials:
```bash
# Windows
set KAGGLE_TOKEN=KGAT_your_token
set HF_TOKEN=hf_your_token

# Linux/Mac
export KAGGLE_TOKEN=KGAT_your_token
export HF_TOKEN=hf_your_token
```

Run tests:
```bash
pytest tests/ -v
```

## Code Style

- Python 3.10+
- `ruff` for linting: `ruff check src/`
- Type hints on all public functions
- Docstrings on all MCP tools (they appear in Copilot's tool list)
- Return JSON strings from all tool functions (not dicts)

## Adding a New Tool

1. Add the implementation in the appropriate `src/kaggle_mcp/tools/*.py` file
2. Register it in `src/kaggle_mcp/server.py` with `@mcp.tool()`
3. Add a test in `tests/`
4. Update the Tools Reference table in `README.md`

## Pull Request Process

1. Fork the repo and create a feature branch: `git checkout -b feat/my-feature`
2. Write tests for your changes
3. Ensure `pytest tests/` passes
4. Submit a PR with a clear description of what it does and why

## Commit Convention

Use conventional commits:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `test:` test additions
- `refactor:` code change without feature/fix
- `chore:` maintenance

Example: `feat: add wandb experiment tracking tool`

## Code of Conduct

Be kind, constructive, and inclusive. This project follows the [Contributor Covenant](https://www.contributor-covenant.org/).
