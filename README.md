<div align="center">

<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/MCP-FastMCP-FF6B35?style=for-the-badge" />
<img src="https://img.shields.io/badge/Kaggle-API%20v1-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" />
<img src="https://img.shields.io/badge/HuggingFace-%F0%9F%A4%97-FFD21E?style=for-the-badge" />
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />

<br /><br />

# 🏆 Kaggle Research MCP

### *Give GitHub Copilot a research team, a GPU cluster, and a Kaggle account.*

**One prompt. 200 papers. A novel architecture. A trained model. Kaggle results.**  
Fully automated, end-to-end, zero manual steps.

[**Quick Start**](#-quick-start) · [**Features**](#-features) · [**How It Works**](#-how-it-works) · [**Tools Reference**](#-tools-reference) · [**Contributing**](#-contributing)

</div>

---

## 🎯 What Is This?

**Kaggle Research MCP** is a [Model Context Protocol](https://modelcontextprotocol.io) server that integrates directly into **VS Code + GitHub Copilot**, giving the AI assistant real-time access to:

| Capability | What It Does |
|-----------|--------------|
| 🔬 **Deep Research** | Searches arXiv, Semantic Scholar, Papers With Code, and SOTA leaderboards |
| 🧠 **Architecture Design** | Analyses baselines, identifies gaps, proposes novel architectures with citations |
| 📓 **Notebook Generation** | Creates complete, GPU-ready Kaggle notebooks (image / NLP / tabular / detection) |
| 🚀 **Kaggle Automation** | Pushes notebooks, monitors execution, retrieves outputs — all automatically |
| 🤗 **HuggingFace Access** | Searches models, downloads datasets, runs Inference API |
| 📦 **ZIP Intake** | Decodes your professor's half-done research ZIP — code, papers, baselines, datasets |

### The Magic

You upload a ZIP of your research material and type *one sentence* to Copilot.  
Copilot reads your existing work, surveys 50–200 papers, designs a better architecture, generates a full training notebook, uploads it to Kaggle with GPU enabled, waits for results, and returns a structured analysis — **without you doing anything else**.

---

## ✨ Features

- **35+ MCP tools** covering every Kaggle and HuggingFace API endpoint
- **Autonomous research pipeline** — single prompt triggers everything
- **Smart notebook generation** templated for 5 task types with best practices baked in:
  - MixUp / CutMix / RandAugment augmentation
  - Mixed precision (FP16), gradient clipping, early stopping
  - TTA inference, checkpoint saving, training curves
- **SOTA benchmark lookup** — exact leaderboard numbers from Papers With Code
- **ZIP decoder** — analyses your professor's code, infers the problem type, extracts baselines
- **Credential-safe** — tokens stored as VS Code secrets, never in files
- **Works for any Kaggle competition** — not tied to a specific dataset
- **Free research APIs** — arXiv, Semantic Scholar, Papers With Code (no additional API keys needed)

---

## ⚡ Quick Start

### Step 1 — Install

**Option A: Via `uvx` (zero-install, recommended)**
```bash
# Nothing to install — VS Code handles it via mcp.json
```

**Option B: Via pip**
```bash
pip install kaggle-mcp-research
```

**Option C: From source**
```bash
git clone https://github.com/vvinayakkk/kaggle-mcp-research.git
cd kaggle-mcp-research
pip install -e .
```

---

### Step 2 — Configure VS Code

1. Copy `.vscode/mcp.json` from this repo to your project's `.vscode/` folder  
   *(or open this repo directly in VS Code — it auto-detects)*

2. Open VS Code → press `Ctrl+Shift+P` → **"MCP: Connect to Server"**  
   VS Code will securely prompt you for:
   - **Kaggle API Token** — get it at [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token
   - **HuggingFace Token** — get it at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

   > Tokens are stored in VS Code's secret storage — never written to disk.

---

### Step 3 — Select Claude Sonnet 4.6

In the VS Code Copilot chat panel, click the model picker and choose **Claude Sonnet 4.6** for the best research-pipeline reasoning.

---

### Step 4 — Run Your First Pipeline

Open the Copilot chat (`Ctrl+Shift+I`) and type:

```
I have a scene classification competition on Kaggle: dl-ise-problem-statement-one
Here's my research ZIP: C:/Users/me/research.zip
Please do a full analysis, find the best architecture, generate a notebook, and run it.
```

Copilot will autonomously execute all 6 phases of the research pipeline.

---

## 🔄 How It Works

```
User: "here is my zip + competition slug"
           │
           ▼
 ┌─────────────────────┐
 │  Phase 1: Decode    │  extract_zip_research()
 │  Research Material  │  → task type, baselines, datasets
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │  Phase 2: Literature│  full_literature_sweep()
 │  Survey (50+ papers)│  search_arxiv() + get_sota_for_task()
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │  Phase 3: Design    │  Copilot synthesises architecture
 │  Novel Architecture │  based on SOTA gaps + your baseline
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │  Phase 4: Generate  │  generate_kaggle_notebook()
 │  GPU Notebook       │  → complete .ipynb with all tricks
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │  Phase 5: Deploy    │  kaggle_push_and_run()
 │  and Monitor        │  → GPU kernel + live monitoring
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │  Phase 6: Analyse   │  kaggle_kernel_output_log()
 │  Results            │  → metrics, charts, submission
 └─────────────────────┘
           │
           ▼
    >> Architecture report + training curves + submission.csv
```

---

## 🛠 Tools Reference

### Credential Validation
| Tool | Description |
|------|-------------|
| `validate_kaggle_token` | Verify Kaggle API token, returns username |
| `validate_hf_token` | Verify HuggingFace token |
| `validate_all_credentials` | Check both at once |

### Kaggle — Competitions
| Tool | Description |
|------|-------------|
| `kaggle_list_competitions` | Browse entered / public competitions |
| `kaggle_competition_files` | List downloadable dataset files |
| `kaggle_download_competition_file` | Download a specific file |
| `kaggle_download_all_competition_data` | Download complete dataset ZIP |
| `kaggle_leaderboard` | Get competition leaderboard |
| `kaggle_submit_predictions` | Submit a CSV file |
| `kaggle_my_submissions` | View your submission history |

### Kaggle — Kernels
| Tool | Description |
|------|-------------|
| `kaggle_push_kernel` | Push a notebook or script |
| `kaggle_kernel_status` | Check run status |
| `kaggle_kernel_output_log` | Get stdout / stderr |
| `kaggle_kernel_output_files` | Download all output files |
| `kaggle_monitor_kernel` | Block and poll until done |
| `kaggle_push_and_run` | **One-shot: push + wait + results** |
| `kaggle_list_kernels` | List your kernels |
| `kaggle_pull_kernel` | Download kernel source |

### Kaggle — Datasets and Models
| Tool | Description |
|------|-------------|
| `kaggle_search_datasets` | Search public datasets |
| `kaggle_dataset_files` | List files in a dataset |
| `kaggle_download_dataset` | Download dataset |
| `kaggle_create_dataset` | Upload a new dataset |
| `kaggle_list_my_datasets` | List your datasets |
| `kaggle_search_models` | Search Kaggle model hub |
| `kaggle_model_info` | Get model details |

### HuggingFace
| Tool | Description |
|------|-------------|
| `hf_search_datasets` | Search HF Hub datasets |
| `hf_dataset_info` | Dataset metadata |
| `hf_dataset_files` | List dataset files |
| `hf_download_dataset_file` | Download a file |
| `hf_search_models` | Search models by task/library |
| `hf_model_info` | Model metadata |
| `hf_model_card` | Full model card / README |
| `hf_run_inference` | Run Inference API |
| `hf_search_spaces` | Find demo Spaces |

### Research
| Tool | Description |
|------|-------------|
| `search_arxiv` | Search arXiv papers |
| `search_semantic_scholar` | Search with citation counts |
| `search_paperswithcode` | Find papers with code |
| `get_sota_for_task` | SOTA leaderboard numbers |
| `get_task_methods` | Common methods for a task |
| `full_literature_sweep` | **All-in-one literature survey** |

### Notebook and Pipeline
| Tool | Description |
|------|-------------|
| `generate_kaggle_notebook` | Create complete .ipynb |
| `extract_zip_research` | Analyse uploaded ZIP |
| `analyze_research_directory` | Analyse extracted directory |
| `summarize_code_file` | Summarise .py / .ipynb file |
| `run_full_research_pipeline` | **One-shot: ZIP → train → results** |

---

## 📁 Repository Structure

```
kaggle-mcp-research/
├── src/
│   └── kaggle_mcp/
│       ├── server.py              ← FastMCP server (all 35+ tools)
│       ├── config.py              ← Token loading from env vars
│       ├── __init__.py
│       ├── __main__.py            ← python -m kaggle_mcp
│       └── tools/
│           ├── kaggle.py          ← All Kaggle API operations
│           ├── huggingface.py     ← HuggingFace Hub operations
│           ├── research.py        ← arXiv, S2, Papers With Code
│           ├── notebook.py        ← Notebook generator templates
│           └── zip_processor.py   ← ZIP/directory analyzer
├── tests/
│   └── test_credentials.py       ← API validation test suite
├── .github/
│   └── copilot-instructions.md   ← Copilot pipeline instructions
├── .vscode/
│   └── mcp.json                  ← VS Code MCP config
├── pyproject.toml
├── .env.example
└── README.md
```

---

## 🧪 Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run credential tests (requires real API tokens)
KAGGLE_TOKEN=KGAT_xxx HF_TOKEN=hf_xxx pytest tests/ -v
```

Expected output:
```
PASSED tests/test_credentials.py::test_kaggle_api_valid
PASSED tests/test_credentials.py::test_hf_api_valid
PASSED tests/test_credentials.py::test_arxiv_reachable
PASSED tests/test_credentials.py::test_semantic_scholar_reachable
PASSED tests/test_credentials.py::test_paperswithcode_reachable
PASSED tests/test_credentials.py::test_mcp_server_tools_registered
```

---

## 🔧 Local Development

```bash
git clone https://github.com/vvinayakkk/kaggle-mcp-research.git
cd kaggle-mcp-research
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / Mac
pip install -e ".[dev]"

# Set tokens in shell
set KAGGLE_TOKEN=KGAT_your_token   # Windows
export KAGGLE_TOKEN=KGAT_your_token  # Linux/Mac

# Run the MCP server directly
python -m kaggle_mcp
```

For VS Code with local server, update `.vscode/mcp.json`:
```json
{
  "servers": {
    "kaggle-mcp-local": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "kaggle_mcp"],
      "env": {
        "KAGGLE_TOKEN": "${input:kaggle_token}",
        "HF_TOKEN":     "${input:hf_token}"
      }
    }
  }
}
```

---

## 💡 Usage Examples

### Full Research Pipeline from ZIP
```
I have a Kaggle competition (dl-ise-problem-statement-one) and a ZIP  
with my professor's starter code at C:/research/project.zip.  
Please research everything, build a better model, and run it.
```

### Find Best Dataset for a Topic
```
I want to work on fine-grained vehicle recognition.
Where can I find good datasets? Also show me the SOTA baselines.
```

### Paper Deep-Dive
```
Search arXiv for the last 6 months of papers on Vision Mamba.
Find all SOTA scores and papers with GitHub implementations.
```

### Submit to Competition
```
My submission CSV is at ./outputs/submission.csv.
Submit it to competition dl-ise-problem-statement-one.
```

---

## 🔐 Security

- API tokens passed as environment variables — **never written to disk**
- VS Code's `inputs` with `"password": true` stores tokens in OS keychain
- No outbound requests except to official Kaggle, HuggingFace, arXiv, and S2 APIs
- No telemetry, no logging of credentials

---

## 🤝 Contributing

Contributions are very welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

**Ideas:**
- W&B / MLflow experiment tracking tools
- Automated hyperparameter search report tool
- Support for Kaggle Notebooks v2
- Domain-specific templates (medical imaging, satellite, etc.)
- RAG over PDF research papers

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

Built with [FastMCP](https://github.com/jlowin/fastmcp) · [Kaggle API](https://github.com/Kaggle/kaggle-api) · [HuggingFace Hub](https://huggingface.co) · [Papers With Code](https://paperswithcode.com/api) · [Semantic Scholar](https://api.semanticscholar.org/) · [arXiv API](https://arxiv.org/help/api/)

---

<div align="center">

**Made with love for the research and Kaggle community**

Star this repo if it saves you time!

</div>