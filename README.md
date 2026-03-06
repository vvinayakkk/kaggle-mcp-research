<div align="center">

<img src="https://img.shields.io/pypi/v/kaggle-mcp-research?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI" />
<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/MCP-FastMCP-FF6B35?style=for-the-badge" />
<img src="https://img.shields.io/badge/Kaggle-API%20v1-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" />
<img src="https://img.shields.io/badge/HuggingFace-%F0%9F%A4%97-FFD21E?style=for-the-badge" />
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
<img src="https://img.shields.io/github/actions/workflow/status/vvinayakkk/kaggle-mcp-research/ci.yml?style=for-the-badge&label=CI" />

<br /><br />

# 🏆 Kaggle Research MCP v2

### *Give GitHub Copilot a research team, a brutal critic, a GPU cluster, and a Kaggle account.*

**One prompt. 200 papers. Brutally evaluated architecture. Notebook generated. Trained. Submitted.**  
Fully automated, end-to-end, zero manual steps — with academic-grade evaluation baked in.

[**Quick Start**](#-quick-start) · [**How It Works**](#-how-it-works-8-phase-pipeline) · [**Tools Reference**](#-tools-reference) · [**Contributing**](#-contributing)

</div>

---

## 🎯 What Is This?

**Kaggle Research MCP v2** is a [Model Context Protocol](https://modelcontextprotocol.io) server that integrates directly into **VS Code + GitHub Copilot**, giving the AI assistant 55+ real-time tools across the full ML lifecycle:

| Capability | What It Does |
|-----------|--------------|
| 🔬 **Deep Research** | Method evolution timelines, competition winning solutions, cross-dataset SOTA comparison |
| 🧠 **Architecture Design** | Designs novel architectures from literature gaps + your baselines |
| 🔥 **Brutal Evaluation** | Red-team critiques, failure mode detection, compute feasibility — before any GPU time |
| 🎓 **Academic Review Sim** | Full ICLR/NeurIPS/CVPR rubric scoring with accept/reject prediction |
| 📊 **Training Diagnostics** | Parses any training log → overfitting / underfitting / plateau / instability diagnosis |
| 📓 **Notebook Generation** | Complete, GPU-ready Kaggle notebooks (image / NLP / tabular / detection / segmentation) |
| 🚀 **Kaggle Automation** | Pushes notebooks, monitors execution, retrieves outputs — all automatically |
| 🤗 **HuggingFace Access** | Searches models, downloads datasets, runs Inference API |
| 📦 **ZIP Intake** | Decodes your professor's half-done research ZIP — code, papers, baselines, datasets |

### The Magic

You upload a ZIP of your research material and type *one sentence* to Copilot.  
Copilot reads your existing work, surveys 50–200 papers, **brutally evaluates and iterates the architecture 3 times**, generates a publication-ready notebook, uploads it to Kaggle, waits for results, and routes to the right Q1 journal — **without you doing anything else**.

---

## ✨ Features

- **55+ MCP tools** covering every Kaggle and HuggingFace API endpoint plus deep research and evaluation
- **Mandatory 3-iteration brutal evaluation loop** — architecture is red-teamed before any GPU is touched
- **Academic reviewer simulation** — ICLR / NeurIPS / CVPR / ICML rubrics with accept-probability estimates
- **Smart notebook generation** templated for 5 task types with best practices baked in:
  - MixUp / CutMix / RandAugment augmentation
  - Mixed precision (FP16), gradient clipping, early stopping
  - TTA inference, checkpoint saving, training curves
- **SOTA benchmark lookup** — exact leaderboard numbers from Papers With Code
- **Method evolution timelines** — track how a technique progressed year-by-year
- **Compute feasibility estimator** — memory + runtime estimates before submitting to Kaggle
- **Training log interpreter** — paste any log, get a diagnosis (overfitting / plateau / instability)
- **ZIP decoder** — analyses your professor's code, infers the problem type, extracts baselines
- **Q1 journal routing** — after strong results, recommends IEEE TPAMI / IJCV / TIP with required experiments
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

Copilot will autonomously execute all **8 phases** of the research pipeline.

---

## 🔄 How It Works — 8-Phase Pipeline

```
User: "here is my zip + competition slug"  (that's all you type)
              │
        ┌─────▼──────────────────────────────────────────┐
        │  Phase 0 · Intake                              │
        │  extract_zip_research + analyze_research_dir   │
        │  → task type, baselines, datasets, code        │
        └─────┬──────────────────────────────────────────┘
              │
        ┌─────▼──────────────────────────────────────────┐
        │  Phase 1 · Deep Literature (50–200 papers)     │
        │  full_literature_sweep + analyse_method_       │
        │  evolution + find_competition_winning_solutions│
        │  + papers_with_negative_results                │
        └─────┬──────────────────────────────────────────┘
              │
        ┌─────▼──────────────────────────────────────────┐
        │  Phase 2 · Gap Analysis → arch_v1 designed     │
        │  compare_sota_methods + identify_research_gaps │
        │  + cross_dataset_analysis                      │
        └─────┬──────────────────────────────────────────┘
              │
        ┌─────▼──────────────────────────────────────────┐
        │  Phase 3 · Brutal Evaluation Loop ×3           │
        │  brutal_evaluate → reiterate_architecture ×3   │
        │  → roast_approach (final)                      │
        │  !! ABORTS if score < 4 after 3 iterations !!  │
        └─────┬──────────────────────────────────────────┘
              │
        ┌─────▼──────────────────────────────────────────┐
        │  Phase 4 · Academic Review Gate                │
        │  reviewer_perspective + paper_worthiness       │
        │  Gate: readiness ≥ 40 % to proceed             │
        └─────┬──────────────────────────────────────────┘
              │
        ┌─────▼──────────────────────────────────────────┐
        │  Phase 5 · Compute Planning                    │
        │  estimate_kaggle_feasibility + design_ablation │
        │  + identify_hard_samples                       │
        └─────┬──────────────────────────────────────────┘
              │
        ┌─────▼──────────────────────────────────────────┐
        │  Phase 6 · Notebook + Submit                   │
        │  generate_kaggle_notebook → kaggle_push_and_run│
        │  → kaggle_kernel_output_log                    │
        └─────┬──────────────────────────────────────────┘
              │
        ┌─────▼──────────────────────────────────────────┐
        │  Phase 7 · Results Analysis                    │
        │  interpret_training_log + suggest_ensemble     │
        │  + generate_hypothesis_test_plan               │
        └─────┬──────────────────────────────────────────┘
              │
        ┌─────▼──────────────────────────────────────────┐
        │  Phase 8 · Publication Route (if strong)       │
        │  q1_journal_analysis → ranked list:            │
        │  IEEE TPAMI / IJCV / TIP / Pattern Recognition │
        └─────────────────────────────────────────────────┘
              │
              ▼
   ✅  Architecture report · training curves
       submission.csv · journal roadmap

```

> The diagram above focuses on the flow.
> For complete Phase definitions, see [`.github/copilot-instructions.md`](.github/copilot-instructions.md).

---

## ⚡ Quick Start — old anchor kept for links

<!-- anchor: quick-start -->
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

### Deep Research *(v2.0.0)*
| Tool | Description |
|------|-------------|
| `analyse_method_evolution` | Year-by-year timeline of a technique, inflection points, current frontier |
| `find_competition_winning_solutions` | Top Kaggle kernels for a competition with detected architectures |
| `compare_sota_methods` | Side-by-side SOTA table for a task from Papers With Code |
| `identify_research_gaps` | Unstudied technique combinations from a set of abstracts |
| `fetch_paper_implementation` | Repos for a paper sorted by stars / official status |
| `papers_with_negative_results` | Papers reporting failures, limitations, pitfalls for a topic |
| `deep_dive_single_paper` | Citations, references, reproducibility rating for one arXiv paper |
| `cross_dataset_analysis` | SOTA comparison across multiple benchmarks for a task+arch |

### Evaluation & Academic Review *(v2.0.0)*
| Tool | Description |
|------|-------------|
| `brutal_evaluate` | Red-team critique: FATAL_FLAWS, SCORES, VERDICT, practice audit |
| `reiterate_architecture` | Per-iteration prescription (backbone/aug → recipe → TTA/ensemble) |
| `roast_approach` | Humorous but technically precise critique with redemption arc |
| `reviewer_perspective` | Full ICLR/NeurIPS/CVPR/ICML/AAAI rubric scoring |
| `paper_worthiness` | Acceptance probability, missing experiments, priority actions |
| `q1_journal_analysis` | Ranked Q1 journal targets (TPAMI/IJCV/TIP) + required work estimate |

### Training Analysis *(v2.0.0)*
| Tool | Description |
|------|-------------|
| `design_ablation_study` | Ranked ablation components + paper table Markdown |
| `interpret_training_log` | Paste any log → OVERFITTING / UNDERFITTING / PLATEAU / INSTABILITY |
| `estimate_kaggle_feasibility` | Memory + hours estimate, risk level, recommended GPU env |
| `suggest_ensemble_strategy` | Diversity score, diversity-maximising ensemble recipe |
| `identify_hard_samples` | Known hard cases per task + targeted augmentations |
| `generate_hypothesis_test_plan` | Cohen's h, required N, bootstrap recipe for stat significance |

---

## 📁 Repository Structure

```
kaggle-mcp-research/
├── src/
│   └── kaggle_mcp/
│       ├── server.py              ← FastMCP server (55+ tools)
│       ├── config.py              ← Token loading from env vars
│       ├── __init__.py
│       ├── __main__.py            ← python -m kaggle_mcp
│       └── tools/
│           ├── kaggle.py          ← All Kaggle API operations
│           ├── huggingface.py     ← HuggingFace Hub operations
│           ├── research.py        ← arXiv, S2, Papers With Code
│           ├── notebook.py        ← Notebook generator templates
│           ├── zip_processor.py   ← ZIP/directory analyzer
│           ├── evaluation.py      ← Brutal eval + academic review (v2)
│           ├── deep_research.py   ← Method evolution + gap analysis (v2)
│           └── analysis.py        ← Training diagnostics + ablation (v2)
├── tests/
│   ├── conftest.py                ← Shared fixtures
│   ├── test_credentials.py        ← API token validation
│   ├── test_evaluation_tools.py   ← 40+ evaluation tests
│   ├── test_deep_research.py      ← 35+ deep research tests
│   └── test_analysis_and_notebook.py ← 40+ analysis + notebook tests
├── .github/
│   ├── copilot-instructions.md    ← 8-phase pipeline instructions
│   └── workflows/
│       ├── ci.yml                 ← GitHub Actions CI (Python 3.11+3.12)
│       └── publish.yml            ← Auto-publish to PyPI on v* tag
├── .vscode/
│   └── mcp.json                  ← VS Code MCP config
├── Makefile                      ← install / test / lint / build / publish
├── pyproject.toml
├── .env.example
└── README.md
```

---

## 🧪 Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests (requires KAGGLE_TOKEN for live API tests)
KAGGLE_TOKEN=KGAT_xxx HF_TOKEN=hf_xxx pytest tests/ -v --tb=short

# Unit-only, no live API calls
pytest tests/test_evaluation_tools.py tests/test_analysis_and_notebook.py -v
```

Tip: the evaluation and analysis tests are fully offline — no API tokens needed.

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

**Ideas for v3:**
- W&B / MLflow experiment tracking tools
- Automated hyperparameter search (Optuna) report tool
- Domain-specific templates (medical imaging, satellite, multi-modal)
- RAG over PDF research papers
- Real-time leaderboard monitoring with delta alerts
- AutoML baseline generator for tabular competitions

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

Built with [FastMCP](https://github.com/jlowin/fastmcp) · [Kaggle API](https://github.com/Kaggle/kaggle-api) · [HuggingFace Hub](https://huggingface.co) · [Papers With Code](https://paperswithcode.com/api) · [Semantic Scholar](https://api.semanticscholar.org/) · [arXiv API](https://arxiv.org/help/api/)

---

<div align="center">

**Made with love for the research and Kaggle community**

⭐ Star this repo if it saves you GPU hours!

**v2.0.0** · 55+ tools · 8-phase pipeline · brutal evaluation · Q1 journal routing

</div>