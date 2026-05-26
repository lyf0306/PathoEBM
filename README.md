# PathoEBM

AI-powered evidence-based medicine (EBM) pipeline for gynecologic oncology. Takes an MDT treatment plan, performs multi-iteration deep literature search against PubMed and ClinicalTrials.gov, and produces an evidence-validated final report with gap-closing citations.

## Features

- **Auto-hybrid model routing** — external APIs (DeepSeek V4, GPT-4.1 mini) with automatic fallback to local vLLM
- **Structured comorbidity classification** — keyword-based tiering (critical infections / major comorbidities / incidental findings) + LLM safety-net re-classification
- **Multi-iteration deep search** — iterative PubMed + ClinicalTrials.gov retrieval with FDA drug label lookups
- **Evidence-anchored report generation** — merges baseline references with new evidence citations, auto-strips preamble and depersonalizes
- **REST API with job management** — async job queue, progress tracking, cancellation, 30-day auto-cleanup
- **Concurrency infrastructure** — connection pooling, circuit breakers, rate limiters, graceful shutdown

## Project Structure

```
PathoEBM-main/
├── pyproject.toml
├── local_deep_research/         # Main package
│   ├── main.py                  # CLI entry point & pipeline orchestration
│   ├── config.py                # Config & LLM factory functions
│   ├── search_system.py         # AdvancedSearchSystem engine
│   ├── connect_mcp.py           # MCP tool server client
│   ├── tool_selector.py         # LLM-based tool selection
│   ├── tool_executor.py         # MCP tool executor
│   ├── _settings/               # Config files + prompt templates
│   ├── agents/                  # Specialized LLM agents
│   ├── api/                     # FastAPI REST server
│   ├── concurrency/             # Connection pools, rate limiters, etc.
│   ├── evaluation/              # Report quality evaluator
│   ├── pipeline/                # Search planner, knowledge processor
│   ├── prompts/                 # PromptManager (versioned prompt loader)
│   ├── skills/                  # NCCN followup & prognosis modules
│   ├── utilities/               # Shared utilities
│   └── tools/                   # Tool info & embedding caches
├── scripts/                     # Utility/debug scripts
│   ├── check_mcp.py             # MCP endpoint health probe
│   ├── debug_selector.py        # ToolSelector debug harness
│   ├── evaluate_local.py        # Batch evaluation harness
│   ├── score_evaluation_results.py  # LLM-as-judge benchmark scoring
│   ├── init_tools.py            # Tool whitelist initializer
│   └── fix_clinical_tools.py    # Clinical tool config repair
└── tests/                       # Test files
    ├── test_MCP.py              # MCP connectivity test
    ├── test_pubmed_direct.py    # Direct PubMed search test
    ├── test_pubmed_query.py     # PubMed query formatting test
    ├── test_pubmed_raw.py       # Raw PubMed response inspection
    └── test_gog0258_diagnosis.py  # GOG-0258 retrieval gap diagnosis
```

## Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- API keys for at least one of: DeepSeek, OpenAI, or a local vLLM instance
- MCP tool servers running (PubMed, ClinicalTrials.gov, FDA drug labels)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd PathoEBM-main

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Configuration

Create `local_deep_research/_settings/deploy_config.toml`:

```toml
[openai]
api_key = "sk-..."
base_url = "https://api.openai.com/v1"

[deepseek]
api_key = "sk-..."
base_url = "https://api.deepseek.com/v1"

[embedding]
api_key = "sk-..."
model = "text-embedding-3-small"

[local]
base_url = "http://localhost:8000/v1"
model_name = "qwen3-32b"

[mcp]
url = "http://localhost:8788"

[model]
provider = "auto"          # auto | deepseek | gpt | local
fallback = "local"         # local | none

[api]
api_key = ""               # optional: set to enable X-API-Key auth

[storage]
jobs_dir = "api/jobs_output"
retention_days = 30
```

## Usage

### CLI Mode

```bash
python -m local_deep_research.main
```

Follow the interactive prompts to paste a treatment plan (Markdown) or load from file. Choose model provider and iteration depth.

### API Server

```bash
uvicorn local_deep_research.api.app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (status, active jobs, uptime) |
| `POST` | `/jobs` | Submit a new EBM pipeline job |
| `GET` | `/jobs` | List all jobs (newest first) |
| `GET` | `/jobs/{job_id}` | Get job status + progress snapshot |
| `GET` | `/jobs/{job_id}/result` | Get completed job's final report |
| `DELETE` | `/jobs/{job_id}` | Cancel a running/pending job |

### Example: Submit a Job

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "treatment_context": "# MDT Report\n...",
    "model_choice": "auto",
    "max_iterations": 2
  }'
```

### Example: Get Results

```bash
curl http://localhost:8000/jobs/{job_id}/result \
  -H "X-API-Key: your-api-key"
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test
python tests/test_MCP.py
```

## Running Scripts

```bash
# Check MCP server connectivity
python scripts/check_mcp.py

# Debug tool selection
python scripts/debug_selector.py

# Run batch evaluation
python scripts/evaluate_local.py
```

## Dependencies

Core runtime dependencies:

- **LLM**: `langchain`, `langchain-openai`, `langchain-deepseek`, `openai`
- **MCP**: `mcp`, `langchain-mcp-adapters`
- **API**: `fastapi`, `uvicorn`, `pydantic`
- **Data**: `datasets`, `scikit-learn`, `networkx`
- **Utilities**: `pymupdf`, `python-docx`, `python-pptx`, `openpyxl`, `markdown2`

See `pyproject.toml` for the full list.

## License

Proprietary. All rights reserved.
