# MaterialsCopilot

**Evidence-grounded RAG assistant for materials science papers.**

MaterialsCopilot lets you ask questions over uploaded research PDFs and get answers with a full citation trail — doc, page, chunk, and similarity score. No hallucinations, no black-box answers.

Supports two model providers: **OpenAI API** (faster, higher quality) and **Ollama** (fully local — no data leaves your machine).  
Supports two PDF parsers: **pypdf** (default, reliable) and **LlamaParse** (better table/equation handling, with pypdf fallback).

> Research use only. Do not upload proprietary or confidential data.

---

## Architecture

**PDF ingest → chunking → embeddings → ChromaDB retrieval → cited answers / summaries → evidence audit trail**

---

## Tech stack

- **FastAPI** — API
- **Streamlit** — UI
- **pypdf** — PDF text extraction (default)
- **LlamaParse** — cloud PDF parser for scientific documents (optional, falls back to pypdf)
- **OpenAI** — chat completions + embeddings (cloud)
- **Ollama** — local model runtime (privacy mode)
- **ChromaDB** — persistent vector store
- **Docker / Docker Compose** — runtime

---

## Quickstart

### Option A — Docker (recommended)

**1. Create `.env` in the project root**

```env
APP_NAME=materialscopilot-api
APP_VERSION=0.1.0
API_HOST=0.0.0.0
API_PORT=8000

# Model provider: "openai" or "ollama"
MODEL_PROVIDER=openai

# OpenAI (used when MODEL_PROVIDER=openai)
OPENAI_API_KEY=YOUR_KEY_HERE
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small

# Ollama (used when MODEL_PROVIDER=ollama)
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_CHAT_MODEL=mistral
OLLAMA_EMBED_MODEL=nomic-embed-text

# PDF parser: "pypdf" (default) or "llamaparse"
PARSER=pypdf
# LLAMA_CLOUD_API_KEY=llx-your-key-here  # only needed when PARSER=llamaparse

DATA_DIR=data
MAX_UPLOAD_MB=30

# Logging: DEBUG | INFO | WARNING
LOG_LEVEL=INFO
```

**2. Build and run**

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| UI | http://localhost:8501 |
| API docs | http://localhost:8000/docs |

```bash
docker compose down   # to stop
```

---

### Option B — Local (venv)

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -U pip
pip install -e ".[dev]"

uvicorn apps.api.main:app --reload --port 8000
streamlit run apps/ui/Home.py
```

---

### Option C — Fully local (Ollama, no data leaves your machine)

**1. Install Ollama** from [ollama.com](https://ollama.com/download)

**2. Pull the required models**

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

**3. Set `.env`**

```env
MODEL_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_CHAT_MODEL=mistral
OLLAMA_EMBED_MODEL=nomic-embed-text
```

**4. Run**

```bash
docker compose up --build
```

The sidebar will show **🔒 Local model (Ollama) — no data leaves this machine**.

> Note: re-ingest documents when switching providers — embedding dimensions differ between OpenAI (1536) and Ollama/nomic (768).

---

## Demo flow

1. **Upload** — ingest a materials science paper PDF
2. **Ask** — select the doc, ask a question, get a cited answer
3. **Evidence** — inspect retrieved chunks, pages, and similarity scores
4. **Summarize** — pick a mode (`tldr`, `methods`, `key_findings`, `materials_properties`)

---

## API

### `POST /ingest`
Upload a PDF. Returns `doc_id`, page count, chunks indexed, and dedup status.

### `GET /documents`
List all indexed documents.

### `POST /ask`
```json
{
  "question": "What is the yield strength of the alloy?",
  "doc_ids": ["doc_1234abcd"],
  "top_k": 5,
  "mode": "rag"
}
```
Returns `answer` + `citations[]` (doc_id, page, chunk_id, snippet, score).

### `POST /summarize`
```json
{
  "doc_ids": ["doc_1234abcd"],
  "style": "key_findings"
}
```
Styles: `tldr` · `methods` · `key_findings` · `materials_properties`

Returns `summary` + `citations[]`.

---

## Logging

Structured logs written to stdout via Python's `logging` module. Each line includes timestamp, level, module, and message:

```
2026-05-15 14:02:11 | INFO     | apps.api.config         | MODEL_PROVIDER: openai
2026-05-15 14:02:11 | INFO     | apps.api.config         | PARSER: pypdf
2026-05-15 14:02:11 | INFO     | apps.api.routers.ingest | Ingest started — job: job_abc doc: doc_123
2026-05-15 14:02:18 | INFO     | apps.api.routers.ingest | Ingest complete — job: job_abc chunks: 312
2026-05-15 14:03:45 | INFO     | core.rag.pipeline       | Query received — provider: openai mode: rag top_k: 5
```

View live:
```bash
docker compose logs api --follow
```

Filter errors only:
```bash
docker compose logs api --tail=100 | grep ERROR
```

Control verbosity via `LOG_LEVEL` in `.env`: `DEBUG` · `INFO` (default) · `WARNING`

---

## PDF Parsing

Two parsers supported, toggled via `PARSER` in `.env`:

| Parser | Quality | Privacy | Cost |
|--------|---------|---------|------|
| `pypdf` (default) | Good for clean text PDFs | Local — no data leaves | Free |
| `llamaparse` | Better tables, equations, multi-column | Cloud — data sent to LlamaCloud | Paid API |

When `PARSER=llamaparse`, the system tries LlamaParse first and automatically falls back to pypdf if it fails (missing API key, network error, or empty output). The active parser is shown in the sidebar.

---

## Roadmap

- [ ] SPECTER2 / MatBERT domain-specific embedding models
- [ ] DocLing integration for fully local scientific PDF parsing
- [ ] LLaVA figure captioning at ingest time
- [ ] Hybrid search (BM25 + vector) for exact term matching
- [ ] RAGAS evaluation for faithfulness and answer relevancy
- [ ] LangGraph agentic pipeline for multi-step reasoning

---

## Notes

- `.env` is gitignored — never commit your API key.
- ChromaDB and the document registry persist to `data/` across restarts.
- PDF extraction quality depends on the source file; scanned PDFs without OCR will not index well.
- Ollama responses are slower (~20–60s on CPU) compared to OpenAI (~3–5s). GPU deployment reduces this to 2–5s. For demos, OpenAI is recommended.
- Re-ingest all documents when switching between OpenAI and Ollama — embedding dimensions are incompatible (1536 vs 768).
