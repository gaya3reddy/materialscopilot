# MaterialsCopilot

**Evidence-grounded RAG assistant for materials science papers.**

MaterialsCopilot lets you ask questions over uploaded research PDFs and get answers with a full citation trail — doc, page, chunk, and similarity score. No hallucinations, no black-box answers.

Supports two model providers: **OpenAI API** (faster, higher quality) and **Ollama** (fully local — no data leaves your machine).

> Research use only. Do not upload proprietary or confidential data.

---

## Architecture

**PDF ingest → chunking → embeddings → ChromaDB retrieval → cited answers / summaries → evidence audit trail**

---

## Tech stack

- **FastAPI** — API
- **Streamlit** — UI
- **pypdf** — PDF text extraction
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

DATA_DIR=data
MAX_UPLOAD_MB=30
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

## Notes

- `.env` is gitignored — never commit your API key.
- ChromaDB and the document registry persist to `data/` across restarts.
- PDF extraction quality depends on the source file; scanned PDFs without OCR will not index well.
- Ollama responses are slower (~20–60s on CPU) compared to OpenAI (~3–5s). For demos, OpenAI is recommended.
