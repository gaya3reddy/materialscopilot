# MaterialsCopilot

**Evidence-grounded RAG assistant for materials science papers.**

MaterialsCopilot lets you ask questions over uploaded research PDFs and get answers with a full citation trail — doc, page, chunk, and similarity score. No hallucinations, no black-box answers.

> Research use only. Do not upload proprietary or confidential data.

---

## Architecture

**PDF ingest → chunking → embeddings → ChromaDB retrieval → cited answers / summaries → evidence audit trail**

---

## Tech stack

- **FastAPI** — API
- **Streamlit** — UI
- **pypdf** — PDF text extraction
- **OpenAI** — chat completions + embeddings
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

MODEL_PROVIDER=openai
OPENAI_API_KEY=YOUR_KEY_HERE
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small

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
