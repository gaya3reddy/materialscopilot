from __future__ import annotations

import hashlib
import uuid
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from apps.api.config import settings
from apps.api.job_registry import JobStatus, job_registry
from core.ingestion.chunker import chunk_pages
from core.ingestion.pdf_loader import extract_pages
from core.registry.registry import DocumentRegistry
# from core.retrieval.embedder import OpenAIEmbedder
from core.retrieval.embedder import get_embedder
from core.retrieval.vectorstore import ChromaVectorStore
from core.schemas.models import DocInfo, DocList
import asyncio

router = APIRouter(tags=["ingestion"])

registry = DocumentRegistry(settings.processed_dir / "registry.json")


@router.get("/documents", response_model=DocList)
def list_docs() -> DocList:
    return DocList(items=registry.all())


@router.get("/ingest/status/{job_id}")
def ingest_status(job_id: str):
    job = job_registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job.job_id,
        "doc_id": job.doc_id,
        "status": job.status,
        "progress": job.progress,
        "pages": job.pages,
        "indexed_chunks": job.indexed_chunks,
        "total_chunks": job.total_chunks,
        "error": job.error,
        "message": job.message,
    }


def _run_ingest(
    job_id: str,
    doc_id: str,
    data: bytes,
    title: str | None,
    source: str | None,
    category: str | None,
) -> None:
    """Background worker — runs in a thread."""
    try:
        job_registry.update(job_id, status=JobStatus.PROCESSING)

        pages = extract_pages(data)
        page_pairs = [(p.page, p.text) for p in pages]
        chunks = chunk_pages(page_pairs)

        job_registry.update(
            job_id,
            total_chunks=len(chunks),
            pages=len(pages),
        )

        embedder = get_embedder(settings)
        store = ChromaVectorStore(
            persist_dir=str(settings.processed_dir / "chroma"),
            embedder=embedder,
        )

        # Batch embed with progress updates
        BATCH_SIZE = 50
        indexed = 0
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            store.upsert_chunks(
                doc_id=doc_id,
                title=title,
                source=source,
                category=category,
                chunks=[
                    {"id": c.chunk_id, "text": c.text, "page": c.page} for c in batch
                ],
            )
            indexed += len(batch)
            job_registry.update(job_id, indexed_chunks=indexed)

        job_registry.set_done(job_id, pages=len(pages), chunks=indexed)

    except Exception as e:
        job_registry.set_error(job_id, str(e))


@router.post("/ingest", status_code=202)
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_id: str | None = Form(default=None),
    title: str | None = Form(default=None),
    source: str | None = Form(default=None),
    category: str | None = Form(default=None),
) -> JSONResponse:
    if file.content_type not in ("application/pdf",):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    data = await file.read()
    file_hash = hashlib.sha256(data).hexdigest()

    # Deduplication
    existing_id = registry.get_by_hash(file_hash)
    if existing_id:
        return JSONResponse(
            status_code=200,
            content={
                "job_id": None,
                "doc_id": existing_id,
                "deduped": True,
                "message": "Duplicate PDF detected — returning existing doc_id.",
            },
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {settings.max_upload_mb} MB.",
        )

    safe_doc_id = (doc_id or "").strip()
    if not safe_doc_id or safe_doc_id.lower() == "string":
        safe_doc_id = f"doc_{uuid.uuid4().hex[:8]}"

    if registry.exists(safe_doc_id):
        raise HTTPException(
            status_code=400,
            detail=f"doc_id '{safe_doc_id}' already exists.",
        )

    # Save raw PDF
    out_path = settings.raw_dir / f"{safe_doc_id}.pdf"
    # out_path.write_bytes(data)
    await asyncio.get_event_loop().run_in_executor(None, out_path.write_bytes, data)

    # Register metadata immediately
    doc_info = DocInfo(
        doc_id=safe_doc_id,
        title=title,
        source=source,
        category=category,
    )
    registry.add(doc_info, file_hash)

    # Create job + kick off background task
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job_registry.create(job_id=job_id, doc_id=safe_doc_id)

    background_tasks.add_task(
        _run_ingest, job_id, safe_doc_id, data, title, source, category
    )

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "doc_id": safe_doc_id,
            "deduped": False,
            "message": "Ingest started. Poll /ingest/status/{job_id} for progress.",
        },
    )
