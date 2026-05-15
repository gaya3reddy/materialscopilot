from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, Optional


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    version: str
    provider: str = "openai"
    parser: str = "pypdf"


class Citation(BaseModel):
    doc_id: str
    page: int
    chunk_id: str
    snippet: str
    score: float = Field(ge=0.0, le=1.0)


class Meta(BaseModel):
    request_id: str
    latency_ms: int
    model: str
    prompt_version: str


class IngestResponse(BaseModel):
    doc_id: str
    chunks_indexed: int
    pages: int
    deduped: bool = False
    message: str | None = None


class DocInfo(BaseModel):
    doc_id: str
    title: str | None = None
    source: str | None = None
    category: str | None = None


class DocList(BaseModel):
    items: list[DocInfo]


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    doc_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: Literal["rag", "no_rag"] = "rag"


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    meta: Meta


class SummarizeRequest(BaseModel):
    doc_ids: list[str] = Field(default_factory=list)
    style: Literal["tldr", "methods", "key_findings", "materials_properties"] = "tldr"
    query: str | None = None


class SummarizeResponse(BaseModel):
    summary: str
    citations: list[Citation] = Field(default_factory=list)
    meta: Meta


class ErrorResponse(BaseModel):
    detail: str
    request_id: Optional[str] = None
