from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass
from typing import List

from pypdf import PdfReader
from core.logger import get_logger
logger = get_logger(__name__)


@dataclass
class PageText:
    page: int  # 1-based page number
    text: str


def _extract_pypdf(pdf_bytes: bytes) -> List[PageText]:
    """Reliable fallback — always works for text-based PDFs."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: List[PageText] = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        pages.append(PageText(page=i + 1, text=txt.strip()))
    return pages


def _extract_llamaparse(pdf_bytes: bytes) -> List[PageText]:
    try:
        from llama_parse import LlamaParse
    except ImportError:
        raise ValueError("llama-parse not installed. Run: pip install llama-parse")
    
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError("LLAMA_CLOUD_API_KEY not set")
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            tmp_path = f.name

        parser = LlamaParse(
            result_type="markdown",
            api_key=api_key,
        )
        documents = parser.load_data(tmp_path)

        if not documents:
            raise ValueError("LlamaParse returned empty documents")

        pages: List[PageText] = []
        for i, doc in enumerate(documents):
            text = doc.text.strip()
            if text:
                page_num = int(doc.metadata.get("page_label", i + 1))
                pages.append(PageText(page=page_num, text=text))

        if not pages:
            raise ValueError("LlamaParse produced no extractable text")

        return pages

    finally:
        # Clean up temp file — no reference to 'e' here
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

def extract_pages(pdf_bytes: bytes) -> List[PageText]:
    """
    Try LlamaParse first (better quality for scientific PDFs).
    Fall back to pypdf if LlamaParse fails for any reason —
    API key missing, network error, empty output, or parse error.
    """
    parser = os.getenv("PARSER", "pypdf").lower()

    if parser == "llamaparse":
        try:
            pages = _extract_llamaparse(pdf_bytes)
            logger.info("PDF parsed with LlamaParse ✓ — pages: %d", len(pages))
            return pages
        except Exception as e:
            logger.warning("LlamaParse failed (%s) — falling back to pypdf", e)
            return _extract_pypdf(pdf_bytes)

    return _extract_pypdf(pdf_bytes)