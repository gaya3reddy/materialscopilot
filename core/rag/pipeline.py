from __future__ import annotations

from typing import Any, Dict, List

from openai import OpenAI

from apps.api.config import settings
from core.rag.prompts import ASK_SYSTEM
from core.retrieval.embedder import OpenAIEmbedder
from core.retrieval.vectorstore import ChromaVectorStore
from typing import Generator
import json


def _build_context(citations: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, c in enumerate(citations, start=1):
        meta = c["meta"]
        doc_id = meta.get("doc_id")
        page = meta.get("page")
        text = c["text"]
        blocks.append(f"[{i}] ({doc_id} p.{page})\n{text}")
    return "\n\n".join(blocks)


def _merge_and_topk(results: list[list[dict]], top_k: int) -> list[dict]:
    merged = []
    for r in results:
        merged.extend(r)
    merged.sort(key=lambda x: x["distance"])  # smaller distance = better
    return merged[:top_k]


def answer_question(
    question: str,
    top_k: int = 5,
    doc_ids: list[str] | None = None,
    mode: str = "rag",
) -> Dict[str, Any]:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY missing. Set it in .env.")

    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key, model=settings.openai_embed_model
    )
    store = ChromaVectorStore(
        persist_dir=str(settings.processed_dir / "chroma"), embedder=embedder
    )

    # --- NEW: retrieval logic ---
    retrieved: list[dict] = []
    if mode == "no_rag":
        system_prompt = (
            "You are a helpful medical assistant. Answer from your general knowledge."
        )
        user_prompt = question
        retrieved = []
    else:
        system_prompt = ASK_SYSTEM
        doc_ids = doc_ids or []
        if len(doc_ids) == 0:
            retrieved = store.query(question=question, top_k=top_k, doc_id=None)
        elif len(doc_ids) == 1:
            retrieved = store.query(question=question, top_k=top_k, doc_id=doc_ids[0])
        else:
            per_doc = [
                store.query(question=question, top_k=top_k, doc_id=did)
                for did in doc_ids
            ]
            retrieved = _merge_and_topk(per_doc, top_k=top_k)

    context = _build_context(retrieved)
    user_prompt = f"""Question: {question}

Guideline excerpts:
{context}
"""

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer = resp.choices[0].message.content.strip()
    return {"answer": answer, "citations": retrieved}


def stream_answer(
    question: str,
    top_k: int = 5,
    doc_ids: list[str] | None = None,
    mode: str = "rag",
) -> Generator[str, None, None]:
    """Yields answer tokens one by one, then yields citations as a JSON line."""

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY missing.")

    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key, model=settings.openai_embed_model
    )
    store = ChromaVectorStore(
        persist_dir=str(settings.processed_dir / "chroma"), embedder=embedder
    )

    retrieved: list[dict] = []
    if mode != "no_rag":
        doc_ids = doc_ids or []
        if len(doc_ids) == 0:
            retrieved = store.query(question=question, top_k=top_k, doc_id=None)
        elif len(doc_ids) == 1:
            retrieved = store.query(question=question, top_k=top_k, doc_id=doc_ids[0])
        else:
            per_doc = [
                store.query(question=question, top_k=top_k, doc_id=did)
                for did in doc_ids
            ]
            retrieved = _merge_and_topk(per_doc, top_k=top_k)

    context = _build_context(retrieved)

    if mode == "no_rag":
        system_prompt = (
            "You are a helpful medical assistant. Answer from your general knowledge."
        )
        user_prompt = question
    else:
        system_prompt = ASK_SYSTEM
        user_prompt = f"Question: {question}\n\nGuideline excerpts:\n{context}"

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        stream=True,  # enable streaming
    )

    # Stream answer tokens
    for chunk in resp:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

    # After stream ends, yield citations as a single JSON line
    yield "\n\n__CITATIONS__:" + json.dumps(retrieved)


# def _summarize_retrieval_query(style: str) -> str:
#     style = (style or "tldr").lower()
#     if style == "key_steps":
#         return "key steps recommendations algorithm workflow what to do"
#     if style == "contraindications":
#         return "contraindications warnings precautions adverse events do not use avoid"
#     if style == "eligibility":
#         return "eligibility criteria inclusion exclusion who qualifies indications"
#     return "summary overview key recommendations purpose scope main findings"
def _summarize_retrieval_query(style: str, title: str | None = None) -> str:
    style = (style or "tldr").lower()
    prefix = f"{title} " if title else ""
    
    if style == "methods":
        return f"{prefix}experimental methods synthesis characterization procedure"
    if style == "key_findings":
        return f"{prefix}key findings results conclusions main outcomes performance"
    if style == "materials_properties":
        return f"{prefix}mechanical thermal electrical optical properties composition microstructure"
    return f"{prefix}summary overview purpose scope main findings"

    # if style == "key_steps":
    #     return f"{prefix}key steps recommendations algorithm workflow what to do"
    # if style == "contraindications":
    #     return f"{prefix}contraindications warnings precautions adverse events do not use avoid"
    # if style == "eligibility":
    #     return f"{prefix}eligibility criteria inclusion exclusion who qualifies indications"
    # return f"{prefix}summary overview key recommendations purpose scope"


def _summarize_user_prompt(style: str, context: str) -> str:
    style = (style or "tldr").lower()

#     if style == "key_steps":
#         instructions = """Create a concise, step-by-step summary of the guideline:
# - 6–12 bullet steps, in order
# - include any thresholds / timing / decision points if present
# - be specific and actionable
# """
#     elif style == "contraindications":
#         instructions = """Create a focused summary of contraindications / warnings:
# - bullet list grouped by theme (e.g., meds, conditions, situations)
# - include "Do not..." / "Avoid..." / "Use caution..." wording when appropriate
# - keep it short and clinically readable
# """
#     elif style == "eligibility":
#         instructions = """Create a focused eligibility summary:
# - "Eligible if" bullet list
# - "Not eligible if" bullet list
# - include any numeric thresholds if present
# """
#     else:
#         instructions = """Create a TL;DR summary:
# - 6–10 bullets max
# - include purpose + key recommendations + any critical warnings
# - keep it non-hallucinated and grounded in excerpts
# """
    if style == "methods":
        instructions = """Summarize the experimental methods section:
    - synthesis/fabrication steps in order
    - characterization techniques used
    - key parameters (temperature, pressure, composition, etc.)
    """
    elif style == "key_findings":
        instructions = """Summarize the key findings and results:
    - main outcomes as bullet points
    - include quantitative results and performance metrics where present
    - note any comparisons to prior work
    """
    elif style == "materials_properties":
        instructions = """Summarize the materials properties reported:
    - list properties by category (mechanical, thermal, electrical, etc.)
    - include values and units where present
    - note measurement conditions if given
    """
    else:  # tldr
        instructions = """Create a TL;DR summary:
    - 6–10 bullets max
    - include: material studied, method, key result, and significance
    - stay grounded in the excerpts only
    - avoid hallucination or adding information not in the excerpts"""

    return f"""{instructions}

Guideline excerpts (cite internally by referring to the numbered blocks):
{context}
"""


def summarize_guideline(
    style: str = "tldr",
    doc_ids: list[str] | None = None,
    top_k: int = 8,
    mode: str = "rag",
) -> Dict[str, Any]:
    """
    Returns: {"summary": <text>, "citations": <retrieved chunks list>}
    """
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY missing. Set it in .env.")

    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key, model=settings.openai_embed_model
    )
    store = ChromaVectorStore(
        persist_dir=str(settings.processed_dir / "chroma"), embedder=embedder
    )

    retrieved: list[dict] = []
    if mode == "no_rag":
        retrieved = []
    else:
        # Look up title from ChromaDB metadata
        doc_title = None
        if doc_ids and len(doc_ids) == 1:
            temp_results = store.query(
                question="title purpose scope", top_k=1, doc_id=doc_ids[0]
            )
            if temp_results:
                doc_title = temp_results[0]["meta"].get("title")
        query = _summarize_retrieval_query(style, title=doc_title)
        doc_ids = doc_ids or []
        if len(doc_ids) == 0:
            retrieved = store.query(question=query, top_k=top_k, doc_id=None)
        elif len(doc_ids) == 1:
            retrieved = store.query(question=query, top_k=top_k, doc_id=doc_ids[0])
        else:
            per_doc = [
                store.query(question=query, top_k=top_k, doc_id=did) for did in doc_ids
            ]
            retrieved = _merge_and_topk(per_doc, top_k=top_k)

    context = _build_context(retrieved)

    # separate system prompt for summarization (simpler + safer)
    summarize_system = (
        "You are a helpful assistant summarizing materials science paper excerpts. "
        "Use ONLY the provided excerpts. If information is missing, say so."
    )
    user_prompt = _summarize_user_prompt(style=style, context=context)

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": summarize_system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    summary = resp.choices[0].message.content.strip()
    return {"summary": summary, "citations": retrieved}
