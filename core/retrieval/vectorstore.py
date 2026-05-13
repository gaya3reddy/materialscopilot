from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.retrieval.embedder import get_embedder


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str,
        embedder: Any,
        collection_name: str = "papers",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.col = self.client.get_or_create_collection(name=collection_name)
        self.embedder = embedder

    def upsert_chunks(
        self,
        doc_id: str,
        title: str | None,
        source: str | None,
        category: str | None,
        chunks: List[Dict[str, Any]],  # [{"id", "page", "text",}]
        batch_size: int = 50,
    ) -> int:
        ids = [f"{doc_id}:{c['id']}" for c in chunks]
        docs = [c["text"] for c in chunks]
        metas = [
            {
                "doc_id": doc_id,
                "chunk_id": c["id"],
                "page": int(c["page"]),
                "title": title,
                "source": source,
                "category": category,
            }
            for c in chunks
        ]
        # embs = self.embedder.embed(docs) # this can be slow example, 300 chunks, so we do it in batches
        # Embed + upsert in batches to avoid OpenAI rate limits on large PDFs
        total_indexed = 0
        for i in range(0, len(chunks), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_docs = docs[i : i + batch_size]  # batch of text = 50 chunks
            batch_metas = metas[i : i + batch_size]

            batch_embs = self.embedder.embed(
                batch_docs
            )  # Doing this in batches to avoid rate limits
            self.col.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=batch_embs,
            )
            total_indexed += len(batch_ids)
        return total_indexed

    def query(
        self,
        question: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        q_emb = self.embedder.embed([question])[0]
        where = {"doc_id": doc_id} if doc_id else None

        res = self.col.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        out: List[Dict[str, Any]] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            out.append(
                {
                    "text": doc,
                    "meta": meta,
                    "distance": float(dist),
                }
            )
        return out
