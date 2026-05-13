from __future__ import annotations

from typing import List, Optional

from openai import OpenAI


class OpenAIEmbedder:
    def __init__(self, api_key: Optional[str], model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for embeddings.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        import requests
        embeddings = []
        for text in texts:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
        return embeddings


def get_embedder(settings):
    if settings.model_provider == "ollama":
        return OllamaEmbedder(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embed_model,
        )
    return OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.openai_embed_model,
    )