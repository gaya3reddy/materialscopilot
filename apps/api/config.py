from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"  # project root/.env
load_dotenv(dotenv_path=ENV_PATH)


def _get_env(name: str, default: str | None = None) -> str:
    val = os.getenv(name, default)
    if val is None or val == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return val


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str

    api_host: str
    api_port: int

    model_provider: str
    openai_api_key: str | None
    openai_chat_model: str
    openai_embed_model: str
    
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embed_model: str

    data_dir: Path
    raw_dir: Path
    processed_dir: Path

    max_upload_mb: int

    @staticmethod
    def load() -> "Settings":
        app_name = os.getenv("APP_NAME", "materialscopilot-api")
        app_version = os.getenv("APP_VERSION", "0.1.0")

        api_host = os.getenv("API_HOST", "0.0.0.0")
        api_port = int(os.getenv("API_PORT", "8000"))

        model_provider = os.getenv("MODEL_PROVIDER", "openai").lower()

        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        openai_embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_chat_model = os.getenv("OLLAMA_CHAT_MODEL", "mistral")
        ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        # ollama_chat_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3")

        data_dir = Path(os.getenv("DATA_DIR", "data")).resolve()
        raw_dir = data_dir / "raw"
        processed_dir = data_dir / "processed"

        max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "30"))

        # Make dirs
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        chroma_dir = processed_dir / "chroma"
        chroma_dir.mkdir(parents=True, exist_ok=True)

        if model_provider == "openai" and (
            openai_api_key is None or openai_api_key.strip() == ""
        ):
            # We allow running without key for Day-1 skeleton (health endpoint),
            # but warn via logs later.
            openai_api_key = None
        print("DEBUG OPENAI_API_KEY loaded:", bool(openai_api_key))
        return Settings(
            app_name=app_name,
            app_version=app_version,
            api_host=api_host,
            api_port=api_port,
            model_provider=model_provider,
            openai_api_key=openai_api_key,
            openai_chat_model=openai_chat_model,
            openai_embed_model=openai_embed_model,
            ollama_base_url=ollama_base_url,
            ollama_chat_model=ollama_chat_model,
            ollama_embed_model=ollama_embed_model,
            data_dir=data_dir,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            max_upload_mb=max_upload_mb,
        )


settings = Settings.load()
