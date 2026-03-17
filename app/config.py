"""
Configuración centralizada — lee valores de variables de entorno.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Settings:
    # ── Servidor ─────────────────────────────────────────────────────────
    PORT: int = int(os.getenv("PORT", "8000"))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/tmp/uploads")
    CORS_ORIGINS: List[str] = field(default_factory=lambda: [
        os.getenv("CORS_ORIGIN", "*")
    ])

    # ── Google Cloud / Vertex AI ────────────────────────────────────────
    GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "")
    GCP_LOCATION: str = os.getenv("GCP_LOCATION", "us-central1")

    # Si usas Service Account en Render, pon la ruta al JSON:
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

    # ── Gemini (LLM vía LangChain) ──────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # ── Vertex AI RAG Engine ────────────────────────────────────────────
    RAG_EMBEDDING_MODEL: str = os.getenv(
        "RAG_EMBEDDING_MODEL",
        "publishers/google/models/text-embedding-005"
    )
    RAG_CORPUS_DISPLAY_NAME: str = os.getenv("RAG_CORPUS_DISPLAY_NAME", "profilexai_kb")
    RAG_SIMILARITY_TOP_K: int = int(os.getenv("RAG_SIMILARITY_TOP_K", "5"))
    RAG_VECTOR_DISTANCE_THRESHOLD: float = float(os.getenv("RAG_VECTOR_DISTANCE_THRESHOLD", "0.3"))
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "512"))
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))

    # ── XAI ─────────────────────────────────────────────────────────────
    SHAP_BACKGROUND_SIZE: int = int(os.getenv("SHAP_BACKGROUND_SIZE", "50"))
    SHAP_NSAMPLES: int = int(os.getenv("SHAP_NSAMPLES", "15"))


settings = Settings()
