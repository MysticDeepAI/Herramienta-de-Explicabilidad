"""
RAG Engine — Vertex AI RAG Engine para ingesta/retrieval + LangChain + Gemini para generación.

Flujo:
  1. Vertex AI RAG Engine crea un corpus y maneja chunking + embeddings + vector DB
  2. Subimos archivos al corpus → Google se encarga de fragmentar e indexar
  3. Recuperamos contexto relevante con retrieveContexts API
  4. LangChain + Gemini 2.5 genera narrativas adaptadas al perfil
  5. Chat interactivo con memoria gestionado por LangChain
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Dict, List, Optional

import vertexai
from vertexai import rag

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage

from app.config import settings

logger = logging.getLogger(__name__)

# ── Prompts por perfil ───────────────────────────────────────────────────

PROFILE_PROMPTS = {
    "data-scientist": (
        "You are an expert AI assistant speaking to a Machine Learning engineer.\n"
        "Use precise technical terminology, reference model internals (feature weights, "
        "attributions, metrics), and include quantitative details.\n\n"
        "Context from knowledge base:\n{context}\n\n"
        "Explanation data:\n{explanation}\n\n"
        "Question: {question}\n\n"
        "Provide a thorough technical explanation."
    ),
    "domain-expert": (
        "You are a helpful AI assistant speaking to a domain expert (e.g., a doctor "
        "or engineer). Translate ML concepts into domain-specific terminology. "
        "Avoid raw model jargon but keep professional depth.\n\n"
        "Context from knowledge base:\n{context}\n\n"
        "Explanation data:\n{explanation}\n\n"
        "Question: {question}\n\n"
        "Provide a clear domain-oriented explanation."
    ),
    "non-expert": (
        "You are a friendly AI assistant explaining results to someone without "
        "technical background. Use simple analogies, everyday language, and avoid "
        "jargon. Be concise and reassuring.\n\n"
        "Context from knowledge base:\n{context}\n\n"
        "Explanation data:\n{explanation}\n\n"
        "Question: {question}\n\n"
        "Explain this in simple, easy-to-understand terms."
    ),
}

CHAT_SYSTEM_PROMPT = (
    "You are ProfileXAI, an AI assistant that helps users understand ML model "
    "predictions. You have access to a knowledge base about the domain. "
    "Answer follow-up questions clearly, adapting your language to the user's "
    "profile level: {profile}.\n\n"
    "Context from knowledge base:\n{context}\n\n"
    "Previous conversation:\n{history}\n\n"
    "User question: {question}\n\n"
    "Provide a helpful, grounded response."
)


class RAGEngine:
    """
    Motor RAG que usa Vertex AI RAG Engine para ingesta/retrieval
    y LangChain + Gemini para generación de narrativas y chat.
    """

    def __init__(self):
        # ── Inicializar Vertex AI ────────────────────────────────────────
        if not settings.GCP_PROJECT_ID:
            raise ValueError("GCP_PROJECT_ID no configurado")

        # Si hay credenciales de Service Account, configurarlas
        if settings.GOOGLE_APPLICATION_CREDENTIALS:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS

        vertexai.init(
            project=settings.GCP_PROJECT_ID,
            location=settings.GCP_LOCATION,
        )

        # ── LangChain LLM (Gemini) ──────────────────────────────────────
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.3,
        )

        # ── Estado ───────────────────────────────────────────────────────
        self.corpus = None
        self.corpus_name: Optional[str] = None
        # Memoria de chat por perfil
        self._chat_memories: Dict[str, ConversationBufferWindowMemory] = {}

    # ────────────────────────────────────────────────────────────────────
    # 1. CORPUS — Creación (Vertex AI RAG Engine se encarga de la DB)
    # ────────────────────────────────────────────────────────────────────

    def _ensure_corpus(self) -> None:
        """Crea el corpus si no existe, o reutiliza uno existente."""
        if self.corpus is not None:
            return

        # Buscar corpus existente con el mismo nombre
        try:
            existing = rag.list_corpora()
            for c in existing:
                if c.display_name == settings.RAG_CORPUS_DISPLAY_NAME:
                    self.corpus = c
                    self.corpus_name = c.name
                    logger.info("Corpus existente reutilizado: %s", c.name)
                    return
        except Exception as e:
            logger.warning("Error listando corpora: %s", e)

        # Crear nuevo corpus
        embedding_config = rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model=settings.RAG_EMBEDDING_MODEL,
            )
        )

        self.corpus = rag.create_corpus(
            display_name=settings.RAG_CORPUS_DISPLAY_NAME,
            description="ProfileXAI knowledge base corpus",
            backend_config=rag.RagVectorDbConfig(
                rag_embedding_model_config=embedding_config,
            ),
        )
        self.corpus_name = self.corpus.name
        logger.info("Corpus creado: %s", self.corpus_name)

    # ────────────────────────────────────────────────────────────────────
    # 2. INGESTA — Subir archivos (Vertex AI hace chunking + embedding)
    # ────────────────────────────────────────────────────────────────────

    def ingest(self, file_paths: List[str]) -> int:
        """
        Sube archivos al corpus de Vertex AI RAG Engine.
        Google se encarga del chunking, embedding e indexación.
        Retorna la cantidad de archivos procesados.
        """
        self._ensure_corpus()

        count = 0
        for path in file_paths:
            try:
                # Configuración de chunking
                chunk_config = rag.ChunkingConfig(
                    chunk_size=settings.RAG_CHUNK_SIZE,
                    chunk_overlap=settings.RAG_CHUNK_OVERLAP,
                )
                transformation_config = rag.TransformationConfig(
                    chunking_config=chunk_config,
                )

                rag.upload_file(
                    corpus_name=self.corpus_name,
                    path=path,
                    display_name=os.path.basename(path),
                    description=f"KB file: {os.path.basename(path)}",
                    transformation_config=transformation_config,
                )
                count += 1
                logger.info("Archivo subido al corpus: %s", os.path.basename(path))
            except Exception as e:
                logger.warning("Error subiendo %s: %s", path, e)

        return count

    # ────────────────────────────────────────────────────────────────────
    # 3. RETRIEVAL — Recuperar contexto (Vertex AI RAG Engine)
    # ────────────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Recupera los fragmentos más relevantes del corpus.
        Vertex AI se encarga de la búsqueda vectorial.
        """
        if self.corpus_name is None:
            return []

        k = top_k or settings.RAG_SIMILARITY_TOP_K

        try:
            response = rag.retrieval_query(
                rag_resources=[
                    rag.RagResource(rag_corpus=self.corpus_name)
                ],
                text=query,
                similarity_top_k=k,
                vector_distance_threshold=settings.RAG_VECTOR_DISTANCE_THRESHOLD,
            )

            chunks = []
            if response and response.contexts and response.contexts.contexts:
                for ctx in response.contexts.contexts:
                    if ctx.text:
                        chunks.append(ctx.text)

            logger.info("Recuperados %d chunks para query", len(chunks))
            return chunks

        except Exception as e:
            logger.warning("Error en retrieval: %s", e)
            return []

    # ────────────────────────────────────────────────────────────────────
    # 4. GENERACIÓN — Narrativa con LangChain + Gemini
    # ────────────────────────────────────────────────────────────────────

    def generate_narrative(
        self,
        explanation_data: dict,
        profile: str = "non-expert",
        question: Optional[str] = None,
    ) -> str:
        """
        Genera una narrativa en lenguaje natural adaptada al perfil,
        usando contexto recuperado de Vertex AI RAG Engine
        y generación con LangChain + Gemini.
        """
        explanation_str = json.dumps(explanation_data, indent=2, ensure_ascii=False)

        # Recuperar contexto del corpus
        query = question or f"Explain prediction: {explanation_data.get('prediction', '')}"
        context_chunks = self.retrieve(query)
        context = "\n---\n".join(context_chunks) if context_chunks else "No knowledge base context available."

        # Seleccionar prompt por perfil
        template = PROFILE_PROMPTS.get(profile, PROFILE_PROMPTS["non-expert"])
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "explanation", "question"],
        )

        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "explanation": explanation_str,
            "question": question or "Explain this prediction to me.",
        })

        return response.content

    # ────────────────────────────────────────────────────────────────────
    # 5. CHAT — Conversacional con LangChain + memoria + RAG retrieval
    # ────────────────────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        profile: str = "non-expert",
        history: Optional[List[dict]] = None,
    ) -> dict:
        """
        Chat interactivo: recupera contexto de Vertex AI RAG Engine,
        gestiona la memoria con LangChain y genera respuesta con Gemini.
        """
        # Recuperar contexto relevante del corpus
        context_chunks = self.retrieve(message)
        context = "\n---\n".join(context_chunks) if context_chunks else "No additional context available."

        # Construir historial para el prompt
        history_text = ""
        if history:
            for msg in history[-10:]:  # últimos 10 mensajes
                role = "User" if msg.get("role") == "user" else "Assistant"
                history_text += f"{role}: {msg.get('content', '')}\n"

        # Prompt con contexto + historial + perfil
        prompt = PromptTemplate(
            template=CHAT_SYSTEM_PROMPT,
            input_variables=["profile", "context", "history", "question"],
        )

        chain = prompt | self.llm
        response = chain.invoke({
            "profile": profile,
            "context": context,
            "history": history_text,
            "question": message,
        })

        # Extraer fuentes
        sources = [chunk[:200] for chunk in context_chunks[:3]]

        return {
            "response": response.content,
            "sources": sources,
        }

    # ────────────────────────────────────────────────────────────────────
    # Cleanup
    # ────────────────────────────────────────────────────────────────────

    def delete_corpus(self) -> None:
        """Elimina el corpus de Vertex AI (limpieza)."""
        if self.corpus_name:
            try:
                rag.delete_corpus(name=self.corpus_name)
                logger.info("Corpus eliminado: %s", self.corpus_name)
                self.corpus = None
                self.corpus_name = None
            except Exception as e:
                logger.warning("Error eliminando corpus: %s", e)
