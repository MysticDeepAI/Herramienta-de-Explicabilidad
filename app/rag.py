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


#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI # Versión moderna
#from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage

from app.config import settings

logger = logging.getLogger(__name__)

# ── Prompts por perfil ───────────────────────────────────────────────────

PROFILE_PROMPTS = {
    "data-scientist": (
        "Eres un asistente experto en IA hablando con un ingeniero de Machine Learning.\n"
        "Usa terminología técnica precisa, haz referencia a los componentes internos del modelo "
        "(pesos de características, atribuciones, métricas) e incluye detalles cuantitativos.\n\n"
        "Contexto de la base de conocimientos:\n{context}\n\n"
        "Datos de la explicación:\n{explanation}\n\n"
        "Pregunta: {question}\n\n"
        "Proporciona una explicación técnica en ESPAÑOL. "
        "REGLA: Menciona las 3 características más relevantes con sus valores de contribución. "
        "Entre 3 y 5 oraciones."
    ),
    "domain-expert": (
        "Eres un asistente de IA hablando con un experto del dominio (ej. un médico o funcionario).\n"
        "Traduce los conceptos de Machine Learning a terminología del dominio. "
        "Evita jerga técnica pura de ML pero mantén profundidad profesional.\n\n"
        "Contexto de la base de conocimientos:\n{context}\n\n"
        "Datos de la explicación:\n{explanation}\n\n"
        "Pregunta: {question}\n\n"
        "Proporciona una explicación orientada al dominio en ESPAÑOL. "
        "REGLA: Menciona entre 2 y 4 factores relevantes que influyeron en la decisión, "
        "no solo el más importante. Entre 3 y 5 oraciones."
    ),
    "non-expert": (
        "Eres un asistente de IA amigable que explica resultados a alguien sin formación técnica.\n"
        "Usa lenguaje cotidiano, evita tecnicismos y sé tranquilizador.\n\n"
        "Contexto de la base de conocimientos:\n{context}\n\n"
        "Datos de la explicación:\n{explanation}\n\n"
        "Pregunta: {question}\n\n"
        "Explica el resultado en ESPAÑOL en términos simples. "
        "REGLA: Menciona 2 o 3 razones concretas que llevaron a esta decisión, no solo una. "
        "Entre 3 y 4 oraciones claras."
    ),
}

CHAT_PROFILE_PROMPTS = {
    "data-scientist": (
        "Eres el asistente de la Herramienta de Explicabilidad. Hablas con un Data Scientist o ingeniero de ML.\n"
        "Puedes citar valores SHAP, LIME, anchors y métricas directamente.\n\n"
        "REGLA DE PROFUNDIDAD — lee el historial antes de responder:\n"
        "- Si es la PRIMERA pregunta sobre un tema: responde en 2 oraciones precisas.\n"
        "- Si el usuario pide aclaración o más detalle: NO repitas lo ya dicho. "
        "Agrega el siguiente nivel técnico: distribución del feature, interacción entre variables, "
        "implicación en el pipeline, o limitación del método de explicación usado. Máximo 3 oraciones.\n"
        "- Nunca repitas literalmente algo que ya aparece en el historial.\n\n"
        "EXPLICACIÓN:\n{explanation_context}\n\n"
        "DOCUMENTOS:\n{context}\n\n"
        "HISTORIAL:\n{history}\n\n"
        "PREGUNTA: {question}\n\n"
        "RESPUESTA (sin introducción, sin saludos, directo al punto):"
    ),
    "domain-expert": (
        "Eres el asistente de la Herramienta de Explicabilidad. Hablas con un profesional experto en el dominio (médico, funcionario SUSESO).\n"
        "PROHIBIDO usar estas palabras: SHAP, LIME, anchor, feature, contribución, modelo, algoritmo, variable técnica, `cant_instrumentos`, backticks.\n"
        "En su lugar usa: 'la cantidad de licencias', 'el departamento asignado', 'el tipo de patología', 'influyó fuertemente', 'pesó en contra'.\n\n"
        "REGLA DE PROFUNDIDAD — lee el historial antes de responder:\n"
        "- Si es la PRIMERA pregunta sobre un tema: responde en 2 oraciones en lenguaje del dominio.\n"
        "- Si el usuario pide aclaración o más detalle: NO repitas lo ya dicho. "
        "Agrega un ángulo nuevo: implicación clínica o administrativa del factor, patrón histórico observado, "
        "o qué significaría ese factor en la práctica para el caso. Máximo 3 oraciones.\n"
        "- Nunca repitas literalmente algo que ya aparece en el historial.\n\n"
        "EXPLICACIÓN:\n{explanation_context}\n\n"
        "DOCUMENTOS:\n{context}\n\n"
        "HISTORIAL:\n{history}\n\n"
        "PREGUNTA: {question}\n\n"
        "RESPUESTA (sin introducción, sin saludos, directo al punto):"
    ),
    "non-expert": (
        "Eres el asistente de la Herramienta de Explicabilidad. Hablas con una persona sin conocimientos técnicos ni médicos.\n"
        "PROHIBIDO usar estas palabras: SHAP, LIME, anchor, feature, modelo, algoritmo, contribución, variable, `cant_instrumentos`, backticks, porcentaje técnico.\n"
        "Usa solo lenguaje cotidiano y sencillo.\n\n"
        "REGLA DE PROFUNDIDAD — lee el historial antes de responder:\n"
        "- Si es la PRIMERA pregunta sobre un tema: responde en 1-2 oraciones simples.\n"
        "- Si el usuario pide aclaración ('explícame mejor', 'no entendí', 'por qué', 'puedes detallar'): "
        "NO repitas lo que ya dijiste. Agrega un ángulo nuevo: una consecuencia práctica, una analogía del "
        "mundo real, o un ejemplo concreto de lo que significa ese factor en la vida del trabajador. "
        "Máximo 3 oraciones.\n"
        "- Nunca repitas literalmente algo que ya aparece en el historial.\n\n"
        "EXPLICACIÓN:\n{explanation_context}\n\n"
        "DOCUMENTOS:\n{context}\n\n"
        "HISTORIAL:\n{history}\n\n"
        "PREGUNTA: {question}\n\n"
        "RESPUESTA (sin introducción, sin saludos, directo al punto):"
    ),
}


def _build_label_instruction(label_map: Optional[Dict[str, str]]) -> str:
    """Returns a strict instruction block so Gemini uses exact label names."""
    if not label_map:
        return ""
    pairs = ", ".join(f'"{k}" → "{v}"' for k, v in label_map.items())
    return (
        f"\n[ETIQUETAS DE CLASE — usa SOLO estos nombres, nunca traduzcas ni interpretes IDs numéricos: {pairs}]\n"
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

        # ── LangChain LLM (Gemini vía Vertex AI) ─────────────────────────
        # NOTA: Cambiamos ChatGoogleGenerativeAI por ChatVertexAI
        self.llm = ChatVertexAI(
            model_name=settings.GEMINI_MODEL, # Asegúrate que en settings sea "gemini-1.5-flash"
            project=settings.GCP_PROJECT_ID,
            location=settings.GCP_LOCATION,
            temperature=0.3,
        )

        # ── Estado ───────────────────────────────────────────────────────
        self.corpus = None
        self.corpus_name: Optional[str] = None


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
            description="herramienta-explicabilidad knowledge base corpus",
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
            retrieval_config = rag.RagRetrievalConfig(
                top_k=k,
                )
            #vector_distance_threshold=settings.RAG_VECTOR_DISTANCE_THRESHOLD

            response = rag.retrieval_query(
                rag_resources=[
                    rag.RagResource(rag_corpus=self.corpus_name)
                ],
                text=query,
                rag_retrieval_config=retrieval_config,
            )

            chunks = [ctx.text for ctx in response.contexts.contexts if ctx.text]
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
        label_map: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Genera una narrativa en lenguaje natural adaptada al perfil,
        usando contexto recuperado de Vertex AI RAG Engine
        y generación con LangChain + Gemini.
        """
        label_instruction = _build_label_instruction(label_map)
        explanation_str = label_instruction + json.dumps(explanation_data, indent=2, ensure_ascii=False)

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
        explanation_context: str = "",
        label_map: Optional[Dict[str, str]] = None,
    ) -> dict:
        """
        Chat interactivo: recupera contexto de Vertex AI RAG Engine,
        gestiona la memoria con LangChain y genera respuesta con Gemini.
        """
        # Recuperar contexto relevante del corpus
        context_chunks = self.retrieve(message)
        context = "\n---\n".join(context_chunks) if context_chunks else "No additional context."

        # Prepend label instruction so Gemini uses exact class names
        label_instruction = _build_label_instruction(label_map)
        if label_instruction:
            explanation_context = label_instruction + explanation_context

        history_text = ""
        for msg in history[-6:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_text += f"{role}: {msg.get('content', '')}\n"

        template = CHAT_PROFILE_PROMPTS.get(profile, CHAT_PROFILE_PROMPTS["non-expert"])
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "history", "explanation_context", "question"],
        )

        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "history": history_text,
            "explanation_context": explanation_context,
            "question": message,
        })

        return {"response": response.content, "sources": context_chunks[:3]}
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
