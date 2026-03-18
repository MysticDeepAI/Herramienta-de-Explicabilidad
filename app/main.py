"""
Profile XAI API — Backend v2 (intermedio).

Motor XAI completo: SHAP + LIME + Anchor + métricas + selección automática.
RAG y Chat desactivados hasta configurar Vertex AI.
Narrativas generadas con texto simple por ahora.
"""

import json
import os
import uuid
import logging
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.utils import infer_column_type, save_upload
from app.modeling import ModelExplainer
from app.explanation import ExplanationEngine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Detectar si RAG está disponible ──────────────────────────────────────
HAS_RAG = False
try:
    from app.rag import RAGEngine
    HAS_RAG = True
    logger.info("RAG Engine (Vertex AI + LangChain) disponible")
except ImportError as e:
    logger.info("RAG no disponible (%s) — narrativas sin LLM", e)


# ─── App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ProfileXAI API",
    version="2.0.0",
    description="API de explicabilidad adaptativa — SHAP, LIME, Anchor + RAG + Chat",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOB_STORE: Dict[str, Dict[str, Any]] = {}


def _get_job(job_id: str) -> Dict[str, Any]:
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail="Job ID no encontrado.")
    return JOB_STORE[job_id]


# ─── Health check ───────────────────────────────────────────────────────

@app.api_route("/", methods=["GET", "HEAD"])
async def health():
    return {
        "status": "ok",
        "service": "profile-xai-api",
        "version": "2.0.0",
        "rag_available": HAS_RAG,
    }


# ─── Procesamiento ──────────────────────────────────────────────────────

@app.post("/api/processing/start")
async def start_processing():
    jid = f"job_{uuid.uuid4().hex[:8]}"
    JOB_STORE[jid] = {
        "dataset_csv": None,
        "model_path": None,
        "kb_files": [],
        "rag_engine": None,
        "explanation_engine": None,
    }
    return {"jobId": jid, "status": "ready"}


# ─── Uploads ────────────────────────────────────────────────────────────

@app.post("/api/upload/{upload_type}")
async def upload_files(
    upload_type: str, jobId: str, files: List[UploadFile] = File(...),
):
    job = _get_job(jobId)
    folder = os.path.join(settings.UPLOAD_DIR, jobId, upload_type)

    saved: list[dict] = []
    for f in files:
        path = await save_upload(f, folder)
        saved.append({"name": f.filename, "path": path})

        if upload_type == "dataset" and f.filename.endswith(".csv"):
            job["dataset_csv"] = path
        elif upload_type == "model" and f.filename.endswith((".pkl", ".h5", ".joblib")):
            job["model_path"] = path
        elif upload_type == "knowledge-base":
            job["kb_files"].append(path)

    # Inicializar RAG si está disponible
    if upload_type == "knowledge-base" and job["kb_files"] and HAS_RAG:
        try:
            if job["rag_engine"] is None:
                job["rag_engine"] = RAGEngine()
            n = job["rag_engine"].ingest(job["kb_files"])
            logger.info("RAG: %d archivos indexados para job %s", n, jobId)
        except Exception as e:
            logger.warning("RAG no pudo inicializarse: %s", e)

    return {"jobId": jobId, "uploaded": saved}


# ─── Schema del dataset ─────────────────────────────────────────────────

@app.get("/api/dataset/schema")
async def dataset_schema(jobId: str):
    job = _get_job(jobId)
    if not job.get("dataset_csv"):
        return {"columns": []}

    df = pd.read_csv(job["dataset_csv"])
    columns: list[dict] = []
    for col in df.columns:
        ctype = infer_column_type(df[col])
        item: dict = {"name": str(col), "type": ctype}
        if ctype == "categorical":
            opciones = sorted(df[col].dropna().unique().tolist())
            item["options"] = [str(o) for o in opciones]
        columns.append(item)

    return {"columns": columns}


# ─── Instancia aleatoria ────────────────────────────────────────────────

@app.get("/api/dataset/random-instance")
async def random_instance(jobId: str):
    job = _get_job(jobId)
    if not job.get("dataset_csv"):
        raise HTTPException(status_code=400, detail="Dataset no subido.")

    df = pd.read_csv(job["dataset_csv"], nrows=500)
    row = df.sample(1).iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


# ─── Explicación principal ──────────────────────────────────────────────

def _get_or_create_engine(job: dict, features: List[str]) -> ExplanationEngine:
    """Inicializa el ExplanationEngine si no existe."""
    if job["explanation_engine"] is not None:
        return job["explanation_engine"]

    df = pd.read_csv(job["dataset_csv"])

    me = ModelExplainer(
        pipeline_path=job["model_path"],
        background_data=df[features],
    )

    # Detectar clases y crear label_map
    if hasattr(me.model, "classes_"):
        # Quitamos el int(c) para que soporte tanto números como textos
        label_map = {c: str(c) for c in me.model.classes_}
    else:
        label_map = {0: "Class 0", 1: "Class 1"}

    engine = ExplanationEngine(
        model_explainer=me,
        target_name="target",
        label_map=label_map,
    )
    job["explanation_engine"] = engine
    return engine


@app.post("/api/explain")
async def explain(payload: dict):
    """
    Genera predicción + explicación con ExplanationEngine completo.
    Soporta selección automática del mejor método o método forzado.
    """
    job_id = payload.get("jobId")
    instance_dict = payload.get("instance", {})
    profile = payload.get("profile", "non-expert")
    method = payload.get("method")  # "shap", "lime", "anchor", o None=auto

    if not instance_dict:
        raise HTTPException(status_code=400, detail="El formulario llegó vacío.")

    job = _get_job(job_id)
    if not job.get("dataset_csv") or not job.get("model_path"):
        raise HTTPException(status_code=400, detail="Faltan dataset o modelo.")

    features = list(instance_dict.keys())

    try:
        engine = _get_or_create_engine(job, features)
        result = engine.explain_instance(instance_dict, method=method)

        # ── Generar narrativa ────────────────────────────────────────────
        natural_text = ""

        # Intentar con RAG si está disponible
        rag_engine = job.get("rag_engine")
        if rag_engine:
            try:
                natural_text = rag_engine.generate_narrative(
                    explanation_data=result, profile=profile,
                )
            except Exception as e:
                logger.warning("RAG narrative falló: %s", e)

        # Fallback: narrativa simple sin LLM
        if not natural_text:
            natural_text = _build_narrative(result, profile)

        # ── Formatear respuesta para el frontend ─────────────────────────
        technical = _format_technical_for_frontend(result)

        return {
            "prediction": result["prediction"],
            "label": result.get("label", result["prediction"]),
            "profile": profile,
            "method_used": result["method_used"],
            "natural": natural_text,
            "technical": technical,
        }

    except Exception as e:
        logger.error("Error en explain: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _build_narrative(result: dict, profile: str) -> str:
    """Genera narrativa adaptada al perfil sin usar LLM."""
    pred = result.get("prediction", "?")
    label = result.get("label", pred)
    conf = result.get("confidence", 0)
    method = result.get("method_used", "unknown")

    # Extraer el feature más importante
    exp = result.get("explanation", {})
    top_feat = "desconocida"

    if method in ("shap", "lime"):
        feats = exp.get("features", [])
        if feats:
            if method == "shap":
                sorted_f = sorted(feats, key=lambda f: abs(f.get("shap_value", 0)), reverse=True)
            else:
                sorted_f = sorted(feats, key=lambda f: abs(f.get("lime_weight", 0)), reverse=True)
            top_feat = sorted_f[0].get("name", "desconocida")
    elif method == "anchor":
        anchor = exp.get("anchor", {})
        conditions = anchor.get("conditions", [])
        if conditions:
            top_feat = conditions[0]

    if profile == "data-scientist":
        return (
            f"El modelo clasificó la instancia como '{label}' (clase {pred}) "
            f"con una confianza del {conf:.2f}% usando el método {method.upper()}.\n\n"
            f"El factor determinante fue '{top_feat}', con el mayor peso en la "
            f"decisión local. Se recomienda evaluar la estabilidad de esta variable "
            f"y comparar con los resultados de los otros explicadores disponibles."
        )
    elif profile == "domain-expert":
        return (
            f"El análisis indica que este caso corresponde a '{label}' "
            f"(confianza: {conf:.2f}%). El método utilizado fue {method.upper()}.\n\n"
            f"El factor más relevante fue '{top_feat}'. Desde la perspectiva del "
            f"dominio, este factor debería considerarse prioritariamente en la "
            f"evaluación profesional."
        )
    else:
        return (
            f"¡Hola! El sistema analizó los datos y concluye que este caso "
            f"corresponde a '{label}'.\n\n"
            f"La característica que más influyó fue '{top_feat}'. "
            f"En términos sencillos: si el valor de '{top_feat}' fuera diferente, "
            f"es muy probable que el resultado también hubiera cambiado."
        )


def _format_technical_for_frontend(result: dict) -> dict:
    """
    Transforma la salida del ExplanationEngine al formato que espera el frontend:
      - technical.lime: [{feature, weight}, ...]
      - technical.shap: [{feature, contribution}, ...]
      - technical.anchors: ["condition1", "condition2", ...]
      - technical.metrics: [{name, value}, ...]
    """
    exp = result.get("explanation", {})
    method = result.get("method_used", "")
    confidence = result.get("confidence", 0)
    metrics_data = result.get("metrics")

    technical: dict = {
        "lime": [],
        "shap": [],
        "anchors": [],
        "metrics": [{"name": "Confianza", "value": f"{confidence:.2f}%"}],
    }

    # ── Llenar según el método usado ─────────────────────────────────
    features = exp.get("features", [])
    anchor_data = exp.get("anchor", {})

    if method == "lime" and features:
        technical["lime"] = [
            {"feature": f.get("name", ""), "weight": f.get("lime_weight", 0)}
            for f in features
        ]
    elif method == "shap" and features:
        technical["shap"] = [
            {"feature": f.get("name", ""), "contribution": f.get("shap_value", 0)}
            for f in features
        ]
    elif method == "anchor" and anchor_data:
        technical["anchors"] = anchor_data.get("conditions", [])

    # ── Métricas del explanation engine ───────────────────────────────
    if metrics_data and isinstance(metrics_data, dict):
        # Agregar métricas del método seleccionado
        method_metrics = metrics_data.get(method, {})
        if method_metrics:
            technical["metrics"].extend([
                {"name": "Infidelity", "value": f"{method_metrics.get('infidelity', 0):.4f}"},
                {"name": "Lipschitz", "value": f"{method_metrics.get('lipschitz', 0):.4f}"},
                {"name": "Eff. Complexity", "value": f"{method_metrics.get('effective_complexity', 0):.1f}"},
            ])

        # Agregar precisión y cobertura si es anchor
        if method == "anchor" and anchor_data:
            precision = anchor_data.get("precision")
            coverage = anchor_data.get("coverage")
            if precision is not None:
                technical["metrics"].append(
                    {"name": "Precisión Anchor", "value": f"{precision:.2%}"}
                )
            if coverage is not None:
                technical["metrics"].append(
                    {"name": "Cobertura Anchor", "value": f"{coverage:.2%}"}
                )

    return technical


# ─── Chat ───────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(payload: dict):
    """
    Chat conversacional. Usa Vertex AI RAG + LangChain si está disponible,
    sino responde que el módulo requiere configuración.
    """
    job_id = payload.get("jobId")
    message = payload.get("message", "")
    profile = payload.get("profile", "non-expert")
    history = payload.get("history", [])

    if not message:
        raise HTTPException(status_code=400, detail="Mensaje vacío.")

    job = _get_job(job_id)
    rag_engine = job.get("rag_engine")

    if rag_engine:
        try:
            result = rag_engine.chat(
                message=message, profile=profile, history=history,
            )
            return result
        except Exception as e:
            logger.warning("Chat RAG falló: %s", e)

    return {
        "response": (
            "El módulo de chat estará disponible próximamente. "
            "Requiere configurar Vertex AI RAG Engine y Gemini. "
            "Por ahora, puedes ver la explicación generada en la sección principal."
        ),
        "sources": [],
    }
