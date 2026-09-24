"""
Herramienta de Explicabilidad API — Backend v2 (intermedio).

Motor XAI completo: SHAP + LIME + Anchor + métricas + selección automática.
RAG y Chat desactivados hasta configurar Vertex AI.
Narrativas generadas con texto simple por ahora.
"""

import json
import os
import uuid
import logging
import gc
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import asyncio

from app.config import settings
from app.utils import infer_column_type, save_upload
from app.explanation import ExplanationEngine
from app.converter_client import get_converter
from app.modeling import ModelExplainer, AgnosticModelExplainer
from app.security import scan_pickle, patch_safe_loader


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Capa 2 activa desde el primer import: joblib.load() queda restringido
patch_safe_loader()

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
    title="Herramienta de Explicabilidad API",
    version="2.0.0",
    description="API de explicabilidad adaptativa — SHAP, LIME, Anchor + RAG + Chat",
)

# ── Middleware de API Key ────────────────────────────────────────────────
# Rutas públicas que NO requieren autenticación (health check y OPTIONS)
_PUBLIC_PATHS = {"/", "/docs", "/openapi.json", "/redoc"}

class _APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not settings.API_KEY:
            # Sin API_KEY configurada → modo desarrollo, sin restricción
            return await call_next(request)
        if request.method == "OPTIONS" or request.url.path in _PUBLIC_PATHS:
            return await call_next(request)
        key = request.headers.get("X-API-Key", "")
        if key != settings.API_KEY:
            return JSONResponse(status_code=401, content={"detail": "API Key inválida o ausente."})
        return await call_next(request)

app.add_middleware(_APIKeyMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key"],
)

JOB_STORE: Dict[str, Dict[str, Any]] = {}

# Límite máximo de jobs simultáneos en memoria y TTL en segundos (2 horas)
_JOB_MAX = int(os.getenv("JOB_MAX", "50"))
_JOB_TTL = int(os.getenv("JOB_TTL_SECONDS", "7200"))


def _purge_expired_jobs() -> None:
    """Elimina jobs expirados y recorta si se supera el límite máximo."""
    now = datetime.utcnow().timestamp()
    expired = [jid for jid, j in JOB_STORE.items() if now - j.get("_created_at", now) > _JOB_TTL]
    for jid in expired:
        JOB_STORE.pop(jid, None)
        logger.info("Job expirado eliminado: %s", jid)

    # Si aún supera el límite, eliminar los más antiguos
    if len(JOB_STORE) >= _JOB_MAX:
        sorted_jobs = sorted(JOB_STORE.items(), key=lambda x: x[1].get("_created_at", 0))
        for jid, _ in sorted_jobs[:len(JOB_STORE) - _JOB_MAX + 1]:
            JOB_STORE.pop(jid, None)
            logger.info("Job eliminado por límite de capacidad: %s", jid)


# ─── Manejador global de errores 500 ────────────────────────────────────
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error [%s %s]: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor. Por favor intente más tarde."},
    )


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
    _purge_expired_jobs()
    if len(JOB_STORE) >= _JOB_MAX:
        raise HTTPException(status_code=503, detail="Capacidad máxima de sesiones alcanzada. Intente más tarde.")
    jid = f"job_{uuid.uuid4().hex[:8]}"
    JOB_STORE[jid] = {
        "_created_at": datetime.utcnow().timestamp(),
        "dataset_csv": None,
        "model_path": None,
        "kb_files": [],
        "rag_engine": None,
        "explanation_engine": None,
        "last_instance": None,
        "last_result": None,
        "label_map": None,
        "detected_classes": None,
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
        # 1. Log para ver QUÉ archivo está entrando exactamente
        logger.info(f"Recibiendo archivo: {f.filename} para el endpoint: {upload_type}")
        
        path = await save_upload(f, folder)
        saved.append({"name": f.filename, "path": path})

        # 2. Hacemos el check en minúsculas para evitar errores
        nombre_archivo = f.filename.lower()

        if upload_type == "dataset" and nombre_archivo.endswith(".csv"):
            job["dataset_csv"] = path
            # 3. Log para confirmar que la memoria se actualizó
            logger.info(f"¡ÉXITO! Dataset detectado y guardado en memoria: {path}")

        elif upload_type == "model" and nombre_archivo.endswith((".pkl", ".joblib")):
            # ── CAPA 1: Scan estático del pickle antes de registrar el path ──
            is_safe, reason = scan_pickle(path)
            if not is_safe:
                import os as _os
                _os.remove(path)  # borrar el archivo sospechoso del disco
                raise HTTPException(
                    status_code=400,
                    detail=f"El archivo de modelo fue rechazado por el análisis de seguridad: {reason}",
                )
            job["model_path"] = path

        elif upload_type == "model" and nombre_archivo.endswith(".onnx"):
            job["model_path"] = path  # ONNX es seguro por diseño (no es pickle)

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

    df = pd.read_csv(job["dataset_csv"], sep=None, engine="python")
    columns: list[dict] = []
    for col in df.columns:
        ctype = infer_column_type(df[col])
        item: dict = {"name": str(col), "type": ctype}
        if ctype == "categorical":
            opciones = sorted(df[col].dropna().unique().tolist())
            item["options"] = [str(o) for o in opciones]
        columns.append(item)


    return {"columns": columns}


# ─── Conversión del modelo y etiquetado ─────────────────────────────────

@app.post("/api/job/{job_id}/convert")
async def convert_model(job_id: str):
    """
    Ejecuta el pipeline docker-in-docker: .pkl → .onnx + metadata.json
    Devuelve las clases detectadas para que el frontend muestre el formulario
    de etiquetado.
    """
    job = _get_job(job_id)

    if not job.get("model_path") or not job.get("dataset_csv"):
        raise HTTPException(status_code=400, detail="Se requieren modelo y dataset antes de convertir.")

    model_path: str = job["model_path"]
    dataset_path: str = job["dataset_csv"]

    # Si ya es .onnx, solo leer metadata si existe
    if model_path.endswith(".onnx"):
        meta_path = os.path.splitext(model_path)[0] + ".metadata.json"
        classes_detected: List[str] = []
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            classes_detected = [str(c) for c in meta.get("classes", [])]
        job["detected_classes"] = classes_detected
        return {"status": "ready", "classes_detected": classes_detected, "n_classes": len(classes_detected)}

    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    onnx_path = os.path.join(os.path.dirname(model_path), f"{model_stem}_agnostico.onnx")

    try:
        await get_converter().convert(model_path, onnx_path, dataset_path)
    except Exception as e:
        logger.error("Error en conversión: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Conversión fallida: {e}")

    meta_path = os.path.splitext(onnx_path)[0] + ".metadata.json"
    classes_detected = []
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        classes_detected = [str(c) for c in meta.get("classes", [])]

    job["model_path"] = onnx_path
    job["detected_classes"] = classes_detected
    logger.info("Conversión exitosa. Listo para explicar.")

    return {"status": "ready", "classes_detected": classes_detected, "n_classes": len(classes_detected)}


@app.post("/api/job/{job_id}/labels")
async def set_labels(job_id: str, payload: dict):
    """Guarda el mapeo clase → nombre legible para el job."""
    job = _get_job(job_id)
    label_map = payload.get("labelMap", {})
    if not label_map:
        raise HTTPException(status_code=400, detail="labelMap no puede estar vacío.")
    job["label_map"] = label_map
    # Invalidar engine para que se recree con el nuevo label_map
    job["explanation_engine"] = None
    return {"jobId": job_id, "labelMap": label_map, "status": "saved"}


# ─── Instancia aleatoria ────────────────────────────────────────────────

@app.get("/api/dataset/random-instance")
async def random_instance(jobId: str):
    job = _get_job(jobId)
    if not job.get("dataset_csv"):
        raise HTTPException(status_code=400, detail="Dataset no subido.")

    df = pd.read_csv(job["dataset_csv"], sep=None, engine="python", nrows=500)
    row = df.sample(1).iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


# ─── Explicación principal ──────────────────────────────────────────────

def _get_or_create_engine(job: dict, features: List[str]) -> ExplanationEngine:
    """Inicializa el ExplanationEngine si no existe."""
    if job["explanation_engine"] is not None:
        return job["explanation_engine"]
    
    df = pd.read_csv(job["dataset_csv"], sep=None, engine="python")
    bg_data = df[features].sample(n=70, random_state=42) if len(df) > 100 else df[features]

    # --- RUTEO INTELIGENTE ---
    model_path = job["model_path"]
    
    if model_path.endswith('.onnx'):
        logger.info("Usando motor agnóstico (Fase 2)")
        me = AgnosticModelExplainer(model_path=model_path, background_data=bg_data)
    else:
        logger.info("Usando motor nativo")
        me = ModelExplainer(pipeline_path=model_path, background_data=bg_data)

    # Prioridad: label_map del usuario → classes del modelo → fallback genérico
    stored_label_map = job.get("label_map")
    if stored_label_map:
        label_map = stored_label_map
    elif hasattr(me.model, "classes_") and me.model.classes_ is not None:
        label_map = {str(c): str(c) for c in me.model.classes_}
    else:
        logger.warning("El modelo no expone clases explícitas. Usando mapeo genérico.")
        label_map = {"0": "Clase 0", "1": "Clase 1"}

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
        # Verificamos si los datos del paciente/instancia son idénticos a los de la última petición
        if job.get("last_instance") == instance_dict and job.get("last_result") is not None:
            logger.info("Instancia repetida detectada. Reciclando cálculos XAI en caché...")
            result = job["last_result"]
            
            # Si el usuario forzó un método distinto, actualizamos la etiqueta
            if method is not None:
                result["method_used"] = method
        else:
            # Si es una instancia nueva, o es la primera vez, calculamos desde cero
            logger.info("Instancia nueva. Calculando explicaciones XAI desde cero...")
            engine = _get_or_create_engine(job, features)
            result = engine.explain_instance(instance_dict, method=method)
            
            # Guardamos los resultados en el caché del Job para la próxima vez
            job["last_instance"] = instance_dict
            job["last_result"] = result

        # ── Generar narrativa ────────────────────────────────────────────
        natural_text = ""

        # Intentar con RAG si está disponible
        rag_engine = job.get("rag_engine")
        if rag_engine:
            try:
                natural_text = rag_engine.generate_narrative(
                    explanation_data=result,
                    profile=profile,
                    label_map=job.get("label_map"),
                )
            except Exception as e:
                logger.warning("RAG narrative falló: %s", e)

        # Fallback: narrativa simple sin LLM
        if not natural_text:
            natural_text = _build_narrative(result, profile)

        # ── Formatear respuesta para el frontend ─────────────────────────
        technical = _format_technical_for_frontend(result)

        # Preparamos la respuesta final
        response_data = {
            "prediction": result["prediction"],
            "label": result.get("label", result["prediction"]),
            "profile": profile,
            "method_used": result["method_used"],
            "natural": natural_text,
            "technical": technical,
        }

        # ─── LIMPIEZA DE MEMORIA FORZADA PARA RENDER ───
        import gc
        gc.collect()

        return response_data

    except Exception as e:
        logger.error("Error en explain: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Error procesando la explicación. Por favor intente más tarde.")

def _build_narrative(result: dict, profile: str) -> str:
    pred = result.get("prediction", "?")
    label = result.get("label", pred)
    conf = result.get("confidence", 0)
    method = result.get("method_used", "unknown")
    exps = result.get("explanations", {})

    # 1. Extraemos los top features (Para SHAP y LIME)
    top_feats: list[str] = []
    if method in ("shap", "lime") and method in exps:
        feats = exps[method].get("features", [])
        if feats:
            if method == "shap":
                sorted_f = sorted(feats, key=lambda f: abs(f.get("shap_value", 0)), reverse=True)
            else:
                sorted_f = sorted(feats, key=lambda f: abs(f.get("lime_weight", 0)), reverse=True)
            top_feats = [f.get("name", "desconocida") for f in sorted_f[:3] if f.get("name")]
    top_feat = top_feats[0] if top_feats else "desconocida"

    def _feats_text(feats: list[str]) -> str:
        if not feats:
            return "los factores analizados"
        if len(feats) == 1:
            return f"'{feats[0]}'"
        return ", ".join(f"'{f}'" for f in feats[:-1]) + f" y '{feats[-1]}'"

    # 2. Extraemos las condiciones (Para ANCHOR)
    condiciones = "condiciones desconocidas"
    if method == "anchor" and "anchor" in exps:
        anchor_data = exps["anchor"].get("anchor", {})
        conds = anchor_data.get("conditions", [])
        if conds:
            condiciones = " Y ".join(conds)

    if profile == "data-scientist":
        if method == "anchor":
            return (
                f"El modelo clasificó la instancia como '{label}' (clase {pred}) "
                f"usando explicaciones basadas en reglas (Método: ANCHOR).\n\n"
                f"Se encontró que las condiciones [{condiciones}] son suficientes para anclar la predicción. "
                f"Revise la cobertura y precisión de esta regla en la pestaña de métricas."
            )
        else:
            return (
                f"El modelo clasificó la instancia como '{label}' (clase {pred}) "
                f"con una confianza del {conf:.2f}% (Método: {method.upper()}).\n\n"
                f"Los factores con mayor contribución marginal fueron {_feats_text(top_feats)}. "
                f"El primero presentó el peso local más alto; los siguientes refuerzan o contrarrestan la predicción. "
                f"Se recomienda revisar las otras pestañas para confirmar la estabilidad de la explicación."
            )

    elif profile == "domain-expert":
        if method == "anchor":
            return (
                f"El análisis indica que este caso corresponde a '{label}'.\n\n"
                f"El sistema encontró una regla estricta: si se cumple que [{condiciones}], "
                f"el resultado siempre será este. Valide si esta combinación de condiciones coincide con los protocolos de su área."
            )
        else:
            return (
                f"El análisis indica que este caso corresponde a '{label}' "
                f"(confianza: {conf:.2f}%).\n\n"
                f"Los factores que más influyeron en esta decisión fueron {_feats_text(top_feats)}. "
                f"Desde la perspectiva del dominio, evalúe si el peso de estas variables tiene sentido lógico o profesional."
            )

    else:  # Usuario sin contexto
        if method == "anchor":
            return (
                f"El sistema analizó los datos y concluye que este caso corresponde a '{label}'.\n\n"
                f"La decisión se basó en una regla clara: {condiciones}. "
                f"Cuando se cumplen estas condiciones juntas, el sistema llega siempre al mismo resultado."
            )
        else:
            return (
                f"El sistema analizó los datos y concluye que este caso corresponde a '{label}'.\n\n"
                f"Las características que más influyeron en este resultado fueron {_feats_text(top_feats)}. "
                f"En términos sencillos: si cualquiera de estos valores fuera diferente, "
                f"es probable que la decisión del sistema hubiera cambiado."
            )

def _format_technical_for_frontend(result: dict) -> dict:
    """
    Transforma la salida del ExplanationEngine al formato que espera el frontend.
    Ahora lee múltiples explicaciones simultáneas de 'explanations'.
    """
    # Usamos plural 'explanations' porque ahora vienen todas juntas
    exps = result.get("explanations", {})
    method = result.get("method_used", "") # El método principal que sobrevivió
    confidence = result.get("confidence", 0)
    metrics_data = result.get("metrics")

    technical: dict = {
        "lime": [],
        "shap": [],
        "anchors": [],
        "metrics": [{"name": "Confianza", "value": f"{confidence:.2f}%"}],
    }

    # ── 1. Llenar TODOS los métodos disponibles al mismo tiempo ──
    if "lime" in exps:
        lime_features = exps["lime"].get("features", [])
        technical["lime"] = [
            {"feature": f.get("name", ""), "weight": f.get("lime_weight", 0)}
            for f in lime_features
        ]

    if "shap" in exps:
        shap_features = exps["shap"].get("features", [])
        technical["shap"] = [
            {"feature": f.get("name", ""), "contribution": f.get("shap_value", 0)}
            for f in shap_features
        ]

    if "anchor" in exps:
            anchor_data = exps["anchor"].get("anchor", {})
            conditions = anchor_data.get("conditions", [])
            
            if conditions:
                # Rescatamos el nombre o número de la clase que el modelo predijo
                clase_justificada = result.get("label", result.get("prediction", "?"))
                
                # Agregamos un elemento visual al final de la lista de reglas
                conditions.append(f"➔ ENTONCES LA CLASE ES: {clase_justificada}")
                
            technical["anchors"] = conditions

    # ── 2. Métricas del explanation engine (Tu código original restaurado) ──
    if metrics_data and isinstance(metrics_data, dict):
        # Agregar métricas del método seleccionado
        method_metrics = metrics_data.get(method, {})
        if method_metrics:
            technical["metrics"].extend([
                {"name": "Infidelity", "value": f"{method_metrics.get('infidelity', 0):.4f}"},
                {"name": "Lipschitz", "value": f"{method_metrics.get('lipschitz', 0):.4f}"},
                {"name": "Eff. Complexity", "value": f"{method_metrics.get('effective_complexity', 0):.1f}"},
            ])

        # Agregar precisión y cobertura si el método principal es anchor
        if method == "anchor" and "anchor" in exps:
            anchor_data = exps["anchor"].get("anchor", {})
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
    explanation_context = payload.get("explanation_context", "")

    if not message:
        raise HTTPException(status_code=400, detail="Mensaje vacío.")

    job = _get_job(job_id)
    rag_engine = job.get("rag_engine")

    if rag_engine:
        try:
            result = rag_engine.chat(
                message=message,
                profile=profile,
                history=history,
                explanation_context=explanation_context,
                label_map=job.get("label_map"),
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


# ─── Feedback ────────────────────────────────────────────────────────────────

FEEDBACK_DIR = Path(__file__).parent.parent / "feedback_reports"
FEEDBACK_DIR.mkdir(exist_ok=True)


class FeedbackRequest(BaseModel):
    # Metadatos
    timestamp: str
    usuario: str = "Anónimo"
    # Paso 1 — Perfil + Likert
    perfil: str
    q1_1: int; q1_2: int; q1_3: int   # Dimensión 1
    q2_1: int; q2_2: int; q2_3: int   # Dimensión 2
    q3_1: int; q3_2: int               # Dimensión 3
    # Paso 2 — Cualitativo
    q_confuso: str = ""
    q_info_adicional: str = ""
    q_mejoras: str = ""
    # Paso 3 — Comentarios generales
    organizacion: str = ""
    categoria: str = "Comentario general"
    comentarios: str = ""


def _sanitize(text: str) -> str:
    """Replace characters outside Latin-1 with safe ASCII equivalents."""
    replacements = {
        "–": "-", "—": "-",   # en/em dash
        "‘": "'", "’": "'",   # curly single quotes
        "“": '"', "”": '"',   # curly double quotes
        "…": "...",                # ellipsis
        "°": " grados",            # degree sign (already latin-1, just in case)
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Drop any remaining characters outside latin-1
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _build_feedback_pdf(data: FeedbackRequest, out_path: Path) -> None:
    """Genera el PDF de feedback y lo escribe en out_path."""
    from fpdf import FPDF

    LIKERT_LABELS = {1: "1 - Muy en desacuerdo", 2: "2", 3: "3 - Neutral", 4: "4", 5: "5 - Muy de acuerdo"}
    PERFIL_LABELS = {
        "data-scientist": "Especialista en IA / Data Scientist",
        "domain-expert": "Experto en el Dominio (médico / funcionario)",
        "non-expert": "Usuario General",
    }

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 13)
            self.cell(0, 8, "Herramienta de Explicabilidad - Evaluacion de Usuario", align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_font("Helvetica", "", 9)
            self.cell(0, 5, "Herramienta de Explicabilidad (Piloto SUSESO)", align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(3)
            self.set_draw_color(180, 180, 180)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)

        def footer(self):
            self.set_y(-14)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(130, 130, 130)
            self.cell(0, 8, f"Pagina {self.page_no()} | Generado por la Herramienta de Explicabilidad", align="C")

        def section_title(self, title: str):
            self.ln(3)
            self.set_font("Helvetica", "B", 11)
            self.set_fill_color(240, 245, 255)
            self.set_text_color(30, 60, 120)
            self.cell(0, 8, title, fill=True, new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0, 0, 0)
            self.ln(2)

        def qa_row(self, question: str, answer: str):
            self.set_font("Helvetica", "", 9)
            self.set_text_color(60, 60, 60)
            self.multi_cell(0, 5, question, new_x="LMARGIN", new_y="NEXT")
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(0, 0, 0)
            self.multi_cell(0, 5, answer or "(sin respuesta)", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(18, 18, 18)

    # Sanitize all user text fields
    s = _sanitize

    # ── Metadatos ──────────────────────────────────────────────────
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, s(f"Fecha y hora: {data.timestamp}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, s(f"Usuario: {data.usuario}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, s(f"Perfil: {PERFIL_LABELS.get(data.perfil, data.perfil)}"), new_x="LMARGIN", new_y="NEXT")
    if data.organizacion:
        pdf.cell(0, 5, s(f"Organizacion: {data.organizacion}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    # ── Dimensión 1 ────────────────────────────────────────────────
    pdf.section_title("Dimension 1 - Calidad de la Explicacion")
    pdf.qa_row("La explicacion me ayudo a entender como funciona el modelo de licencias medicas.",
               LIKERT_LABELS.get(data.q1_1, str(data.q1_1)))
    pdf.qa_row("La explicacion contiene suficiente detalle para realizar mi trabajo.",
               LIKERT_LABELS.get(data.q1_2, str(data.q1_2)))
    pdf.qa_row("La explicacion fue precisa y coherente con la normativa.",
               LIKERT_LABELS.get(data.q1_3, str(data.q1_3)))

    # ── Dimensión 2 ────────────────────────────────────────────────
    pdf.section_title("Dimension 2 - Satisfaccion y Confianza")
    pdf.qa_row("Me siento satisfecho con la Herramienta de Explicabilidad.",
               LIKERT_LABELS.get(data.q2_1, str(data.q2_1)))
    pdf.qa_row("Confio en la evaluacion que hizo el sistema sobre la licencia.",
               LIKERT_LABELS.get(data.q2_2, str(data.q2_2)))
    pdf.qa_row("Usaria esta herramienta para apoyar mis decisiones futuras.",
               LIKERT_LABELS.get(data.q2_3, str(data.q2_3)))

    # ── Dimensión 3 ────────────────────────────────────────────────
    pdf.section_title("Dimension 3 - Adaptabilidad")
    pdf.qa_row("El lenguaje utilizado fue apropiado para mi nivel de conocimiento tecnico.",
               LIKERT_LABELS.get(data.q3_1, str(data.q3_1)))
    pdf.qa_row("La cantidad de informacion mostrada fue adecuada.",
               LIKERT_LABELS.get(data.q3_2, str(data.q3_2)))

    # ── Evaluación cualitativa ────────────────────────────────────
    pdf.section_title("Evaluacion Cualitativa")
    pdf.qa_row("Terminos tecnicos o graficos que resultaron confusos:", s(data.q_confuso))
    pdf.qa_row("Informacion adicional que hubiera gustado ver:", s(data.q_info_adicional))
    pdf.qa_row("Partes no intuitivas y sugerencias de mejora:", s(data.q_mejoras))

    # ── Comentarios generales ─────────────────────────────────────
    pdf.section_title("Comentarios Generales")
    pdf.qa_row(s(f"Categoria: {data.categoria}"), "")
    pdf.qa_row("Comentarios:", s(data.comentarios))

    pdf.output(str(out_path))


@app.post("/api/feedback")
async def submit_feedback(payload: FeedbackRequest):
    """Genera el PDF de evaluación y lo guarda en GCS (producción) o en disco (local)."""
    filename = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.pdf"
    tmp_path = Path("/tmp") / filename
    try:
        _build_feedback_pdf(payload, tmp_path)

        if settings.GCS_BUCKET:
            from google.cloud import storage as gcs
            bucket = gcs.Client().bucket(settings.GCS_BUCKET)
            bucket.blob(f"feedback/{filename}").upload_from_filename(
                str(tmp_path), content_type="application/pdf"
            )
            logger.info("Feedback subido a GCS: gs://%s/feedback/%s", settings.GCS_BUCKET, filename)
        else:
            tmp_path.rename(FEEDBACK_DIR / filename)
            logger.info("Feedback guardado localmente: %s", filename)

        return {"success": True, "file": filename, "message": "Feedback registrado correctamente."}
    except Exception as e:
        logger.error("Error generando PDF de feedback: %s", e)
        raise HTTPException(status_code=500, detail=f"Error al guardar feedback: {e}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.get("/api/feedback")
async def list_feedbacks():
    """Lista los archivos de feedback (GCS en producción, disco en local)."""
    if settings.GCS_BUCKET:
        from google.cloud import storage as gcs
        blobs = gcs.Client().list_blobs(settings.GCS_BUCKET, prefix="feedback/")
        files = sorted(
            [b.name.split("/")[-1] for b in blobs if b.name.endswith(".pdf")],
            reverse=True,
        )
        return {"files": files, "count": len(files), "storage": "gcs"}

    files = sorted(FEEDBACK_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"files": [f.name for f in files], "count": len(files), "storage": "local"}


@app.get("/api/feedback/{filename}")
async def download_feedback(filename: str):
    """Descarga un PDF de feedback (GCS en producción, disco en local)."""
    # C3: rechazar separadores de ruta antes de construir cualquier path
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Nombre de archivo inválido.")
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")

    if settings.GCS_BUCKET:
        from google.cloud import storage as gcs
        blob = gcs.Client().bucket(settings.GCS_BUCKET).blob(f"feedback/{filename}")
        if not blob.exists():
            raise HTTPException(status_code=404, detail="Archivo no encontrado.")

        def _stream():
            with blob.open("rb") as f:
                while chunk := f.read(65_536):
                    yield chunk

        return StreamingResponse(
            _stream(),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # Local: C3 — verificar que el path resuelto siga dentro de FEEDBACK_DIR
    path = (FEEDBACK_DIR / filename).resolve()
    if not path.is_relative_to(FEEDBACK_DIR.resolve()):
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")
    return FileResponse(path=str(path), media_type="application/pdf", filename=filename)


# ─── Reporte de explicación ───────────────────────────────────────────────────

PERFIL_LABELS = {
    "data-scientist": "Especialista en IA / Data Scientist",
    "domain-expert": "Experto en el Dominio",
    "non-expert": "Usuario General",
}


class ExplanationReportRequest(BaseModel):
    profile: str
    explanation: str
    chat_history: list[dict] = []
    timestamp: str = ""
    usuario: str = "Anónimo"


def _build_explanation_pdf(data: ExplanationReportRequest, out_path: Path) -> None:
    from fpdf import FPDF

    s = _sanitize

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 13)
            self.cell(0, 8, "Herramienta de Explicabilidad - Reporte", align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_font("Helvetica", "", 9)
            self.cell(0, 5, "Herramienta de Explicabilidad - Piloto SUSESO", align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(3)
            self.set_draw_color(180, 180, 180)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)

        def footer(self):
            self.set_y(-14)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(130, 130, 130)
            self.cell(0, 8, f"Pagina {self.page_no()} | Generado por la Herramienta de Explicabilidad", align="C")

        def section_title(self, title: str):
            self.ln(3)
            self.set_font("Helvetica", "B", 11)
            self.set_fill_color(240, 245, 255)
            self.set_text_color(30, 60, 120)
            self.cell(0, 8, title, fill=True, new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0, 0, 0)
            self.ln(2)

        def body_text(self, text: str):
            self.set_font("Helvetica", "", 10)
            self.multi_cell(0, 6, s(text))
            self.ln(2)

    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(18, 18, 18)

    # Metadatos
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    ts = data.timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 5, s(f"Fecha y hora: {ts}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, s(f"Usuario: {data.usuario}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, s(f"Perfil: {PERFIL_LABELS.get(data.profile, data.profile)}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # Explicación
    pdf.section_title("Explicacion en Lenguaje Natural")
    pdf.body_text(data.explanation)

    # Historial del chat
    if data.chat_history:
        pdf.section_title("Conversacion con el Asistente RAG")
        for msg in data.chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            label = "Usuario" if role == "user" else "Asistente"
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, s(f"{label}:"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, s(content))
            pdf.ln(3)

    pdf.output(str(out_path))


@app.post("/api/report/explanation")
async def generate_explanation_report(payload: ExplanationReportRequest):
    """Genera y descarga un PDF con la explicación y el historial de chat."""
    filename = f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.pdf"
    tmp_path = Path("/tmp") / filename
    try:
        _build_explanation_pdf(payload, tmp_path)
        content = tmp_path.read_bytes()
        return StreamingResponse(
            iter([content]),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        logger.error("Error generando reporte de explicacion: %s", e)
        raise HTTPException(status_code=500, detail="Error al generar el reporte.")
    finally:
        tmp_path.unlink(missing_ok=True)
