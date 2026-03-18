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
import gc 
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
        "last_instance": None, 
        "last_result": None,  
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
        background_data=df[features].sample(n=70, random_state=42) if len(df) > 100 else df[features],
    ) #Seguridad para prueba rapida, solo 70 instancias

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
                    explanation_data=result, profile=profile,
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
        raise HTTPException(status_code=500, detail=str(e))

def _build_narrative(result: dict, profile: str) -> str:
    pred = result.get("prediction", "?")
    label = result.get("label", pred)
    conf = result.get("confidence", 0)
    method = result.get("method_used", "unknown")
    exps = result.get("explanations", {})

    # 1. Extraemos el top_feat (Para SHAP y LIME)
    top_feat = "desconocida"
    if method in ("shap", "lime") and method in exps:
        feats = exps[method].get("features", [])
        if feats:
            if method == "shap":
                sorted_f = sorted(feats, key=lambda f: abs(f.get("shap_value", 0)), reverse=True)
            else:
                sorted_f = sorted(feats, key=lambda f: abs(f.get("lime_weight", 0)), reverse=True)
            top_feat = sorted_f[0].get("name", "desconocida")

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
                f"Se encontró que las condiciones: [{condiciones}] son suficientes para anclar la predicción. "
                f"Revise la cobertura y precisión de esta regla en la pestaña de métricas."
            )
        else:
            return (
                f"El modelo clasificó la instancia como '{label}' (clase {pred}) "
                f"con una confianza del {conf:.2f}% (Método: {method.upper()}).\n\n"
                f"El factor determinante fue '{top_feat}', con la mayor contribución marginal o peso local. "
                f"Se recomienda revisar las otras pestañas para confirmar la estabilidad de la explicación."
            )

    elif profile == "domain-expert":
        if method == "anchor":
            return (
                f"El análisis indica que este caso corresponde a '{label}'.\n\n"
                f"El sistema encontró una regla estricta: si se cumple que [{condiciones}], "
                f"el resultado siempre será este. Valide si esta combinación de reglas coincide con los protocolos de su área."
            )
        else:
            return (
                f"El análisis indica que este caso corresponde a '{label}' "
                f"(confianza: {conf:.2f}%).\n\n"
                f"El factor más relevante en esta decisión fue '{top_feat}'. Desde la perspectiva del "
                f"dominio, evalúe si el peso de esta variable tiene sentido lógico o profesional."
            )

    else: # Usuario sin contexto
        if method == "anchor":
            return (
                f"¡Hola! El sistema analizó los datos y concluye que este caso "
                f"corresponde a '{label}'.\n\n"
                f"El sistema tomó esta decisión basándose en una regla clara. Como se cumplió que: {condiciones}, "
                f"el modelo estuvo completamente seguro de su respuesta. Es como seguir una receta paso a paso."
            )
        else:
            return (
                f"¡Hola! El sistema analizó los datos y concluye que este caso "
                f"corresponde a '{label}'.\n\n"
                f"La característica que más influyó en este resultado fue '{top_feat}'. "
                f"En términos sencillos: si el valor de '{top_feat}' fuera diferente, "
                f"es muy probable que la decisión de la Inteligencia Artificial hubiera cambiado."
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
