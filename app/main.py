"""
Profile XAI API — Backend para explicabilidad de modelos ML con LIME.
Diseñado para despliegue en Render (o cualquier PaaS con soporte Python).
"""

import os
import uuid
from typing import Any, Dict, List

import joblib
import lime.lime_tabular
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder

from app.config import settings
from app.utils import infer_column_type, save_upload

# ─── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Profile XAI API",
    version="1.0.0",
    description="API de explicabilidad adaptada al perfil del usuario (LIME).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Almacén en memoria de trabajos (suficiente para prototipo).
JOB_STORE: Dict[str, Dict[str, Any]] = {}


def _get_job(job_id: str) -> Dict[str, Any]:
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail="Job ID no encontrado.")
    return JOB_STORE[job_id]


# ─── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
async def health():
    """Health-check básico que Render usa para saber si el servicio está vivo."""
    return {"status": "ok", "service": "profile-xai-api"}


@app.post("/api/processing/start")
async def start_processing():
    """Crea un nuevo trabajo y devuelve su ID."""
    jid = f"job_{uuid.uuid4().hex[:8]}"
    JOB_STORE[jid] = {"dataset_csv": None, "model_path": None}
    return {"jobId": jid, "status": "ready"}


@app.post("/api/upload/{upload_type}")
async def upload_files(
    upload_type: str,
    jobId: str,
    files: List[UploadFile] = File(...),
):
    """Sube archivos de dataset (.csv) o modelo (.pkl / .h5)."""
    job = _get_job(jobId)
    folder = os.path.join(settings.UPLOAD_DIR, jobId, upload_type)

    saved: list[dict] = []
    for f in files:
        path = await save_upload(f, folder)
        saved.append({"name": f.filename, "path": path})

        if upload_type == "dataset" and f.filename.endswith(".csv"):
            job["dataset_csv"] = path
        elif upload_type == "model" and f.filename.endswith((".pkl", ".h5")):
            job["model_path"] = path

    return {"jobId": jobId, "uploaded": saved}


@app.get("/api/dataset/schema")
async def dataset_schema(jobId: str):
    """Devuelve el esquema del dataset (nombre, tipo y opciones de cada columna)."""
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


@app.get("/api/dataset/random-instance")
async def random_instance(jobId: str):
    """Devuelve una fila aleatoria del dataset para prellenar el formulario."""
    job = _get_job(jobId)
    if not job.get("dataset_csv"):
        raise HTTPException(status_code=400, detail="Dataset no subido.")

    df = pd.read_csv(job["dataset_csv"], nrows=500)
    row = df.sample(1).iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


@app.post("/api/explain")
async def explain(payload: dict):
    """Genera la predicción + explicación LIME adaptada al perfil del usuario."""
    job_id = payload.get("jobId")
    instance_dict = payload.get("instance", {})
    profile = payload.get("profile", "non-expert")

    if not instance_dict:
        raise HTTPException(status_code=400, detail="El formulario llegó vacío.")

    job = _get_job(job_id)
    if not job.get("dataset_csv") or not job.get("model_path"):
        raise HTTPException(
            status_code=400,
            detail="Faltan archivos: asegúrate de subir dataset y modelo.",
        )

    df_raw = pd.read_csv(job["dataset_csv"])
    model = joblib.load(job["model_path"])
    features = list(instance_dict.keys())

    # --- Codificación de categorías para LIME ---
    label_encoders: dict[str, LabelEncoder] = {}
    categorical_idx: list[int] = []
    df_lime = df_raw[features].copy()

    for i, col in enumerate(features):
        if df_lime[col].dtype == "object" or df_lime[col].dtype.name == "category":
            le = LabelEncoder()
            df_lime[col] = le.fit_transform(df_lime[col].fillna("").astype(str))
            label_encoders[col] = le
            categorical_idx.append(i)
        else:
            df_lime[col] = df_lime[col].fillna(0)

    # Instancia codificada
    inst_df = pd.DataFrame([instance_dict])
    for col, le in label_encoders.items():
        val = str(inst_df[col].iloc[0])
        inst_df[col] = le.transform([val]) if val in le.classes_ else 0

    # Wrapper de predicción
    def _predict(numpy_data: np.ndarray) -> np.ndarray:
        tmp = pd.DataFrame(numpy_data, columns=features)
        for col in features:
            if col in label_encoders:
                tmp[col] = label_encoders[col].inverse_transform(tmp[col].astype(int))
            else:
                tmp[col] = tmp[col].astype(df_raw[col].dtype)
        return model.predict_proba(tmp)

    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=df_lime.values,
            feature_names=features,
            categorical_features=categorical_idx,
            mode="classification",
        )
        exp = explainer.explain_instance(
            data_row=inst_df.iloc[0].values,
            predict_fn=_predict,
            num_features=10,
        )

        # Predicción real (sin codificar)
        df_real = pd.DataFrame([instance_dict])
        prediccion = str(model.predict(df_real)[0])
        confianza = float(np.max(model.predict_proba(df_real)[0])) * 100

        # Explicación natural adaptada al perfil
        sorted_feats = sorted(exp.as_list(), key=lambda x: abs(x[1]), reverse=True)
        top_feat = sorted_feats[0][0] if sorted_feats else "desconocida"

        if profile == "domain-expert":
            texto = (
                f"El modelo ha clasificado la instancia como '{prediccion}' "
                f"con una confianza del {confianza:.2f}%.\n\n"
                f"Desde una perspectiva analítica, el factor determinante ha sido "
                f"'{top_feat}', el cual presenta el mayor peso en la decisión local "
                f"de LIME. Las métricas de importancia sugieren evaluar cuidadosamente "
                f"esta variable para validar la consistencia del resultado."
            )
        else:
            texto = (
                f"¡Hola! El sistema ha analizado los datos y concluye que este caso "
                f"corresponde a '{prediccion}'.\n\n"
                f"La característica que más influyó fue '{top_feat}'. "
                f"En términos sencillos: si el valor de '{top_feat}' fuera diferente, "
                f"es muy probable que el resultado también hubiera cambiado."
            )

        return {
            "prediction": prediccion,
            "profile": profile,
            "natural": texto,
            "technical": {
                "lime": [
                    {"feature": f, "weight": float(w)} for f, w in exp.as_list()
                ],
                "metrics": [{"name": "Confianza", "value": f"{confianza:.2f}%"}],
            },
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
