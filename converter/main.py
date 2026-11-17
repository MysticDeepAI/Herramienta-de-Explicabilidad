"""
ProfileXAI Converter — microservicio de conversión .pkl/.joblib → .onnx.

Usado solo con CONVERTER_BACKEND=docker (desarrollo local / docker-compose).
En producción (Cloud Run) este servicio no existe; su rol lo toma Cloud Build.
"""

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ProfileXAI Converter", version="1.0.0")


class ConvertRequest(BaseModel):
    model_path: str
    onnx_path: str
    dataset_path: str


@app.get("/health")
async def health():
    return {"status": "ok", "service": "profilexai-converter"}


@app.post("/convert")
async def convert(req: ConvertRequest):
    from app.modelorchestrator import orchestrate_conversion

    if not Path(req.model_path).exists():
        raise HTTPException(status_code=400, detail=f"Modelo no encontrado: {req.model_path}")
    if not Path(req.dataset_path).exists():
        raise HTTPException(status_code=400, detail=f"Dataset no encontrado: {req.dataset_path}")

    try:
        logger.info("Iniciando conversión: %s → %s", req.model_path, req.onnx_path)
        await asyncio.to_thread(
            orchestrate_conversion, req.model_path, req.onnx_path, req.dataset_path
        )
        logger.info("Conversión exitosa: %s", req.onnx_path)
        return {"status": "ok", "onnx_path": req.onnx_path}
    except Exception as e:
        logger.error("Error en conversión: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Conversión fallida: {e}")
