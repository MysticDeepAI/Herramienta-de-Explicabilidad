"""
Cliente de conversión de modelos — abstracción sobre dos backends:

  CONVERTER_BACKEND=docker
      → HTTP al microservicio 'converter' (docker-compose, local)
      → El converter tiene el socket de Docker; el backend no

  CONVERTER_BACKEND=cloudbuild
      → Google Cloud Build API (Cloud Run, producción)
      → Sin socket en ningún contenedor; Cloud Build ejecuta los pasos Docker
"""

from __future__ import annotations

import logging
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Interfaz común ────────────────────────────────────────────────────────────

class ConverterBackend(ABC):
    @abstractmethod
    async def convert(self, model_path: str, onnx_path: str, dataset_path: str) -> str:
        """Convierte model_path → onnx_path. Devuelve la ruta final del .onnx."""


# ── Backend 1: microservicio Docker (local) ───────────────────────────────────

class _DockerConverterBackend(ConverterBackend):
    """
    Delega la conversión al microservicio 'converter' via HTTP interno.
    El converter es el único contenedor que tiene montado el socket de Docker.
    """

    def __init__(self, url: str):
        self._url = url.rstrip("/")

    async def convert(self, model_path: str, onnx_path: str, dataset_path: str) -> str:
        import httpx

        payload = {
            "model_path": model_path,
            "onnx_path": onnx_path,
            "dataset_path": dataset_path,
        }
        logger.info("Enviando conversión al microservicio converter: %s", self._url)

        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(f"{self._url}/convert", json=payload)
            response.raise_for_status()

        result = response.json()
        logger.info("Converter respondió: %s", result.get("status"))
        return result["onnx_path"]


# ── Backend 2: Google Cloud Build (producción) ────────────────────────────────

class _CloudBuildConverterBackend(ConverterBackend):
    """
    Convierte modelos usando Google Cloud Build + Cloud Storage.

    Flujo:
      1. Sube model + dataset a GCS
      2. Envía un job a Cloud Build con los mismos pasos Docker que el sandbox local
         (docker build Dockerfile.sandbox → docker run conversión)
      3. Cloud Build sube el .onnx resultante a GCS
      4. Descarga el .onnx al filesystem local del backend
      5. Limpia los archivos temporales de GCS
    """

    def __init__(self, project_id: str, gcs_bucket: str, location: str = "us-central1"):
        self._project_id = project_id
        self._gcs_bucket = gcs_bucket
        self._location = location

    async def convert(self, model_path: str, onnx_path: str, dataset_path: str) -> str:
        import asyncio
        return await asyncio.to_thread(
            self._convert_sync, model_path, onnx_path, dataset_path
        )

    def _convert_sync(self, model_path: str, onnx_path: str, dataset_path: str) -> str:
        from google.cloud import storage  # type: ignore
        from google.cloud.devtools import cloudbuild_v1  # type: ignore

        from app.modelorchestrator import detect_model_info

        job_id = uuid.uuid4().hex[:8]
        model_suffix = Path(model_path).suffix

        # Paths en GCS (temporales, se borran al finalizar)
        gcs_model = f"conversions/{job_id}/model{model_suffix}"
        gcs_dataset = f"conversions/{job_id}/dataset.csv"
        gcs_onnx = f"conversions/{job_id}/model_agnostico.onnx"

        storage_client = storage.Client()
        bucket = storage_client.bucket(self._gcs_bucket)

        # 1. Subir archivos fuente a GCS
        logger.info("CloudBuild: subiendo archivos a gs://%s ...", self._gcs_bucket)
        bucket.blob(gcs_model).upload_from_filename(model_path)
        bucket.blob(gcs_dataset).upload_from_filename(dataset_path)

        # 2. Detectar versiones del modelo (mismo código que el orquestador local)
        info = detect_model_info(model_path)
        framework = info["framework"]
        sklearn_version = info["sklearn_version"] or ""
        lgb_version_spec = info["lgb_version_spec"] or ""

        sk_tag = f"sk{sklearn_version}" if sklearn_version else "skX"
        lgb_tag = (
            f"_lgb{lgb_version_spec.split(',')[0].replace('>=', '').replace('.0.0', '')}"
            if lgb_version_spec else ""
        )
        image_name = f"gcr.io/{self._project_id}/sandbox_conversor:{sk_tag}{lgb_tag}"

        # 3. Definir el job de Cloud Build
        # Replica exactamente los pasos que hace docker-compose localmente:
        #   Paso A: docker build del sandbox con las versiones correctas
        #   Paso B: docker run de la conversión dentro del sandbox
        #   Paso C: gsutil cp del .onnx resultante a GCS
        build = cloudbuild_v1.Build(
            timeout={"seconds": 600},
            steps=[
                # Descargar archivos desde GCS al workspace de Cloud Build
                cloudbuild_v1.BuildStep(
                    name="gcr.io/cloud-builders/gsutil",
                    args=["-m", "cp",
                          f"gs://{self._gcs_bucket}/{gcs_model}",
                          f"/workspace/model{model_suffix}"],
                ),
                cloudbuild_v1.BuildStep(
                    name="gcr.io/cloud-builders/gsutil",
                    args=["-m", "cp",
                          f"gs://{self._gcs_bucket}/{gcs_dataset}",
                          "/workspace/dataset.csv"],
                ),
                # Construir imagen sandbox con las versiones exactas del modelo
                cloudbuild_v1.BuildStep(
                    name="gcr.io/cloud-builders/docker",
                    args=[
                        "build",
                        "--build-arg", f"FRAMEWORK={framework}",
                        "--build-arg", f"SKLEARN_VERSION={sklearn_version}",
                        "--build-arg", f"LGB_VERSION_SPEC={lgb_version_spec}",
                        "-t", image_name,
                        "-f", "app/Dockerfile.sandbox",
                        ".",
                    ],
                ),
                # Ejecutar la conversión dentro del sandbox
                cloudbuild_v1.BuildStep(
                    name=image_name,
                    args=[
                        f"/workspace/model{model_suffix}",
                        "/workspace/model_agnostico.onnx",
                        "/workspace/dataset.csv",
                    ],
                ),
                # Subir el .onnx resultante a GCS
                cloudbuild_v1.BuildStep(
                    name="gcr.io/cloud-builders/gsutil",
                    args=["cp",
                          "/workspace/model_agnostico.onnx",
                          f"gs://{self._gcs_bucket}/{gcs_onnx}"],
                ),
            ],
            # Cachear la imagen del sandbox en GCR para builds futuras más rápidas
            images=[image_name],
        )

        # 4. Enviar job y esperar resultado
        logger.info("CloudBuild: enviando job de conversión (framework=%s)...", framework)
        cb_client = cloudbuild_v1.CloudBuildClient()
        operation = cb_client.create_build(project_id=self._project_id, build=build)
        result = operation.result(timeout=600)

        if result.status.name != "SUCCESS":
            raise RuntimeError(
                f"Cloud Build terminó con estado: {result.status.name}. "
                f"Log: https://console.cloud.google.com/cloud-build/builds/{result.id}"
            )

        # 5. Descargar .onnx al filesystem local del backend
        logger.info("CloudBuild: descargando .onnx desde GCS...")
        bucket.blob(gcs_onnx).download_to_filename(onnx_path)

        # Limpiar archivos temporales de GCS
        for blob_name in [gcs_model, gcs_dataset, gcs_onnx]:
            try:
                bucket.blob(blob_name).delete()
            except Exception:
                pass

        logger.info("CloudBuild: conversión completada → %s", onnx_path)
        return onnx_path


# ── Factory ───────────────────────────────────────────────────────────────────

def get_converter() -> ConverterBackend:
    """
    Devuelve el backend de conversión según la variable de entorno CONVERTER_BACKEND.

      CONVERTER_BACKEND=docker      → microservicio local (default)
      CONVERTER_BACKEND=cloudbuild  → Google Cloud Build
    """
    backend_type = os.getenv("CONVERTER_BACKEND", "docker").lower().strip()

    if backend_type == "cloudbuild":
        from app.config import settings
        if not settings.GCS_BUCKET:
            raise RuntimeError(
                "CONVERTER_BACKEND=cloudbuild requiere la variable GCS_BUCKET configurada."
            )
        logger.info("Usando backend de conversión: Cloud Build (proyecto=%s)", settings.GCP_PROJECT_ID)
        return _CloudBuildConverterBackend(
            project_id=settings.GCP_PROJECT_ID,
            gcs_bucket=settings.GCS_BUCKET,
            location=settings.GCP_LOCATION,
        )

    converter_url = os.getenv("CONVERTER_URL", "http://converter:8001")
    logger.info("Usando backend de conversión: Docker microservicio (%s)", converter_url)
    return _DockerConverterBackend(url=converter_url)
