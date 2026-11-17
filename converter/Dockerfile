# Microservicio converter — tiene acceso al socket de Docker (solo este contenedor)
# Construye y ejecuta el sandbox de conversión .pkl → .onnx
# Contexto de build: raíz del proyecto (para acceder a app/ y converter/)

FROM python:3.11-slim
WORKDIR /src

RUN apt-get update && apt-get install -y build-essential curl docker.io && rm -rf /var/lib/apt/lists/*

# Dependencias del microservicio (FastAPI + runtime mínimo)
COPY converter/requirements.txt /tmp/converter_requirements.txt
RUN pip install --no-cache-dir -r /tmp/converter_requirements.txt

# Código de la app (modelorchestrator, security, conver.py, Dockerfile.sandbox)
COPY app/ /src/app/
COPY converter/ /src/converter/

ENV PYTHONPATH=/src

CMD ["uvicorn", "converter.main:app", "--host", "0.0.0.0", "--port", "8001"]
