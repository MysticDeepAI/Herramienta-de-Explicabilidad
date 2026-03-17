# ProfileXAI API

API de explicabilidad adaptada al perfil del usuario.  
Sube un dataset (`.csv`) y un modelo entrenado (`.pkl`), selecciona una instancia y obtén predicciones acompañadas de explicaciones **SHAP, LIME y Anchor** ajustadas al nivel del usuario (ML engineer / domain expert / non-expert), enriquecidas con **Vertex AI RAG Engine** y generación con **LangChain + Gemini 2.5**.

🚀 **Demo en vivo:** [https://v0-profile-xai-wizard.vercel.app/](https://v0-profile-xai-wizard.vercel.app/)

📂 **Archivos de prueba:** En la carpeta [`test_data/`](./test_data/) encontrarás un dataset, un modelo y un PDF de base de conocimiento listos para probar el sistema.

---

## Arquitectura

```
app/
├── config.py          ← Configuración vía variables de entorno
├── modeling.py        ← ModelExplainer (carga pipelines sklearn)
├── explanation.py     ← ExplanationEngine (SHAP + LIME + Anchor + métricas)
├── rag.py             ← Vertex AI RAG Engine (ingesta/retrieval) + LangChain (generación/chat)
├── utils.py           ← Funciones auxiliares
└── main.py            ← FastAPI (endpoints)
```

## Stack tecnológico

| Componente | Tecnología |
| --- | --- |
| **Ingesta & Indexación** | Vertex AI RAG Engine (chunking, embedding, vector DB gestionados) |
| **Retrieval** | Vertex AI RAG Engine `retrieval_query` |
| **Generación de narrativas** | LangChain + Gemini 2.5 Flash |
| **Chat interactivo** | LangChain (prompts + memoria) + Gemini 2.5 |
| **Explicadores XAI** | SHAP (KernelExplainer), LIME, Anchor |
| **Selección automática** | Métricas: Infidelity, Lipschitz, Effective Complexity |
| **API** | FastAPI |

## Ejecución local

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configurar Google Cloud
export GCP_PROJECT_ID="tu-project-id"
export GCP_LOCATION="us-central1"
export GEMINI_API_KEY="tu-api-key"

# Autenticarse con Google Cloud
gcloud auth application-default login

uvicorn app.main:app --reload --port 8000
```

## Deploy en Render

1. Sube el repositorio a GitHub.
2. En [render.com](https://render.com) → **New** → **Blueprint** → conecta el repo.
3. En el dashboard del servicio, agrega las variables de entorno:
   - `GCP_PROJECT_ID` — tu proyecto de Google Cloud
   - `GEMINI_API_KEY` — API key de Gemini
   - `GOOGLE_APPLICATION_CREDENTIALS` — ruta al JSON de Service Account (si aplica)
4. Render lee `render.yaml` y despliega automáticamente.

## Variables de entorno

| Variable | Default | Descripción |
| --- | --- | --- |
| `PORT` | `8000` | Render lo inyecta automáticamente |
| `UPLOAD_DIR` | `/tmp/uploads` | Carpeta para archivos subidos |
| `CORS_ORIGIN` | `*` | Orígenes permitidos |
| `GCP_PROJECT_ID` | — | ID del proyecto en Google Cloud |
| `GCP_LOCATION` | `us-central1` | Región de Vertex AI |
| `GEMINI_API_KEY` | — | API key de Google Gemini |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Modelo de Gemini |
| `RAG_EMBEDDING_MODEL` | `publishers/google/models/text-embedding-005` | Modelo de embeddings |

## Endpoints

| Método | Ruta | Descripción |
| --- | --- | --- |
| GET | `/` | Health check |
| POST | `/api/processing/start` | Crear nuevo trabajo |
| POST | `/api/upload/{upload_type}` | Subir dataset, modelo o knowledge base |
| GET | `/api/dataset/schema` | Esquema del CSV |
| GET | `/api/dataset/random-instance` | Instancia aleatoria |
| POST | `/api/explain` | Predicción + explicación XAI + narrativa RAG |
| POST | `/api/chat` | Chat conversacional con RAG |
