# Herramienta de Explicabilidad

API de explicabilidad adaptada al perfil del usuario.  
Sube un dataset (`.csv`) y un modelo entrenado (`.pkl`), selecciona una instancia y obtén predicciones acompañadas de explicaciones LIME ajustadas al nivel del usuario (experto / no-experto).

Puede probar el servicio desde este enlace: https://v0-profile-xai-wizard.vercel.app/

**Archivos de prueba:** En la carpeta [`test_data/`](./test_data/) encontrarás un dataset, un modelo y un PDF de base de conocimiento listos para probar el sistema.

---

## Estructura del proyecto

```
profile-xai-api/
├── app/
│   ├── __init__.py
│   ├── config.py      ← Configuración vía variables de entorno
│   ├── main.py         ← Aplicación FastAPI (endpoints)
│   └── utils.py        ← Funciones auxiliares
├── .gitignore
├── render.yaml          ← Blueprint para deploy con un clic en Render
├── requirements.txt
└── README.md
```

## Ejecución local

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Iniciar el servidor
uvicorn app.main:app --reload --port 8000
```

Abre `http://localhost:8000/docs` para ver la documentación interactiva (Swagger).

## Deploy en Render

1. Sube el repositorio a GitHub.
2. Ve a [render.com](https://render.com) → **New** → **Blueprint**.
3. Conecta tu repositorio; Render detectará `render.yaml` automáticamente.
4. Haz clic en **Apply** y espera a que el servicio se construya.

Render te dará una URL pública tipo `https://profile-xai-api.onrender.com`.  
Apunta tu frontend a esa URL y listo — **sin ngrok, sin túneles**.

> **Nota:** el plan gratuito de Render pone el servicio en reposo tras 15 min de inactividad. La primera petición después de eso tarda ~30 s en responder mientras se levanta de nuevo.

## Variables de entorno

| Variable       | Default         | Descripción                              |
| -------------- | --------------- | ---------------------------------------- |
| `PORT`         | `8000`          | Render lo inyecta automáticamente        |
| `UPLOAD_DIR`   | `/tmp/uploads`  | Carpeta para archivos subidos            |
| `CORS_ORIGIN`  | `*`             | Orígenes permitidos (dominio frontend)   |

## Endpoints principales

| Método | Ruta                            | Descripción                          |
| ------ | ------------------------------- | ------------------------------------ |
| GET    | `/`                             | Health check                         |
| POST   | `/api/processing/start`         | Crear un nuevo trabajo               |
| POST   | `/api/upload/{upload_type}`     | Subir dataset o modelo               |
| GET    | `/api/dataset/schema`           | Esquema del CSV (columnas y tipos)   |
| GET    | `/api/dataset/random-instance`  | Instancia aleatoria para formulario  |
| POST   | `/api/explain`                  | Predicción + explicación LIME        |
