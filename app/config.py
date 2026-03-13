"""
Configuración centralizada — lee valores de variables de entorno
para que Render (u otro PaaS) pueda inyectarlos sin tocar código.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Settings:
    # Directorio donde se guardan los archivos subidos.
    # En Render, el disco efímero permite escritura en /tmp.
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/tmp/uploads")

    # Puerto (Render inyecta $PORT automáticamente).
    PORT: int = int(os.getenv("PORT", "8000"))

    # Orígenes CORS permitidos.  Por defecto acepta todo (*);
    # en producción pondrías tu dominio de frontend.
    CORS_ORIGINS: List[str] = field(default_factory=lambda: [
        os.getenv("CORS_ORIGIN", "*")
    ])


settings = Settings()
