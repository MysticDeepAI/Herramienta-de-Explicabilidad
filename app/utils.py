"""
Funciones auxiliares reutilizables.
"""

import os
from fastapi import UploadFile
import pandas as pd


def infer_column_type(series: pd.Series) -> str:
    """Determina si una columna es 'numerical', 'categorical' o 'text'."""
    nunique = series.dropna().nunique()

    if pd.api.types.is_numeric_dtype(series):
        if nunique <= 10 and (series.dropna() % 1 == 0).all():
            return "categorical"
        return "numerical"

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return "categorical" if nunique <= 20 else "text"

    return "text"


async def save_upload(file: UploadFile, folder: str) -> str:
    """Guarda un archivo subido en *folder* y devuelve la ruta resultante."""
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, os.path.basename(file.filename))
    with open(out_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    await file.seek(0)
    return out_path
