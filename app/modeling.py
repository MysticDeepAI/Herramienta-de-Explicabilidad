"""
ModelExplainer — Carga modelos sklearn (pipeline completo o modelo + preprocesador separados)
y prepara los datos de fondo para los explicadores XAI.
"""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class ModelExplainer:
    """
    Carga un pipeline sklearn y prepara datos de fondo para SHAP/LIME/Anchor.

    Soporta dos modos:
      1. Pipeline completo (.joblib con Pipeline sklearn)
      2. Preprocesador + modelo por separado
    """

    def __init__(
        self,
        pipeline_path: Union[str, Path],
        background_data: Union[str, Path, pd.DataFrame, np.ndarray],
        model_path: Optional[Union[str, Path]] = None,
        feature_names: Optional[List[str]] = None,
        has_header: bool = True,
        split_xy: bool = False,
        target_column: Optional[str] = None,
    ):
        # ── 1. Cargar modelo/pipeline ────────────────────────────────────
        if model_path:
            loaded_pipe = joblib.load(pipeline_path)
            if isinstance(loaded_pipe, Pipeline):
                clean_steps = [
                    (name, step)
                    for name, step in loaded_pipe.steps
                    if hasattr(step, "transform")
                ]
                self.preprocessor = Pipeline(clean_steps)
            else:
                self.preprocessor = loaded_pipe

            self.model = joblib.load(model_path)
            self.pipeline = None
        else:
            full_pipe = joblib.load(pipeline_path)
            if not isinstance(full_pipe, Pipeline):
                # Si no es Pipeline, asumimos que es el modelo directamente
                self.pipeline = None
                self.preprocessor = None
                self.model = full_pipe
            else:
                steps = []
                last_idx = len(full_pipe.steps) - 1
                for idx, (name, step) in enumerate(full_pipe.steps):
                    if hasattr(step, "transform") or idx == last_idx:
                        steps.append((name, step))

                self.pipeline = Pipeline(steps)
                pre_steps = [(n, s) for n, s in steps[:-1]]
                self.preprocessor = Pipeline(pre_steps) if pre_steps else None
                self.model = steps[-1][1]

        # ── 2. Preparar DataFrame de fondo ───────────────────────────────
        if isinstance(background_data, (str, Path)):
            if has_header:
                df = pd.read_csv(background_data)
            else:
                df = pd.read_csv(background_data, header=None, names=feature_names)
        elif isinstance(background_data, pd.DataFrame):
            df = background_data.copy()
        elif isinstance(background_data, np.ndarray):
            df = pd.DataFrame(background_data, columns=feature_names)
        else:
            raise ValueError(f"Tipo no soportado para background_data: {type(background_data)}")

        # ── 3. Separar X / y si se solicita ──────────────────────────────
        if split_xy:
            if target_column is None:
                raise ValueError("target_column requerido si split_xy=True")
            self.y_background = df[target_column]
            self.X_background = df.drop(columns=[target_column])
        else:
            self.X_background = df
            self.y_background = None

        self.feature_names = list(self.X_background.columns)

    # ── Métodos de predicción ────────────────────────────────────────────

    def predict(self, X_raw: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.pipeline is not None:
            return self.pipeline.predict(X_raw)
        return self.model.predict(X_raw)

    def predict_proba(self, X_raw: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.pipeline is not None:
            return self.pipeline.predict_proba(X_raw)
        return self.model.predict_proba(X_raw)

    def preprocess(self, x_raw: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Aplica preprocesamiento y devuelve array 1D."""
        if self.preprocessor is None:
            return np.array(x_raw).flatten()
        arr = np.array(x_raw).reshape(1, -1)
        return self.preprocessor.transform(arr)[0]
