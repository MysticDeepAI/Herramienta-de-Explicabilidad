"""
Conversor .pkl → .onnx ejecutado dentro del Sandbox Docker.

Soporta:
  - sklearn Pipeline / estimador puro                → skl2onnx
  - LightGBM Booster puro                           → onnxmltools
  - sklearn Pipeline cuyo paso final es LGBMClassifier/Regressor
                                                     → skl2onnx + onnxmltools
"""

import json
import os
import sys

import joblib
import pandas as pd
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType, StringTensorType


# ── Detección del framework en runtime ──────────────────────────────────────

def _detect_framework(model) -> str:
    """
    Devuelve:
      "sklearn"      — estimador o Pipeline sin LGB
      "lgb_booster"  — lgb.Booster puro
      "sklearn+lgb"  — sklearn Pipeline cuyo paso final es LGBM*
    """
    try:
        import lightgbm as lgb
        if isinstance(model, lgb.Booster):
            return "lgb_booster"
        if isinstance(model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
            return "lgb_sklearn"
    except ImportError:
        pass

    try:
        from sklearn.pipeline import Pipeline
        import lightgbm as lgb
        if isinstance(model, Pipeline):
            for _, step in model.steps:
                if isinstance(step, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
                    return "sklearn+lgb"
    except ImportError:
        pass

    return "sklearn"


# ── Construcción de initial_types ────────────────────────────────────────────

def _build_initial_types(model, df: pd.DataFrame) -> list:
    """
    Construye initial_types para skl2onnx según los tipos del CSV.
    Preserva la lógica original: String / Int64 / Float por columna.
    """
    if hasattr(model, "feature_names_in_"):
        nombres = model.feature_names_in_
        tipos = []
        for nombre in nombres:
            dtype = df[nombre].dtype if nombre in df.columns else "float64"
            if dtype == "object" or str(dtype) == "category":
                tipos.append((str(nombre), StringTensorType([None, 1])))
            elif dtype == "int64":
                tipos.append((str(nombre), Int64TensorType([None, 1])))
            else:
                tipos.append((str(nombre), FloatTensorType([None, 1])))
        return tipos

    n = getattr(model, "n_features_in_", df.shape[1])
    return [("float_input", FloatTensorType([None, n]))]


# ── Conversores por framework ─────────────────────────────────────────────────

def _convert_sklearn(model, initial_types: list, output_path: str) -> None:
    from skl2onnx import convert_sklearn
    onnx_model = convert_sklearn(model, initial_types=initial_types, options={"zipmap": False})
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"¡Conversión exitosa (sklearn)! ONNX guardado en: {output_path}")


def _convert_sklearn_lgb(model, initial_types: list, output_path: str) -> None:
    """sklearn Pipeline con LGBMClassifier/Regressor como paso final."""
    import lightgbm as lgb
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
    from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm

    # Registrar convertidor LGB classifier
    update_registered_converter(
        lgb.LGBMClassifier,
        "LightGbmLGBMClassifier",
        calculate_linear_classifier_output_shapes,
        convert_lightgbm,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )

    # Registrar regressor si está disponible en esta versión de skl2onnx
    try:
        from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes
        update_registered_converter(
            lgb.LGBMRegressor,
            "LightGbmLGBMRegressor",
            calculate_linear_regressor_output_shapes,
            convert_lightgbm,
            options={"nocl": [True, False]},
        )
    except ImportError:
        pass

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        options={lgb.LGBMClassifier: {"zipmap": False}},
        target_opset={"": 17, "ai.onnx.ml": 3},
    )
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"¡Conversión exitosa (sklearn+LGB)! ONNX guardado en: {output_path}")


def _convert_lgb_booster(model, n_features: int, output_path: str) -> None:
    """lgb.Booster puro (sin sklearn wrapper)."""
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloat

    initial_types = [("float_input", OnnxFloat([None, n_features]))]
    onnx_model = onnxmltools.convert_lightgbm(
        model, initial_types=initial_types, target_opset=12
    )
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"¡Conversión exitosa (LGB Booster)! ONNX guardado en: {output_path}")


# ── Extracción de metadatos ───────────────────────────────────────────────────

def _extract_metadata(model, framework: str) -> dict:
    """Extrae task, n_classes y classes del modelo."""
    if hasattr(model, "classes_"):
        classes = [c.item() if hasattr(c, "item") else c for c in model.classes_]
        return {"task": "classification", "n_classes": len(classes), "classes": classes}

    # LGB Booster puro
    if framework == "lgb_booster":
        try:
            import lightgbm as lgb
            n_iter = model.num_model_per_iteration()
            n_classes = 2 if n_iter == 1 else n_iter
            return {"task": "classification", "n_classes": n_classes,
                    "classes": list(range(n_classes))}
        except Exception:
            pass

    if hasattr(model, "n_classes_"):
        n = int(model.n_classes_)
        return {"task": "classification", "n_classes": n,
                "classes": list(range(n))}

    print("[!] No se detectaron clases. Metadata indicará tarea desconocida.")
    return {"task": "unknown"}


# ── Punto de entrada principal ────────────────────────────────────────────────

def convertir_modelo(ruta_entrada: str, ruta_salida: str, ruta_dataset: str) -> None:
    print(f"Cargando modelo desde: {ruta_entrada}")
    try:
        model = joblib.load(ruta_entrada)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        sys.exit(1)

    print(f"Analizando tipos de datos desde: {ruta_dataset}")
    try:
        df = pd.read_csv(ruta_dataset, sep=None, engine="python", nrows=5)
        if "target" in df.columns:
            df = df.drop(columns=["target"])
    except Exception as e:
        print(f"Error al leer el CSV: {e}")
        sys.exit(1)

    framework = _detect_framework(model)
    print(f"Framework detectado: {framework}")

    initial_types = _build_initial_types(model, df)

    print("Convirtiendo a ONNX...")
    if framework == "lgb_booster":
        n_features = getattr(model, "num_feature", lambda: df.shape[1])()
        _convert_lgb_booster(model, n_features, ruta_salida)
    elif framework == "sklearn+lgb":
        _convert_sklearn_lgb(model, initial_types, ruta_salida)
    else:
        # "sklearn" y "lgb_sklearn" (wrapper sklearn puro)
        _convert_sklearn(model, initial_types, ruta_salida)

    # Guardar metadata
    metadata = _extract_metadata(model, framework)
    metadata_path = os.path.splitext(ruta_salida)[0] + ".metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Metadatos: {metadata.get('n_classes', '?')} clases → {metadata.get('classes', [])}")
    print(f"[+] Metadata guardada en: {metadata_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso interno: python conver.py <input.pkl> <output.onnx> <dataset.csv>")
        sys.exit(1)
    convertir_modelo(sys.argv[1], sys.argv[2], sys.argv[3])
