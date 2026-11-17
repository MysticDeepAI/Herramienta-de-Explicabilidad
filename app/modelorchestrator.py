import warnings
import joblib
import re
import sys
import os
import sklearn
import subprocess
from typing import Optional

# Activar SafeUnpickler antes de cualquier joblib.load() en este módulo
from app.security import patch_safe_loader as _patch
_patch()


# ── Información completa del modelo ─────────────────────────────────────────

def detect_model_info(model_path: str) -> dict:
    """
    Detecta el framework y versiones del modelo sin cargarlo (inspección binaria).
    Fallback: carga el modelo para obtener la versión LGB del booster interno.

    Devuelve:
        {
            "framework":        "sklearn" | "lgb" | "sklearn+lgb",
            "sklearn_version":  "1.4.2"  | None,
            "lgb_version_spec": ">=4.0.0,<5.0.0" | None,  # para pip install
        }
    """
    info = {
        "framework": "sklearn",
        "sklearn_version": None,
        "lgb_version_spec": None,
    }
    raw_data = b""

    # ── PASS 1: inspección binaria ───────────────────────────────────────────
    try:
        with open(model_path, "rb") as f:
            raw_data = f.read(15 * 1024 * 1024)  # 15 MB

        # sklearn version
        sk_match = re.search(
            rb"_sklearn_version.*?([0-9]+\.[0-9]+\.[0-9a-zA-Z\.]+)", raw_data
        )
        if sk_match:
            info["sklearn_version"] = sk_match.group(1).decode()

        # LGB markers en el binario
        lgb_present = any(
            marker in raw_data
            for marker in (b"LGBMClassifier", b"LGBMRegressor", b"lightgbm.sklearn")
        )

        if lgb_present:
            # LGB no almacena versión exacta en el binario (solo major: "version=v4")
            v_match = re.search(rb"version=v([0-9]+)", raw_data)
            if v_match:
                major = int(v_match.group(1).decode())
                info["lgb_version_spec"] = f">={major}.0.0,<{major + 1}.0.0"
            # Framework
            info["framework"] = "sklearn+lgb" if info["sklearn_version"] else "lgb"

    except Exception as e:
        print(f"[Warn] Inspección binaria falló: {e}")

    # ── PASS 2: carga del modelo para confirmar LGB y refinar versión ────────
    # Solo si sospechamos LGB pero no obtuvimos versión del binario
    if info["framework"] != "sklearn" and info["lgb_version_spec"] is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = joblib.load(model_path)

            import lightgbm as lgb
            from sklearn.pipeline import Pipeline

            booster = None
            if isinstance(model, lgb.Booster):
                booster = model
                info["framework"] = "lgb"
            elif isinstance(model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
                booster = getattr(model, "booster_", None)
                info["framework"] = "lgb"
            elif isinstance(model, Pipeline):
                for _, step in model.steps:
                    if isinstance(step, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
                        booster = getattr(step, "booster_", None)
                        info["framework"] = "sklearn+lgb"
                        break

            if booster is not None:
                v_match = re.search(r"version=v([0-9]+)", booster.model_to_string()[:200])
                if v_match:
                    major = int(v_match.group(1))
                    info["lgb_version_spec"] = f">={major}.0.0,<{major + 1}.0.0"

        except ImportError:
            # LGB no instalado en el orquestador — usamos major detectado o fallback
            if info["lgb_version_spec"] is None:
                print("[Warn] lightgbm no disponible en orquestador; usando >=4.0.0 como fallback.")
                info["lgb_version_spec"] = ">=4.0.0,<5.0.0"
        except Exception as e:
            print(f"[Warn] Carga de modelo para info LGB falló: {e}")
            if info["lgb_version_spec"] is None:
                info["lgb_version_spec"] = ">=4.0.0,<5.0.0"

    # ── PASS 3 (sklearn fallback): si no logramos ninguna versión, usar la activa
    if info["framework"] in ("sklearn", "sklearn+lgb") and not info["sklearn_version"]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                joblib.load(model_path)
                for warning in w:
                    m = re.search(r"from version (\d+\.\d+(?:\.\d+)?)", str(warning.message))
                    if m:
                        info["sklearn_version"] = m.group(1)
                        break
            except Exception:
                pass
        if not info["sklearn_version"]:
            info["sklearn_version"] = sklearn.__version__

    return info


# ── Orquestación del pipeline de conversión ──────────────────────────────────

def orchestrate_conversion(input_path: str, output_path: str, dataset_path: str) -> None:
    print(f"\n--- Iniciando pipeline agnóstico para: {input_path} ---")

    if not os.path.exists(input_path):
        print(f"[Error] No se encontró el archivo: {input_path}")
        sys.exit(1)

    # 1. Detección de framework y versiones
    info = detect_model_info(input_path)
    framework = info["framework"]
    sklearn_version = info["sklearn_version"]
    lgb_version_spec = info["lgb_version_spec"] or ""

    print(f"[+] Framework detectado: {framework}")
    if sklearn_version:
        print(f"[+] scikit-learn: {sklearn_version}")
    if lgb_version_spec:
        print(f"[+] lightgbm spec: {lgb_version_spec}")

    # 2. Nombre único de imagen (evita colisiones entre versiones)
    sk_tag = f"sk{sklearn_version}" if sklearn_version else "skX"
    lgb_tag = f"_lgb{lgb_version_spec.split(',')[0].replace('>=','').replace('.0.0','')}" \
              if lgb_version_spec else ""
    image_name = f"sandbox_conversor:{sk_tag}{lgb_tag}"

    # 3. Contexto del build (directorio donde vive este script)
    app_dir = os.path.dirname(os.path.abspath(__file__))

    # 4. Construir el Sandbox
    print(f"[+] Construyendo entorno aislado: {image_name} ...")
    build_command = [
        "docker", "build",
        "-f", os.path.join(app_dir, "Dockerfile.sandbox"),
        "--build-arg", f"FRAMEWORK={framework}",
        "--build-arg", f"SKLEARN_VERSION={sklearn_version or ''}",
        "--build-arg", f"LGB_VERSION_SPEC={lgb_version_spec}",
        "-t", image_name,
        app_dir,
    ]
    subprocess.run(build_command, check=True)

    # 5. Ejecutar conversión en el Sandbox
    print("[+] Entorno listo. Ejecutando contenedor Sandbox de conversión...")
    run_command = [
        "docker", "run", "--rm",
        "-v", "xai_shared_data:/shared_uploads",
        image_name,
        input_path,
        output_path,
        dataset_path,
    ]
    subprocess.run(run_command, check=True)
    print("[+] Conversión exitosa. Saliendo del orquestador.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python modelorchestrator.py <modelo.pkl> <salida.onnx> <dataset.csv>")
        sys.exit(1)
    orchestrate_conversion(sys.argv[1], sys.argv[2], sys.argv[3])
