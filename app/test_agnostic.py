# test_e2e.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ... el resto de tus imports ...
import pandas as pd
# ...

import json
import pandas as pd
from app.modeling import AgnosticModelExplainer
from app.explanation import ExplanationEngine

def simulacion_api_explain():
    ruta_modelo_onnx = "../test_data/MODEL_TEST.onnx"
    ruta_dataset = "../test_data/DATASET_TEST.csv"
    
    print("--- 🚀 Iniciando Simulación Real del Endpoint API ---")
    
    # 1. Cargar datos como lo hace main.py
    df = pd.read_csv(ruta_dataset)
    nombre_columna_target = 'target'
    if nombre_columna_target in df.columns:
        df_features = df.drop(columns=[nombre_columna_target])
    else:
        df_features = df

    # 2. Extraer un paciente real y convertirlo al diccionario que envía el frontend
    paciente_fila = df_features.iloc[0] # Tomamos el primer paciente
    instance_dict = paciente_fila.to_dict()
    
    print(f"[+] Paciente simulado recibido del frontend:\n{instance_dict}\n")

    # 3. Inicializar el motor ONNX (Reemplaza la carga nativa)
    print("[+] Cargando Caja Negra ONNX...")
    explainer_agnostico = AgnosticModelExplainer(
        model_path=ruta_modelo_onnx,
        background_data=df_features
    )

    # 4. Construir el ExplanationEngine con el motor ONNX
    print("[+] Inicializando ExplanationEngine (SHAP, LIME, Anchor)...")
    engine = ExplanationEngine(
        model_explainer=explainer_agnostico,
        target_name="target",
        label_map={0: "Clase 0", 1: "Clase 1"},
        mode="classification"
    )

    # 5. ¡El momento de la verdad! Ejecutar la predicción y explicabilidad
    print("\n[+] Procesando explain_instance() ... (Esto puede tomar unos segundos)")
    resultado_api = engine.explain_instance(instance_dict)

    # 6. Mostrar el resultado final tal como se enviaría en el JSON de respuesta
    print("\n--- ✅ RESPUESTA GENERADA PARA EL FRONTEND ---")
    print(json.dumps(resultado_api, indent=2, ensure_ascii=False))
    
    print("\n[ÉXITO] Si ves las explicaciones arriba, tu backend está listo para producción.")

if __name__ == "__main__":
    simulacion_api_explain()