"""
ExplanationEngine — Genera explicaciones con SHAP, LIME y Anchor,
evalúa métricas (Infidelity, Lipschitz, Effective Complexity) y
selecciona automáticamente el mejor explicador por instancia.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap
from anchor.anchor_tabular import AnchorTabularExplainer
from lime.lime_tabular import LimeTabularExplainer

from app.modeling import ModelExplainer

# ── Compatibilidad sklearn ───────────────────────────────────────────────
try:
    import sklearn.tree._classes as _tree_classes
    if not hasattr(_tree_classes.BaseDecisionTree, "monotonic_cst"):
        _tree_classes.BaseDecisionTree.monotonic_cst = None
except Exception:
    pass


class ExplanationEngine:
    """
    Motor de explicabilidad que soporta SHAP, LIME y Anchor sobre el
    espacio de features original (pre-encoding).
    """

    def __init__(
        self,
        model_explainer: ModelExplainer,
        target_name: str,
        label_map: Dict[int, str],
        mode: str = "classification",
    ):
        if not isinstance(model_explainer, ModelExplainer):
            raise TypeError("model_explainer debe ser instancia de ModelExplainer")

        self.me = model_explainer
        self.target = target_name
        self.label_map = label_map
        self.mode = mode

        self._setup_model()
        self._setup_anchor()
        self._setup_lime()
        self._setup_shap()

    # ────────────────────────────────────────────────────────────────────
    # Setup interno
    # ────────────────────────────────────────────────────────────────────

    def _setup_model(self) -> None:
        df = self.me.X_background.copy()
        self.feature_names = list(df.columns)
        self.class_names = list(self.me.model.classes_)

        self.cat_feats = [
            i for i, col in enumerate(self.feature_names)
            if df[col].dtype == object
            or pd.api.types.is_categorical_dtype(df[col])
        ]
        self.cat_names = {
            idx: list(df.iloc[:, idx].astype("category").cat.categories)
            for idx in self.cat_feats
        }
        self.X_raw_df = df

        df_codes = df.copy()
        for idx in self.cat_feats:
            col = self.feature_names[idx]
            df_codes[col] = df_codes[col].astype("category").cat.codes
        self.train_data = df_codes.values

        self.preproc = self.me.preprocessor if hasattr(self.me, "preprocessor") else None

    def _setup_anchor(self) -> None:
        encoder = None
        if self.preproc is not None:
            encoder = lambda X: self.preproc.transform(
                pd.DataFrame(X, columns=self.feature_names)
            )
        self.anchor_explainer = AnchorTabularExplainer(
            class_names=self.class_names,
            feature_names=self.feature_names,
            train_data=self.train_data,
            categorical_names=self.cat_names,
            **({"encoder_fn": encoder} if encoder else {}),
        )

    def _setup_lime(self) -> None:
        self.lime_explainer = LimeTabularExplainer(
            training_data=self.train_data,
            feature_names=self.feature_names,
            categorical_features=self.cat_feats,
            categorical_names=self.cat_names,
            discretize_continuous=False,
            class_names=self.class_names,
            mode=self.mode,
        )

        def decoder_fn(X_codes: np.ndarray) -> np.ndarray:
            df_codes = pd.DataFrame(X_codes, columns=self.feature_names)
            for idx in self.cat_feats:
                mapping = {i: cat for i, cat in enumerate(self.cat_names[idx])}
                col = self.feature_names[idx]
                df_codes[col] = df_codes[col].astype(int).map(mapping)
            return self.me.predict_proba(df_codes)

        self._lime_predict_fn = decoder_fn

    def _setup_shap(self) -> None:
        bg = shap.kmeans(self.train_data, min(50, len(self.train_data)))
        self.shap_explainer = shap.KernelExplainer(
            model=self._lime_predict_fn,
            data=bg,
            link="logit",
            nsamples=15,
        )

    # ────────────────────────────────────────────────────────────────────
    # Codificación de instancias
    # ────────────────────────────────────────────────────────────────────

    def _to_code_array(self, instance) -> np.ndarray:
        """Convierte cualquier instancia a un ndarray codificado."""
        if isinstance(instance, int):
            return self.train_data[instance].astype(float)

        if isinstance(instance, (pd.Series, dict)):
            x = np.zeros(len(self.feature_names), dtype=float)
            for i, fname in enumerate(self.feature_names):
                val = instance[fname]
                if i in self.cat_feats:
                    cats = self.cat_names[i]
                    try:
                        code = cats.index(val)
                    except ValueError:
                        raise ValueError(f"Categoría desconocida '{val}' en '{fname}'")
                    x[i] = code
                else:
                    x[i] = float(val)
            return x

        arr = np.asarray(instance, dtype=float)
        if arr.ndim == 1 and arr.shape[0] == len(self.feature_names):
            return arr
        raise ValueError(f"No se puede convertir {type(instance)} a vector código")

    def decode_instance_as_list(self, x_numeric: np.ndarray) -> List:
        """Decodifica instancia numérica a valores originales."""
        x = np.asarray(x_numeric).flatten()
        decoded = []
        for idx, fname in enumerate(self.feature_names):
            val = x[idx]
            if idx in self.cat_feats:
                categories = self.cat_names[idx]
                code = int(val)
                if 0 <= code < len(categories):
                    decoded.append(categories[code])
                else:
                    decoded.append(str(val))
            else:
                decoded.append(val)
        return decoded

    # ────────────────────────────────────────────────────────────────────
    # Explicaciones individuales
    # ────────────────────────────────────────────────────────────────────

    def anchor(self, instance, threshold=0.95, delta=0.1, batch_size=50) -> str:
        x = self._to_code_array(instance)
        proba = self._lime_predict_fn(np.array([x]))[0]
        idx_pred = np.argmax(proba)
        predicted_class = self.class_names[idx_pred]
        probability = float(proba[idx_pred])

        exp = self.anchor_explainer.explain_instance(
            x,
            lambda z: self.me.model.predict_proba(z)[:, idx_pred],
            threshold=threshold, delta=delta, batch_size=batch_size,
        )

        return self._generate_explanation(
            instance=str(x), predicted_class=str(predicted_class),
            probability=probability, feature_names=self.feature_names,
            feature_values=x.tolist(), method="anchor",
            anchor_conditions=exp.names(),
            anchor_precision=exp.precision(),
            anchor_coverage=exp.coverage(),
        )

    def lime(self, instance) -> str:
        x = self._to_code_array(instance)
        exp = self.lime_explainer.explain_instance(
            x, self._lime_predict_fn,
            num_features=len(self.feature_names), labels=[1],
        )
        lime_list = exp.as_list(label=1)
        conds, weights = zip(*lime_list)

        proba = self._lime_predict_fn(np.array([x]))[0]
        idx_pred = np.argmax(proba)
        predicted_class = self.class_names[idx_pred]
        probability = float(proba[idx_pred])

        return self._generate_explanation(
            instance=str(x), predicted_class=str(predicted_class),
            probability=probability, feature_names=self.feature_names,
            feature_values=x.tolist(), method="lime",
            lime_conditions=list(conds), lime_weights=list(weights),
        )

    def shap_explain(self, instance, output_index=None) -> str:
        x = self._to_code_array(instance)
        shap_vals = self.shap_explainer(x)
        shap_arr, base_arr = shap_vals.values, shap_vals.base_values

        proba = self._lime_predict_fn(np.array([x]))[0]
        idx = int(np.argmax(proba)) if output_index is None else output_index
        predicted_class = self.class_names[idx]
        probability = float(proba[idx])

        vals = shap_arr[:, idx].tolist()
        base_val = float(base_arr[0, idx])

        return self._generate_explanation(
            instance=str(x), predicted_class=str(predicted_class),
            probability=probability, feature_names=self.feature_names,
            feature_values=x.tolist(), method="shap",
            shap_values=vals,
            base_values=[base_val] * len(self.feature_names),
        )

    # ────────────────────────────────────────────────────────────────────
    # Evaluación de métricas y selección del mejor explicador
    # ────────────────────────────────────────────────────────────────────

    def evaluate_metrics(
        self, instance, eps: float = 1e-2, sigma: float = 0.1, n_lip: int = 10,
    ) -> Tuple[Dict[str, Dict[str, float]], str]:
        """
        Calcula Infidelity, Local Lipschitz y Effective Complexity para los
        tres explicadores. Devuelve (resultados, mejor_método).
        """
        rng = np.random.default_rng(0)
        x = self._to_code_array(instance)
        baseline = np.mean(self.train_data, axis=0)

        probs = self._lime_predict_fn(x.reshape(1, -1))[0]
        cls = int(np.argmax(probs))
        f = lambda z: self._lime_predict_fn(z.reshape(1, -1))[0][cls]
        f_prob = lambda T: self._lime_predict_fn(np.asarray(T))[:, cls]
        f0 = f(x)

        # ── Obtener atribuciones ─────────────────────────────────────────
        # SHAP
        shap_list = self.shap_explainer.shap_values(x)
        shap_vals = shap_list[:, cls]

        # LIME
        lime_exp = self.lime_explainer.explain_instance(
            x, self._lime_predict_fn,
            num_features=len(self.feature_names), labels=[cls],
        )
        lime_vals = np.zeros(len(self.feature_names))
        for cond, w in lime_exp.as_list(label=cls):
            feat = self._extract_feature_name(cond)
            if feat in self.feature_names:
                lime_vals[self.feature_names.index(feat)] = w

        # ANCHOR
        anc = self.anchor_explainer.explain_instance(
            x, lambda z: self.me.model.predict_proba(z)[:, cls],
            threshold=0.90, delta=0.15, batch_size=256,
        )
        anchor_vals = self._get_active_vector(set(anc.names()))

        attributions = {"shap": shap_vals, "lime": lime_vals, "anchor": anchor_vals}

        # ── Calcular métricas ────────────────────────────────────────────
        results = {}
        for method, a in attributions.items():
            infid = 0 if method == "anchor" else self._infidelity_factorial(
                x, a, baseline, f_prob, M=5, rng=rng
            )
            lip = self._lipschitz(x, a, method, cls, sigma, n_lip, rng)
            ef = self._effective_complexity(x, a, baseline, f, f0, eps)
            results[method] = {
                "infidelity": infid,
                "lipschitz": lip,
                "effective_complexity": ef,
                "class": cls,
            }

        # ── Normalizar y seleccionar mejor ───────────────────────────────
        metrics = ["infidelity", "lipschitz", "effective_complexity"]
        arrs = {m: np.array([results[met][m] for met in results]) for m in metrics}
        norm = {}
        for m, arr in arrs.items():
            norm[m] = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)
        scores = {
            met: float(sum(norm[m][i] for m in metrics))
            for i, met in enumerate(results)
        }
        best = min(scores, key=scores.get)
        return results, best

    def explain_instance(self, instance, method: Optional[str] = None) -> dict:
        """
        Genera explicaciones con LIME y Anchor de forma segura, e intenta generar SHAP.
        Devuelve todos los resultados exitosos para que el frontend pueda mostrar
        las tres pestañas simultáneamente.
        """
        x = self._to_code_array(instance)
        proba = self._lime_predict_fn(np.array([x]))[0]
        idx_pred = np.argmax(proba)
        predicted_class = self.class_names[idx_pred]
        probability = float(proba[idx_pred])

        explanations = {}

        # 1. Calcular LIME (Estable)
        try:
            explanations["lime"] = json.loads(self.lime(instance))
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Error aislando LIME: {e}")

        # 2. Calcular ANCHOR (Estable)
        try:
            explanations["anchor"] = json.loads(self.anchor(instance))
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Error aislando Anchor: {e}")

        # 3. Calcular SHAP (Inestable por ceros absolutos, con protección)
        try:
            raw_shap = self.shap_explain(instance)
            parsed_shap = json.loads(raw_shap)
            
            # Verificación matemática de NaNs
            features = parsed_shap.get("features", [])
            has_nans = any(
                math.isnan(float(f.get("shap_value", 0))) 
                for f in features 
                if isinstance(f.get("shap_value"), (int, float))
            )
            
            if not has_nans:
                explanations["shap"] = parsed_shap
            else:
                import logging
                logging.getLogger(__name__).warning("SHAP generó NaNs. Omitiendo del paquete.")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Error crítico en SHAP: {e}. Omitiendo del paquete.")

        # Definir un método "principal" para la narrativa de texto
        primary_method = "shap" if "shap" in explanations else ("lime" if "lime" in explanations else "anchor")

        return {
            "method_used": primary_method,
            "prediction": str(predicted_class),
            "label": self.label_map.get(predicted_class, str(predicted_class)),            "confidence": round(probability * 100, 2),
            "explanations": explanations, # Enviamos el diccionario con los 3
            "metrics": None, # Desactivamos métricas temporalmente para ahorrar RAM
        }

    # ────────────────────────────────────────────────────────────────────
    # Helpers internos
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_feature_name(condition: str) -> str:
        m = re.match(r"^(.*?)\s*(?:=|<=|>=|<|>)", condition)
        return m.group(1).strip() if m else condition.split()[0]

    def _get_active_vector(self, anchor_names: set) -> np.ndarray:
        splitter = re.compile(r"\s*(?:<=|>=|=|<|>)\s*")
        active = {splitter.split(a, 1)[0] for a in anchor_names}
        return np.array([int(f in active) for f in self.feature_names], dtype=int)

    @staticmethod
    def _infidelity_factorial(x, phi, baseline, f_prob, M=5, rng=None):
        d = x.size
        diff = x - baseline
        phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)

        eps_val = 1e-6
        mask = np.abs(diff) > eps_val
        phi_star = np.zeros_like(phi)
        phi_star[mask] = phi[mask] / diff[mask]

        fact = math.factorial
        w_k = np.zeros(d + 1, dtype=float)
        for k in range(1, d):
            w_k[k] = fact(d - k - 1) * fact(k)
        w_k /= w_k.sum()
        ks = rng.choice(d + 1, size=M, p=w_k)
        Z = np.zeros((M, d), dtype=bool)
        for i, k in enumerate(ks):
            Z[i, rng.choice(d, size=k, replace=False)] = True

        I = x - (Z * x + (~Z) * baseline)
        delta_phi = np.nan_to_num((I * phi_star).sum(axis=1), nan=0.0)
        f_x = f_prob(x.reshape(1, -1))[0]
        f_z = f_prob(Z * x + (~Z) * baseline)
        delta_f = f_x - f_z

        return float(np.mean((delta_phi - delta_f) ** 2))

    def _lipschitz(self, x, a, method, cls, sigma, n_lip, rng):
        ratios = []
        actual_n = 3 if method == "anchor" else n_lip

        for _ in range(actual_n):
            x_p = x + rng.normal(0, sigma, size=x.shape)
            if method == "shap":
                sv = self.shap_explainer.shap_values(x_p)
                a_p = np.asarray(sv[:, cls])
            elif method == "lime":
                exp_p = self.lime_explainer.explain_instance(
                    x_p, self._lime_predict_fn,
                    num_features=len(self.feature_names), labels=[cls],
                )
                a_p = np.zeros(len(self.feature_names))
                for cond, w in exp_p.as_list(label=cls):
                    feat = self._extract_feature_name(cond)
                    if feat in self.feature_names:
                        a_p[self.feature_names.index(feat)] = w
            else:
                anc = self.anchor_explainer.explain_instance(
                    x_p, lambda z: self.me.model.predict_proba(z)[:, cls],
                    threshold=0.95, delta=0.1, batch_size=50,
                )
                a_p = self._get_active_vector(set(anc.names()))

            e = 1e-12
            a_n = a / (np.linalg.norm(a, ord=1) + e)
            a_p_n = a_p / (np.linalg.norm(a_p, ord=1) + e)
            ratios.append(np.linalg.norm(a_n - a_p_n) / (np.linalg.norm(x - x_p) + e))

        return float(max(ratios)) if ratios else 0.0

    @staticmethod
    def _effective_complexity(x, a, baseline, f, f0, eps):
        order = np.argsort(-np.abs(a))
        for k in range(1, len(order) + 1):
            mask = order[:k]
            x_m = x.copy()
            x_m[mask] = baseline[mask]
            if abs(f0 - f(x_m)) < eps:
                return float(k)
        return float(len(order))

    def _generate_explanation(
        self, instance, predicted_class, probability, feature_names, feature_values,
        *, method, shap_values=None, base_values=None,
        lime_weights=None, lime_conditions=None,
        anchor_conditions=None, anchor_precision=None, anchor_coverage=None,
    ) -> str:
        exp: Dict[str, Any] = {
                    "model_output": {
                        "predicted_class": predicted_class,
                        "probability": probability,
                        "label": self.label_map.get(predicted_class, str(predicted_class)),
                    }
                }

        if method in ("shap", "lime"):
            feats = []
            decoded_values = self.decode_instance_as_list(feature_values)
            fnamesvalues = dict(zip(feature_names, decoded_values))

            for i, name in enumerate(feature_names):
                if method == "shap":
                    info = {"name": name, "value": str(decoded_values[i])}
                    info["base_value"] = base_values[i]
                    info["shap_value"] = shap_values[i]
                else:
                    f_name = self._extract_feature_name(lime_conditions[i])
                    f_val = fnamesvalues.get(f_name, decoded_values[i])
                    info = {"name": f_name, "value": str(f_val)}
                    info["condition"] = lime_conditions[i]
                    info["lime_weight"] = float(lime_weights[i])
                feats.append(info)
            exp["features"] = feats
        else:
            exp["anchor"] = {
                "conditions": anchor_conditions,
                "precision": anchor_precision,
                "coverage": anchor_coverage,
            }

        return json.dumps(exp, indent=2, ensure_ascii=False)
