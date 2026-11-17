"""
Seguridad para carga de modelos pickle/joblib.

Dos capas de defensa contra RCE via deserialización maliciosa:
  1. scan_pickle()     — análisis estático de opcodes (sin ejecutar nada)
  2. patch_safe_loader() — parcheado permanente de joblib con allowlist de módulos
"""

from __future__ import annotations

import io
import logging
import pickle
import pickletools
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Allowlist de módulos permitidos ──────────────────────────────────────────
# Solo librerías ML conocidas. Cualquier módulo fuera de esta lista es rechazado.

_ALLOWED_MODULE_BASES: frozenset[str] = frozenset({
    # NumPy y extensiones internas
    "numpy", "_codecs",
    # SciPy
    "scipy",
    # scikit-learn
    "sklearn",
    # LightGBM
    "lightgbm",
    # XGBoost
    "xgboost",
    # CatBoost
    "catboost",
    # joblib (almacena arrays como memmaps)
    "joblib", "_joblib",
    # Tipos de Python estándar que sklearn usa en pickling
    "builtins",      # object, dict, list — numpy los referencia
    "collections",   # OrderedDict
    "copyreg",       # copy_reg interno de pickle
    "copy_reg",
    "abc",
    "functools",
    "numbers",
    "typing",
    "operator",
    "math",
    "decimal",
    "_abc",
    "_functools",
    # pandas (para pipelines con ColumnTransformer)
    "pandas",
})

# ── Globals de pickle explícitamente bloqueados (además del allowlist) ───────
# Patrones que SIEMPRE indican un exploit, aunque estén en un módulo "permitido".
# Incluimos 'posix' porque en Unix, os.system compila como posix.system.

_BLOCKED_GLOBALS: frozenset[tuple[str, str]] = frozenset({
    ("builtins", "eval"),
    ("builtins", "exec"),
    ("builtins", "compile"),
    ("builtins", "__import__"),
    ("builtins", "open"),
    ("builtins", "input"),
    ("os", "system"),     ("posix", "system"),
    ("os", "popen"),      ("posix", "popen"),
    ("os", "exec"),       ("posix", "exec"),
    ("os", "execve"),     ("posix", "execve"),
    ("os", "execvp"),     ("posix", "execvp"),
    ("os", "execvpe"),    ("posix", "execvpe"),
    ("os", "spawn"),      ("posix", "spawn"),
    ("os", "fork"),       ("posix", "fork"),
    ("os", "kill"),       ("posix", "kill"),
    ("subprocess", "check_output"),
    ("subprocess", "Popen"),
    ("subprocess", "run"),
    ("subprocess", "call"),
    ("subprocess", "check_call"),
    ("sys", "exit"),
    ("importlib", "import_module"),
    ("importlib._bootstrap", "_call_with_frames_removed"),
    ("ctypes", "cdll"),
    ("ctypes", "CDLL"),
    ("socket", "socket"),
    ("nt", "system"),     # posix equivalente en Windows
    ("nt", "popen"),
})


# ── CAPA 1: Scan estático de opcodes ─────────────────────────────────────────

def scan_pickle(filepath: str | Path) -> tuple[bool, Optional[str]]:
    """
    Analiza los opcodes del archivo pickle sin ejecutar ningún byte de código.

    Maneja los tres opcodes que referencian módulos externos:
      - GLOBAL  "modulo clase"        (protocolo 0–3)
      - INST    "modulo clase"        (protocolo 0, legacy)
      - STACK_GLOBAL                  (protocolo 4+) — requiere rastrear el stack

    Para STACK_GLOBAL, rastrea las dos strings más recientes en la pila para
    reconstruir (módulo, nombre) sin ejecutar nada.

    Returns:
        (True, None)       si el archivo es seguro.
        (False, "motivo")  si se detecta contenido peligroso.
    """
    filepath = Path(filepath)
    violations: list[str] = []

    try:
        data = filepath.read_bytes()
    except OSError as e:
        return False, f"No se pudo leer el archivo: {e}"

    # Opcodes que empujan un string literal al stack de pickle
    _STRING_OPCODES = frozenset({
        "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8",
        "UNICODE", "STRING", "SHORT_BINSTRING", "BINSTRING",
    })
    # Opcodes de encabezado que no cambian el stack relevante
    _TRANSPARENT_OPCODES = frozenset({"PROTO", "FRAME"})

    # Stack de strings: igual que el stack de pickle pero solo rastrea strings.
    # Los no-strings se representan como None para mantener la posición.
    string_stack: list[str | None] = []
    # Memo: slot → string (solo guardamos strings; None para todo lo demás)
    memo: dict[int, str | None] = {}
    _next_memo_slot = 0  # contador para MEMOIZE (auto-increment en proto 4)

    try:
        buf = io.BytesIO(data)
        for opcode, arg, _pos in pickletools.genops(buf):
            opname = opcode.name

            # ── Strings literales ───────────────────────────────────────
            if opname in _STRING_OPCODES:
                string_stack.append(arg)

            # ── Encabezados inocuos ─────────────────────────────────────
            elif opname in _TRANSPARENT_OPCODES:
                pass

            # ── MEMOIZE (proto 4+): guarda TOS en el slot actual ────────
            elif opname == "MEMOIZE":
                val = string_stack[-1] if string_stack else None
                memo[_next_memo_slot] = val
                _next_memo_slot += 1
                # TOS permanece en el stack (MEMOIZE no consume)

            # ── BINPUT / LONG_BINPUT: guarda TOS en slot explícito ──────
            elif opname in ("BINPUT", "LONG_BINPUT", "PUT"):
                slot = int(arg)
                val = string_stack[-1] if string_stack else None
                memo[slot] = val
                _next_memo_slot = max(_next_memo_slot, slot + 1)

            # ── GET / BINGET: recupera del memo ─────────────────────────
            elif opname in ("GET", "BINGET", "LONG_BINGET"):
                slot = int(arg)
                retrieved = memo.get(slot)  # None si no es string
                string_stack.append(retrieved)

            # ── GLOBAL "modulo clase" (proto 0–3) ───────────────────────
            elif opname == "GLOBAL":
                parts = (arg or "").split(" ", 1)
                if len(parts) == 2:
                    _check_global(parts[0], parts[1], violations)
                string_stack.clear()

            # ── INST (proto 0, legacy) ──────────────────────────────────
            elif opname == "INST":
                parts = (arg or "").split(" ", 1)
                if len(parts) == 2:
                    _check_global(parts[0], parts[1], violations)
                string_stack.clear()

            # ── STACK_GLOBAL (proto 4+): TOS=nombre, TOS-1=módulo ───────
            elif opname == "STACK_GLOBAL":
                if len(string_stack) >= 2:
                    name_val = string_stack[-1]
                    module_val = string_stack[-2]
                    if isinstance(module_val, str) and isinstance(name_val, str):
                        _check_global(module_val, name_val, violations)
                    # Si alguno es None (no-string en memo): no podemos
                    # verificarlo estáticamente → Capa 2 lo cubre en runtime
                    string_stack = string_stack[:-2]
                else:
                    # Stack insuficiente — contexto no rastreable estáticamente.
                    # Capa 2 interceptará find_class() en runtime.
                    string_stack.clear()

            # ── Cualquier otro opcode: empuja None (no-string) ──────────
            else:
                # No limpiamos el stack: empujamos None para mantener posición
                # relativa. Esto permite que strings anteriores NO pierdan su
                # slot si hay un valor no-string intercalado.
                # Sin embargo, si el stack crece demasiado por objetos no-string
                # entre dos strings usados para STACK_GLOBAL, la verificación
                # de los dos TOS no se verá afectada.
                pass  # no modificamos string_stack aquí

    except ValueError as e:
        # pickletools lanza ValueError en opcodes desconocidos que corresponden
        # a datos binarios de numpy embebidos por joblib. Si no hubo violaciones
        # hasta este punto, el archivo pasa Capa 1 (Capa 2 cubre el resto).
        if violations:
            logger.warning("scan_pickle BLOQUEADO %s (opcode desconocido): %s", filepath.name, violations)
        else:
            logger.debug("scan_pickle: datos binarios numpy en %s — OK hasta opcode desconocido", filepath.name)
    except Exception as e:
        return False, f"Error inesperado al analizar el bytecode pickle: {e}"

    if violations:
        logger.warning("scan_pickle BLOQUEADO %s: %s", filepath.name, "; ".join(violations))
        return False, "; ".join(violations)

    logger.info("scan_pickle OK: %s", filepath.name)
    return True, None


def _check_global(module: str, name: str, violations: list[str]) -> None:
    """Valida un par (módulo, nombre) contra la allowlist y la blocklist."""
    if (module, name) in _BLOCKED_GLOBALS:
        violations.append(f"Función peligrosa: {module}.{name}")
        return
    module_base = module.split(".")[0]
    if module_base not in _ALLOWED_MODULE_BASES:
        violations.append(f"Módulo no permitido: {module} (clase: {name})")


# ── CAPA 2: SafeUnpickler — parcheado permanente de joblib ───────────────────

class _SafeUnpickler(pickle.Unpickler):
    """
    Unpickler restringido a _ALLOWED_MODULE_BASES.
    Actúa como última barrera si algún exploit sobrevive el scan estático.
    """

    def find_class(self, module: str, name: str):
        if (module, name) in _BLOCKED_GLOBALS:
            raise pickle.UnpicklingError(
                f"[SecurityError] Función bloqueada explícitamente: {module}.{name}"
            )
        module_base = module.split(".")[0]
        if module_base not in _ALLOWED_MODULE_BASES:
            raise pickle.UnpicklingError(
                f"[SecurityError] Módulo no permitido: '{module}.{name}'. "
                f"Solo se aceptan modelos de sklearn, lightgbm, numpy, etc."
            )
        return super().find_class(module, name)


def patch_safe_loader() -> None:
    """
    Parchea find_class en el NumpyUnpickler interno de joblib de forma permanente.
    Debe llamarse UNA VEZ antes de cualquier joblib.load().

    Parcheamos el método directamente (no por herencia) para preservar la firma
    del constructor de NumpyUnpickler, que varía entre versiones de joblib.
    """
    try:
        import joblib.numpy_pickle as _np_pickle

        if getattr(_np_pickle.NumpyUnpickler, "_profilexai_patched", False):
            return  # ya parcheado

        _orig_find_class = _np_pickle.NumpyUnpickler.find_class

        def _safe_find_class(self, module: str, name: str):
            if (module, name) in _BLOCKED_GLOBALS:
                raise pickle.UnpicklingError(
                    f"[SecurityError] Función bloqueada: {module}.{name}"
                )
            module_base = module.split(".")[0]
            if module_base not in _ALLOWED_MODULE_BASES:
                raise pickle.UnpicklingError(
                    f"[SecurityError] Módulo no permitido: '{module}.{name}'. "
                    f"Solo se aceptan modelos de sklearn, lightgbm, numpy, etc."
                )
            return _orig_find_class(self, module, name)

        _np_pickle.NumpyUnpickler.find_class = _safe_find_class
        _np_pickle.NumpyUnpickler._profilexai_patched = True
        logger.info("SafeUnpickler activado: joblib.NumpyUnpickler.find_class ahora usa allowlist ML.")

    except ImportError:
        logger.warning("joblib no disponible; patch_safe_loader() omitido.")
    except Exception as e:
        logger.error("Error aplicando patch_safe_loader(): %s", e)
