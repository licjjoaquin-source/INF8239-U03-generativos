"""Comprueba que los modelos conservados en models/ coinciden con su manifiesto y se pueden cargar.

Uso:  uv run python scripts/verificar_modelos.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def main() -> int:
    manifest = json.loads((MODELS / "MANIFEST.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    fallos = 0
    for entry in manifest["archivos"]:
        path = MODELS / entry["archivo"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            print(f"FALLO hash: {entry['archivo']}")
            fallos += 1
            continue
        model = tf.keras.models.load_model(path, compile=False)
        sample = rng.random((4,) + tuple(model.input_shape[1:])).astype("float32")
        outputs = [np.asarray(o) for o in tf.nest.flatten(model.predict(sample, verbose=0))]
        if not all(np.isfinite(o).all() for o in outputs):
            print(f"FALLO salida no finita: {entry['archivo']}")
            fallos += 1
            continue
        print(f"OK  {entry['archivo']:50s} parámetros={model.count_params()}")
    print(f"\n{len(manifest['archivos']) - fallos} de {len(manifest['archivos'])} modelos verificados")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
