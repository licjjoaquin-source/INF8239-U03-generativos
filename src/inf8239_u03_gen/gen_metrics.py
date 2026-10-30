"""Métricas para evaluar modelos generativos (solo NumPy, sin TensorFlow).

Todas las funciones trabajan con imágenes aplanadas o con forma (n, 28, 28, 1)
y valores en [0, 1]. Se separan del script de entrenamiento para poder probarlas
sin instalar TensorFlow.
"""
from __future__ import annotations

import numpy as np


def flatten(images: np.ndarray) -> np.ndarray:
    """(n, 28, 28, 1) -> (n, 784) en float32."""
    images = np.asarray(images, dtype="float32")
    return images.reshape(images.shape[0], -1)


def fixed_indices(n_items: int, count: int = 16, seed: int = 2026) -> np.ndarray:
    """Índices fijos y reproducibles, elegidos al azar con semilla.

    Se usan para el panel de ejemplos "no seleccionados": nadie escoge a mano los
    casos bonitos; la semilla y los índices quedan registrados en el informe.
    """
    if count > n_items:
        raise ValueError("count no puede superar n_items")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_items, size=count, replace=False))


def reconstruction_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Error cuadrático medio por píxel, promediado sobre todas las imágenes."""
    a, b = flatten(original), flatten(reconstructed)
    if a.shape != b.shape:
        raise ValueError(f"Formas distintas: {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))


def nearest_neighbor_distances(queries: np.ndarray, reference: np.ndarray, chunk: int = 500):
    """Distancia euclídea al vecino más cercano y su índice (fuerza bruta por bloques)."""
    q, r = flatten(queries), flatten(reference)
    r_sq = np.sum(r * r, axis=1)
    distances = np.empty(len(q), dtype="float32")
    indices = np.empty(len(q), dtype="int64")
    for start in range(0, len(q), chunk):
        block = q[start:start + chunk]
        sq = np.sum(block * block, axis=1)[:, None] + r_sq[None, :] - 2.0 * block @ r.T
        best = np.argmin(sq, axis=1)
        indices[start:start + chunk] = best
        # La fórmula expandida pierde precisión (float32); se recalcula exacta solo para el ganador.
        distances[start:start + chunk] = np.linalg.norm(block - r[best], axis=1)
    return distances, indices


def memorization_report(generated, train, test, quantile: float = 0.05, chunk: int = 500) -> dict:
    """Compara qué tan cerca están las muestras generadas de los datos de entrenamiento.

    Referencia: la distancia al vecino más cercano de imágenes REALES de prueba (que el
    modelo no vio). Si las muestras generadas están claramente más cerca del
    entrenamiento que lo que están las imágenes reales nuevas, hay señal de memoria.
    El umbral de "casi copia" es el cuantil bajo de esa referencia.
    """
    gen_dist, gen_idx = nearest_neighbor_distances(generated, train, chunk)
    test_dist, _ = nearest_neighbor_distances(test, train, chunk)
    threshold = float(np.quantile(test_dist, quantile))
    return {
        "nn_generated_median": float(np.median(gen_dist)),
        "nn_test_median": float(np.median(test_dist)),
        "nn_ratio": float(np.median(gen_dist) / max(np.median(test_dist), 1e-12)),
        "near_copy_threshold": threshold,
        "near_copy_share": float(np.mean(gen_dist <= threshold)),
        "exact_copy_share": float(np.mean(gen_dist < 1e-6)),
        "generated_distances": gen_dist,
        "generated_neighbor_indices": gen_idx,
    }


def mean_pairwise_distance(images: np.ndarray, n_pairs: int = 20000, seed: int = 0) -> float:
    """Distancia euclídea media entre pares aleatorios distintos (medida de diversidad)."""
    x = flatten(images)
    if len(x) < 2:
        raise ValueError("Se necesitan al menos 2 imágenes")
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(x), size=n_pairs)
    j = rng.integers(0, len(x) - 1, size=n_pairs)
    j = j + (j >= i)  # evita comparar una imagen consigo misma
    return float(np.mean(np.linalg.norm(x[i] - x[j], axis=1)))


def frechet_distance_pixels(real: np.ndarray, generated: np.ndarray) -> float:
    """Distancia de Fréchet entre dos nubes de imágenes, calculada en espacio de píxeles.

    Es una versión simplificada de FID: no usa una red Inception, solo media y
    covarianza de los 784 píxeles. Sirve para comparar modelos entre sí, no para
    comparar con cifras publicadas de FID.
    """
    a, b = flatten(real).astype("float64"), flatten(generated).astype("float64")
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a = np.cov(a, rowvar=False)
    cov_b = np.cov(b, rowvar=False)
    eigenvalues = np.linalg.eigvals(cov_a @ cov_b)
    trace_sqrt = float(np.sum(np.sqrt(np.clip(eigenvalues.real, 0.0, None))))
    value = float(np.sum((mu_a - mu_b) ** 2) + np.trace(cov_a) + np.trace(cov_b) - 2.0 * trace_sqrt)
    return max(value, 0.0)


def class_coverage(predicted: np.ndarray, n_classes: int = 10, min_share: float = 0.01) -> dict:
    """Cuántas clases aparecen entre las muestras y qué tan parejo es el reparto."""
    counts = np.bincount(np.asarray(predicted, dtype=int), minlength=n_classes).astype("float64")
    shares = counts / counts.sum()
    nonzero = shares[shares > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum() / np.log(n_classes))
    return {
        "classes_covered": int(np.sum(shares >= min_share)),
        "normalized_entropy": entropy,
        "shares": shares.tolist(),
    }


def interpolation_path(start: np.ndarray, end: np.ndarray, steps: int = 10) -> np.ndarray:
    """Línea recta entre dos vectores latentes (incluye ambos extremos)."""
    if steps < 2:
        raise ValueError("steps debe ser al menos 2")
    weights = np.linspace(0.0, 1.0, steps)[:, None]
    return (1.0 - weights) * np.asarray(start)[None, :] + weights * np.asarray(end)[None, :]


def summarize_runs(rows: list[dict], group_key: str = "model") -> dict:
    """Media y desviación estándar muestral por modelo, sobre semillas."""
    result: dict[str, dict[str, tuple[float, float]]] = {}
    for name in sorted({row[group_key] for row in rows}):
        subset = [row for row in rows if row[group_key] == name]
        metrics = {}
        for key in subset[0]:
            if key == group_key or not isinstance(subset[0][key], (int, float)):
                continue
            values = np.array([row[key] for row in subset], dtype="float64")
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            metrics[key] = (float(values.mean()), std)
        result[name] = metrics
    return result
