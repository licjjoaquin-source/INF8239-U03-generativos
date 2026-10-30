"""E06 / LAB10 - Evaluación de AE y VAE sobre Fashion-MNIST.

Entrena, por cada semilla, cuatro modelos con la misma receta del proyecto base:
  AE-16  (baseline del laboratorio, latente 16)
  AE-2   (control justo: mismo tamaño de latente que el VAE del laboratorio)
  VAE-2  (VAE del laboratorio)
  VAE-16 (VAE con latente 16, para ver el efecto del tamaño del latente)
y mide reconstrucción, calidad y diversidad de muestras, memoria y costo.

Uso (desde la carpeta del proyecto):
  uv run python scripts/e06_evaluacion.py --seeds 42 43 44
  uv run python scripts/e06_evaluacion.py --seeds 42 --limit 6000 --epochs-ae 2 --epochs-vae 2   (prueba rápida)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import tempfile
from pathlib import Path
from time import perf_counter

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from inf8239_u03_gen.data import normalize_images, validate_images
from inf8239_u03_gen.gen_metrics import (
    class_coverage,
    fixed_indices,
    frechet_distance_pixels,
    interpolation_path,
    mean_pairwise_distance,
    memorization_report,
    nearest_neighbor_distances,
    reconstruction_mse,
    summarize_runs,
)
from inf8239_u03_gen.models import build_autoencoder, build_encoder_decoder
from inf8239_u03_gen.vae import make_vae_class, split_autoencoder

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["Camiseta", "Pantalón", "Suéter", "Vestido", "Abrigo",
               "Sandalia", "Camisa", "Zapatilla", "Bolso", "Bota"]
FIXED_SEED = 2026  # semilla de los 16 ejemplos de prueba "no seleccionados"
VAE_CLASS = make_vae_class(tf)


# ----------------------------------------------------------------------------- utilidades
def set_seed(seed: int) -> None:
    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def say(*parts) -> None:
    print(*parts, flush=True)


def size_kb(*models) -> float:
    total = 0
    with tempfile.TemporaryDirectory() as folder:
        for number, model in enumerate(models):
            path = Path(folder) / f"m{number}.keras"
            model.save(path)
            total += path.stat().st_size
    return total / 1024.0


def show_rows(path: Path, rows, cell: float = 0.85, title: str | None = None) -> None:
    """rows = [(etiqueta, imágenes (n,28,28,1)), ...]; todas con el mismo n."""
    n_rows, n_cols = len(rows), len(rows[0][1])
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * cell + 1.6, n_rows * cell + 0.4), squeeze=False)
    for r, (label, images) in enumerate(rows):
        for c in range(n_cols):
            axes[r, c].imshow(np.asarray(images[c]).squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].text(-0.15, 0.5, label, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8)
    if title:
        figure.suptitle(title, fontsize=10)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


# ----------------------------------------------------------------------------- entrenamiento
def train_ae(latent_dim, x_train, epochs, seed):
    set_seed(seed)
    model = build_autoencoder(tf, latent_dim=latent_dim)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    start = perf_counter()
    model.fit(x_train, x_train, validation_split=0.1, epochs=epochs, batch_size=256, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
    seconds = perf_counter() - start
    encoder, decoder = split_autoencoder(tf, model)
    return {"kind": "AE", "latent": latent_dim, "seconds": seconds, "params": model.count_params(),
            "size_kb": size_kb(model), "encode": lambda x: encoder.predict(x, verbose=0),
            "decode": lambda z: decoder.predict(z, verbose=0)}


def train_vae(latent_dim, x_train, epochs, seed):
    set_seed(seed)
    encoder, decoder = build_encoder_decoder(tf, latent_dim=latent_dim)
    vae = VAE_CLASS(encoder, decoder)
    vae.compile(optimizer="adam")
    start = perf_counter()
    history = vae.fit(x_train, epochs=epochs, batch_size=256, verbose=0)
    seconds = perf_counter() - start
    return {"kind": "VAE", "latent": latent_dim, "seconds": seconds,
            "params": encoder.count_params() + decoder.count_params(),
            "size_kb": size_kb(encoder, decoder),
            "kl_last": float(history.history["kl"][-1]),
            "encode": lambda x: encoder.predict(x, verbose=0)[0],  # media posterior
            "decode": lambda z: decoder.predict(z, verbose=0)}


def train_classifier(x_train, y_train, seed):
    """Clasificador auxiliar: SOLO sirve para juzgar si las muestras se reconocen y cubren clases."""
    set_seed(seed)
    model = tf.keras.Sequential([tf.keras.Input((28, 28, 1)), tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(128, activation="relu"),
                                 tf.keras.layers.Dense(10, activation="softmax")])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=3, batch_size=256, verbose=0)
    return model


# ----------------------------------------------------------------------------- evaluación
def evaluate(label, model, sampler, x_train, x_test, x_ref, classifier, n_gen, seed, fixed_idx):
    row = {"model": label, "seed": seed}
    # Reconstrucción
    test_rec = model["decode"](model["encode"](x_test))
    train_part = x_train[:10000]
    train_rec = model["decode"](model["encode"](train_part))
    row["recon_mse_test"] = reconstruction_mse(x_test, test_rec)
    row["recon_mse_train"] = reconstruction_mse(train_part, train_rec)
    row["recon_gap"] = row["recon_mse_test"] - row["recon_mse_train"]
    row["recon_mse_16"] = reconstruction_mse(x_test[fixed_idx], test_rec[fixed_idx])
    # Muestreo
    rng = np.random.default_rng(seed + 1000)
    z = sampler(n_gen, rng)
    generated = model["decode"](z)
    row["diversity_ratio"] = mean_pairwise_distance(generated, seed=seed) / mean_pairwise_distance(x_ref, seed=seed)
    row["frechet_pixels"] = frechet_distance_pixels(x_test[:2000], generated)
    probabilities = classifier.predict(generated, verbose=0)
    coverage = class_coverage(probabilities.argmax(axis=1))
    row["classes_covered"] = coverage["classes_covered"]
    row["class_entropy"] = coverage["normalized_entropy"]
    row["classifier_confidence"] = float(probabilities.max(axis=1).mean())
    memory = memorization_report(generated, x_train, x_ref)
    row["nn_ratio"] = memory["nn_ratio"]
    row["near_copy_share"] = memory["near_copy_share"]
    row["exact_copy_share"] = memory["exact_copy_share"]
    # Costo
    start = perf_counter()
    model["decode"](model["encode"](x_test[:1000]))
    row["infer_ms_per_1000"] = (perf_counter() - start) * 1000.0
    row["train_seconds"] = model["seconds"]
    row["params"] = model["params"]
    row["size_kb"] = model["size_kb"]
    row["kl_last"] = model.get("kl_last", float("nan")) if model["kind"] == "VAE" else float("nan")
    return row, generated, memory, test_rec


# ----------------------------------------------------------------------------- tabla
COLUMNS = [  # (clave, encabezado, decimales)
    ("recon_mse_test", "MSE recon. (test)", 4),
    ("recon_gap", "Brecha test-train", 5),
    ("frechet_pixels", "Fréchet píxeles", 2),
    ("diversity_ratio", "Diversidad (vs real)", 3),
    ("classes_covered", "Clases cubiertas", 1),
    ("classifier_confidence", "Confianza clasif.", 3),
    ("nn_ratio", "Vecino/real", 3),
    ("near_copy_share", "Casi copias", 3),
    ("train_seconds", "Entren. (s)", 1),
    ("params", "Parámetros", 0),
    ("size_kb", "Tamaño (KB)", 0),
    ("infer_ms_per_1000", "Infer. ms/1000", 0),
]


def write_tables(rows, reports: Path) -> dict:
    import csv

    with open(reports / "e06_resultados_por_semilla.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_runs(rows)
    lines = ["| Modelo | " + " | ".join(header for _, header, _ in COLUMNS) + " |",
             "|---|" + "---|" * len(COLUMNS)]
    csv_lines = ["modelo," + ",".join(key + "_media," + key + "_desv" for key, _, _ in COLUMNS)]
    for name, metrics in summary.items():
        cells, csv_cells = [], []
        for key, _, decimals in COLUMNS:
            mean, std = metrics[key]
            cells.append(f"{mean:.{decimals}f} ± {std:.{decimals}f}" if len(rows) > len(summary) else f"{mean:.{decimals}f}")
            csv_cells += [f"{mean:.6f}", f"{std:.6f}"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
        csv_lines.append(name + "," + ",".join(csv_cells))
    (reports / "e06_tabla_principal.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (reports / "e06_tabla_principal.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return summary


# ----------------------------------------------------------------------------- figuras (primera semilla)
def make_figures(reports, models, variants, x_train, x_test, y_test, fixed_idx, generated, memories, recs):
    originals = x_test[fixed_idx]
    show_rows(reports / "e06_panel_reconstruccion.png",
              [("Original", originals)] + [(name, recs[name][fixed_idx]) for name in ("AE-16", "AE-2", "VAE-2", "VAE-16")],
              title=f"Reconstrucciones de 16 imágenes de prueba fijas (índices elegidos con semilla {FIXED_SEED})")
    show_rows(reports / "e06_muestras.png",
              [(label, generated[label][:16]) for label in variants],
              title="16 muestras nuevas por modelo (las primeras 16, sin escoger)")

    pairs = [(fixed_idx[0], fixed_idx[1]), (fixed_idx[2], fixed_idx[3])]
    rows = []
    for a, b in pairs:
        for name in ("AE-16", "VAE-2", "VAE-16"):
            model = models[name]
            path = interpolation_path(model["encode"](x_test[a:a + 1])[0], model["encode"](x_test[b:b + 1])[0], steps=10)
            rows.append((f"{name}\n{CLASS_NAMES[y_test[a]]}→{CLASS_NAMES[y_test[b]]}", model["decode"](path)))
    show_rows(reports / "e06_interpolacion.png", rows, title="Interpolación lineal en el espacio latente entre dos imágenes reales de prueba")

    for name in ("VAE-2", "VAE-16"):
        memory = memories[name]
        distances, neighbors = memory["generated_distances"], memory["generated_neighbor_indices"]
        closest = np.argsort(distances)[:8]
        first = np.arange(8)
        gen = generated[name]
        show_rows(reports / f"e06_memoria_{name.lower().replace('-', '')}.png",
                  [("Muestra (primeras 8)", gen[first]), ("Vecino en train", x_train[neighbors[first]]),
                   ("Muestra (8 más cercanas)", gen[closest]), ("Vecino en train", x_train[neighbors[closest]])],
                  title=f"{name}: muestras y su vecino más cercano en el entrenamiento")

    z = models["VAE-2"]["encode"](x_test[:5000])
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    scatter = axis.scatter(z[:, 0], z[:, 1], c=y_test[:5000], cmap="tab10", s=4)
    axis.set_xlabel("z1")
    axis.set_ylabel("z2")
    axis.set_title("VAE-2: media latente de 5000 imágenes de prueba")
    axis.legend(handles=scatter.legend_elements()[0], labels=CLASS_NAMES, fontsize=6, loc="center left", bbox_to_anchor=(1, 0.5))
    figure.savefig(reports / "e06_latente_vae2.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


# ----------------------------------------------------------------------------- principal
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs-ae", type=int, default=6)
    parser.add_argument("--epochs-vae", type=int, default=8)
    parser.add_argument("--n-gen", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0, help="usar solo N imágenes de entrenamiento (prueba rápida)")
    args = parser.parse_args()
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = normalize_images(x_train), normalize_images(x_test)
    validate_images(x_train)
    validate_images(x_test)
    if args.limit:
        x_train, y_train = x_train[: args.limit], y_train[: args.limit]
        x_test, y_test = x_test[: max(2000, args.limit // 5)], y_test[: max(2000, args.limit // 5)]
    fixed_idx = fixed_indices(len(x_test), 16, FIXED_SEED)
    x_ref = x_test[:1000]
    all_rows, first_seed_artifacts = [], None

    for position, seed in enumerate(args.seeds):
        say(f"=== Semilla {seed} ({position + 1}/{len(args.seeds)})")
        classifier = train_classifier(x_train, y_train, seed)
        accuracy = float((classifier.predict(x_test, verbose=0).argmax(axis=1) == y_test).mean())
        say(f"    clasificador auxiliar: exactitud en test = {accuracy:.3f}")
        models = {"AE-16": train_ae(16, x_train, args.epochs_ae, seed),
                  "AE-2": train_ae(2, x_train, args.epochs_ae, seed),
                  "VAE-2": train_vae(2, x_train, args.epochs_vae, seed),
                  "VAE-16": train_vae(16, x_train, args.epochs_vae, seed)}
        # El AE no tiene distribución a priori: se prueba con N(0,I) (lo "ingenuo")
        # y con una gaussiana ajustada a los latentes reales de entrenamiento.
        fitted = {}
        for name in ("AE-16", "AE-2"):
            latents = models[name]["encode"](x_train[:20000])
            fitted[name] = (latents.mean(axis=0), np.cov(latents, rowvar=False).reshape(latents.shape[1], -1))

        def standard(dim):
            return lambda n, rng: rng.normal(size=(n, dim)).astype("float32")

        def gaussian(name):
            mean, cov = fitted[name]
            return lambda n, rng: rng.multivariate_normal(mean, cov, size=n, check_valid="ignore").astype("float32")

        variants = {"AE-16 con N(0,I)": ("AE-16", standard(16)),
                    "AE-16 gaussiana ajustada": ("AE-16", gaussian("AE-16")),
                    "AE-2 gaussiana ajustada": ("AE-2", gaussian("AE-2")),
                    "VAE-2": ("VAE-2", standard(2)),
                    "VAE-16": ("VAE-16", standard(16))}
        generated, memories, recs = {}, {}, {}
        for label, (model_name, sampler) in variants.items():
            row, gen, memory, rec = evaluate(label, models[model_name], sampler, x_train, x_test, x_ref,
                                             classifier, args.n_gen, seed, fixed_idx)
            row["classifier_test_accuracy"] = accuracy
            all_rows.append(row)
            generated[label], memories[label], recs[model_name] = gen, memory, rec
            say(f"    {label:28s} MSE={row['recon_mse_test']:.4f} Fréchet={row['frechet_pixels']:.1f} "
                f"div={row['diversity_ratio']:.2f} clases={row['classes_covered']} casi-copias={row['near_copy_share']:.3f}")
        if position == 0:
            make_figures(reports, models, variants, x_train, x_test, y_test, fixed_idx, generated, memories, recs)
            first_seed_artifacts = {"figures_seed": seed}

    summary = write_tables(all_rows, reports)
    reference_floor = frechet_distance_pixels(x_test[:1000], x_test[1000:2000])
    info = {
        "seeds": args.seeds, "epochs_ae": args.epochs_ae, "epochs_vae": args.epochs_vae, "n_generated": args.n_gen,
        "train_images": int(len(x_train)), "test_images": int(len(x_test)),
        "fixed_indices_seed": FIXED_SEED, "fixed_indices": fixed_idx.tolist(),
        "fixed_classes": [CLASS_NAMES[i] for i in y_test[fixed_idx]],
        "frechet_noise_floor_real_vs_real": reference_floor,
        "figures_from_seed": first_seed_artifacts["figures_seed"],
        "python": platform.python_version(), "tensorflow": tf.__version__, "platform": platform.platform(),
        "summary_mean_std": {name: {key: list(value) for key, value in metrics.items()} for name, metrics in summary.items()},
    }
    (reports / "e06_resumen.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    say("\nListo. Archivos en reports/: e06_tabla_principal.md/.csv, e06_resultados_por_semilla.csv, "
        "e06_resumen.json y las figuras e06_*.png")
    say((reports / "e06_tabla_principal.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
