# U03.E06 · Autoencoder y VAE sobre Fashion-MNIST (LAB10)

**Estudiante:** Johanna Joaquín · INF-8239 Ciencia de Datos II · UASD
**Repositorio:** https://github.com/licjjoaquin-source/INF8239-U03-generativos

Comparación reproducible de un autoencoder (AE) y un autoencoder variacional (VAE) sobre Fashion-MNIST: reconstrucción, muestreo, interpolación, fidelidad, diversidad, memoria y costo. El proyecto parte del código base del LAB10 (`train_ae_vae.py`) y le añade una evaluación con varias semillas.

## Datos

Fashion-MNIST (Xiao, Rasul y Vollgraf, 2017), de Zalando Research, licencia MIT: 60,000 imágenes de entrenamiento y 10,000 de prueba de 28×28 píxeles en escala de grises y 10 clases de prendas. Se descarga con `tf.keras.datasets.fashion_mnist.load_data()` y **no** se incluye en el repositorio. No contiene datos personales.

## Contenido

| Ruta | Para qué sirve |
|---|---|
| `src/inf8239_u03_gen/` | `data.py` y `models.py` (base del laboratorio), `vae.py` (clase VAE y separación del AE en encoder/decoder) y `gen_metrics.py` (métricas en NumPy) |
| `scripts/train_ae_vae.py` | Entrenamiento base del laboratorio (un AE de 16 y un VAE de 2 dimensiones) |
| `scripts/e06_evaluacion.py` | Evaluación completa: 4 modelos × 3 semillas, tablas y figuras |
| `tests/` | 16 pruebas (contrato de datos y modelos, métricas, VAE) |
| `reports/` | Tabla, CSV por semilla, resumen JSON y figuras `e06_*.png` |
| `notebooks/E06_resultados.ipynb` | Lectura e interpretación de los resultados |
| `MODEL_CARD.md` | Tarjeta del modelo: uso previsto, límites y decisión |
| `AI_USE_DECLARATION.md` | Declaración de uso de IA |

## Reproducir

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.12.

```powershell
uv sync --extra cpu
uv run pytest -q
uv run python scripts/e06_evaluacion.py --seeds 42 43 44    # unos 5 a 8 minutos en CPU
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/E06_resultados.ipynb
```

Prueba rápida (menos de 1 minuto): `uv run python scripts/e06_evaluacion.py --seeds 42 --limit 6000 --epochs-ae 2 --epochs-vae 2`.
Entorno de los resultados: Python 3.12.13, TensorFlow 2.21.0, Windows-11-10.0.26200-SP0, solo CPU. Semillas 42, 43 y 44; 6 épocas para el AE y 8 para el VAE (los valores por defecto del laboratorio). Con `TF_DETERMINISTIC_OPS=1`, otra máquina u otra versión de TensorFlow puede dar cifras ligeramente distintas.

## Modelos comparados

| Modelo | Qué es | Cómo se generan muestras |
|---|---|---|
| AE-16 | Baseline del laboratorio: Flatten → Dense(16, ReLU) → Dense(784) | `N(0, I)` (ingenuo) y gaussiana ajustada a los latentes de entrenamiento |
| AE-2 | Control con el mismo tamaño de latente que el VAE del laboratorio | Gaussiana ajustada |
| VAE-2 | VAE del laboratorio: Dense(128) → (μ, log σ²) de 2 dimensiones | `N(0, I)` |
| VAE-16 | Como el anterior, con latente de 16 dimensiones | `N(0, I)` |

Un AE no tiene distribución a priori, por eso se prueba con `N(0, I)` y con una gaussiana ajustada después del entrenamiento. **Aviso de comparación:** el AE del laboratorio no tiene capa oculta (26 mil parámetros) y el VAE sí (cerca de 200 mil), así que las diferencias de reconstrucción entre AE y VAE mezclan el objetivo del VAE con la capacidad del modelo.

## Particiones

Se usa la partición oficial: entrenamiento (60,000) y prueba (10,000). El AE reserva el 10 % final del entrenamiento como validación para parar antes (paciencia 2). Los 10,000 de prueba solo se usan para evaluar: nada se ajustó con ellos. Referencias de comparación: reconstrucción con los 10,000 de prueba; Fréchet contra las primeras 2,000 de prueba; diversidad y vecinos contra las primeras 1,000 de prueba. Un clasificador auxiliar (exactitud en prueba 0.853 ± 0.004) se entrena con el entrenamiento y solo sirve para juzgar si las muestras se reconocen y cubren clases.

Los 16 ejemplos de prueba de los paneles **no se escogen a mano**: son índices elegidos con la semilla 2026 (`263, 797, 1773, 1786, 3547, 3650, 3702, 4668, 6391, 6430, 7042, 7211, 7902, 8306, 8505, 9049`; clases: Camiseta, Bolso, Zapatilla, Vestido, Sandalia, Pantalón, Vestido, Abrigo, Zapatilla, Camiseta, Camisa, Abrigo, Bota, Camisa, Zapatilla, Sandalia).

## Resultados (media ± desviación entre 3 semillas)

| Modelo | MSE recon. (test) | Fréchet píxeles | Diversidad (vs real) | Clases cubiertas (de 10) | Confianza clasif. | Vecino/real | Casi copias | Entren. (s) | Parámetros | Tamaño (KB) |
|---|---|---|---|---|---|---|---|---|---|---|
| AE-16 con N(0,I) | 0.0206 ± 0.0004 | 95.0 ± 0.1 | 0.46 ± 0.05 | 1.3 ± 0.6 | 0.97 ± 0.01 | 1.79 ± 0.01 | 0.000 ± 0.000 | 7.3 ± 1.3 | 25888 ± 0 | 331 ± 0 |
| AE-16 gaussiana ajustada | 0.0206 ± 0.0004 | 15.3 ± 1.1 | 0.88 ± 0.01 | 10.0 ± 0.0 | 0.68 ± 0.02 | 1.33 ± 0.01 | 0.000 ± 0.001 | 7.3 ± 1.3 | 25888 ± 0 | 331 ± 0 |
| AE-2 gaussiana ajustada | 0.0585 ± 0.0001 | 44.8 ± 0.7 | 0.65 ± 0.01 | 8.3 ± 0.6 | 0.54 ± 0.02 | 1.14 ± 0.01 | 0.000 ± 0.000 | 7.1 ± 0.2 | 3922 ± 0 | 73 ± 0 |
| VAE-2 | 0.0330 ± 0.0001 | 36.8 ± 4.4 | 0.55 ± 0.04 | 9.3 ± 0.6 | 0.55 ± 0.03 | 0.77 ± 0.04 | 0.142 ± 0.032 | 15.4 ± 0.6 | 202516 ± 0 | 833 ± 0 |
| VAE-16 | 0.0165 ± 0.0001 | 17.5 ± 0.7 | 0.68 ± 0.01 | 10.0 ± 0.0 | 0.66 ± 0.01 | 1.09 ± 0.02 | 0.002 ± 0.002 | 14.8 ± 0.4 | 207920 ± 0 | 854 ± 0 |

Significado: *Fréchet píxeles* es una distancia de Fréchet calculada sobre píxeles (no es FID con Inception); menor es mejor. Entre dos mitades de imágenes reales vale 3.8, que es el piso de ruido. *Diversidad* es la distancia media entre pares de muestras dividida entre la de imágenes reales (1 = tan variadas como las reales). *Vecino/real* es la distancia mediana de las muestras a su vecino más cercano del entrenamiento dividida entre la de imágenes reales de prueba. *Casi copias* es la fracción de muestras más cerca del entrenamiento que el 5 % más cercano de las imágenes reales (en un modelo perfecto valdría cerca de 0.05).

### Figuras

- `reports/e06_panel_reconstruccion.png`: reconstrucciones de los 16 ejemplos fijos.
- `reports/e06_muestras.png`: 16 muestras nuevas por modelo (las primeras 16, sin escoger).
- `reports/e06_interpolacion.png`: interpolación entre dos pares de imágenes reales.
- `reports/e06_memoria_vae2.png` y `e06_memoria_vae16.png`: muestras y su vecino más cercano en el entrenamiento.
- `reports/e06_latente_vae2.png`: mapa del espacio latente de 2 dimensiones.

## Lectura de los resultados

1. **Reconstrucción.** VAE-16 reconstruye mejor (MSE 0.0165 ± 0.0001) que AE-16 (0.0206 ± 0.0004), pero con 8 veces más parámetros. Con 2 dimensiones el error es mucho mayor (VAE-2 0.0330 ± 0.0001; AE-2 0.0585 ± 0.0001): el vestido se reconstruye como pantalón y las prendas de torso se confunden. En ningún modelo hay brecha entre entrenamiento y prueba (del orden de 1e-4): no hay sobreajuste de reconstrucción.
2. **Muestreo.** Un AE muestreado con `N(0, I)` colapsa: casi todas las muestras caen en una sola clase (1.3 de 10) aunque el clasificador esté muy seguro (0.97), así que la confianza no mide calidad. Con una gaussiana ajustada el AE-16 llega a Fréchet 15.3 ± 1.1, parecido al de VAE-16 (17.5 ± 0.7); fue ligeramente mejor en las 3 semillas, pero con solo 3 semillas no se concluye. VAE-16 tiene la ventaja de muestrear directamente de `N(0, I)`, sin paso posterior. Ninguna muestra se acerca al piso de ruido (3.8) y todas son borrosas.
3. **Diversidad.** Todos quedan por debajo de las imágenes reales (0.46 a 0.88): las muestras son más parecidas entre sí. VAE-2 cubre 9.3 clases pero con muchas muestras tenues y poco nítidas.
4. **Memoria.** No se encontró evidencia de memorización: ninguna copia exacta, brecha entrenamiento-prueba cercana a cero y vecinos visualmente distintos de las muestras. Solo VAE-2 muestra la señal métrica de Vecino/real menor que 1 (0.77 ± 0.04) y 14 % de casi copias, pero las figuras `e06_memoria_vae2.png` muestran que las muestras más cercanas son manchas casi negras cuyo vecino es una prenda oscura: la distancia en píxeles es pequeña porque ambas imágenes son oscuras, no porque se haya copiado. Esa métrica es sensible al brillo y por sí sola no prueba memorización.
5. **Costo.** El AE tarda unos 7 s y el VAE unos 15 s en CPU; los parámetros pasan de 26 mil (AE-16) a 208 mil (VAE-16) y el archivo de 331 KB a 854 KB. Los tiempos de inferencia (unos 200 ms por 1,000 imágenes) no difieren más que su variación.

## Decisión

Para generar imágenes nuevas al estilo Fashion-MNIST con un muestreo simple, **se elige VAE-16**: es el mejor reconstructor, cubre las 10 clases y se muestrea sin pasos adicionales. **AE-16 con gaussiana ajustada** es una alternativa más barata (8 veces menos parámetros, la mitad del tiempo) con calidad de muestras comparable. **VAE-2** solo sirve para visualizar el espacio latente (la app de Streamlit del laboratorio), no como generador. Uso solo académico y exploratorio; ver límites en `MODEL_CARD.md`.

## Límites

- Imágenes de 28×28 en gris, un dominio simple: no se puede extrapolar a imágenes reales de mayor resolución, rostros o imágenes médicas.
- Entrenamiento corto (6 y 8 épocas) y arquitecturas pequeñas; las muestras salen borrosas. No se ajustaron hiperparámetros.
- Solo 3 semillas: las diferencias pequeñas (por ejemplo AE-16 ajustado contra VAE-16) no son concluyentes.
- El Fréchet en píxeles y el clasificador auxiliar (exactitud 0.853 ± 0.004) son aproximaciones; no son FID ni juicio humano.
- Memoria medida con distancias en píxeles: no descarta fugas sutiles ni da garantías de privacidad.
- La comparación AE contra VAE está confundida por la capacidad del modelo (ver arriba).

## Uso de IA y licencias

Ver `AI_USE_DECLARATION.md`. Código bajo licencia del curso; Fashion-MNIST bajo licencia MIT (Zalando Research).

## Referencia

Xiao, H., Rasul, K., y Vollgraf, R. (2017). *Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms*. arXiv:1708.07747.
