# Model Card · AE y VAE sobre Fashion-MNIST (U03.E06)

## Resumen

Modelos pequeños de reconstrucción y generación de imágenes de prendas de 28×28 píxeles, entrenados con fines académicos en el LAB10 de INF-8239. Se comparan un autoencoder (AE) y un autoencoder variacional (VAE) con latentes de 2 y 16 dimensiones. **Resultado exploratorio en este dominio (Fashion-MNIST): VAE-16 fue el mejor generador de los comparados; VAE-2 solo sirve para visualizar.**

## Detalles del modelo

- **Autor:** Johanna Joaquín (INF-8239 Ciencia de Datos II, UASD). Versión 1.0, 30/10/2026.
- **Arquitecturas:** AE = Flatten → Dense(latente, ReLU) → Dense(784, sigmoide) → Reshape. VAE = Flatten → Dense(128, ReLU) → (μ, log σ²) → muestreo por reparametrización → Dense(128, ReLU) → Dense(784, sigmoide). Pérdida del VAE: entropía cruzada binaria sumada por imagen más divergencia KL.
- **Entrenamiento:** Adam, lote de 256, 6 épocas para el AE (con parada anticipada, paciencia 2, 10 % de validación) y 8 para el VAE. Semillas 42, 43 y 44. Python 3.12.13, TensorFlow 2.21.0, CPU.
- **Tamaño y costo:** AE-16 26 mil parámetros, 331 KB, ≈ 7 s; VAE-16 208 mil parámetros, 854 KB, ≈ 15 s (ver README).

## Uso previsto

Material didáctico: estudiar reconstrucción, espacio latente, muestreo e interpolación. Exploración con la app de Streamlit del laboratorio.

## Usos fuera de alcance

- Generar imágenes que se presenten como reales o como fotos de productos.
- Aplicarlo a otros datos (rostros, imágenes médicas, imágenes de personas) sin una evaluación nueva: nada de lo medido aquí se extiende a esos dominios.
- Afirmar privacidad: que una muestra sea sintética no garantiza que no se parezca a datos de entrenamiento.
- Cualquier decisión automática sobre personas.

## Datos

Fashion-MNIST (Xiao, Rasul y Vollgraf, 2017), licencia MIT: 60,000 imágenes de entrenamiento y 10,000 de prueba, 10 clases de prendas de Zalando. No contiene datos personales. Limitaciones: catálogo de un solo comercio, fondos limpios, centradas y en gris; sin diversidad de cuerpos, fotografías reales ni contextos. No se redistribuye con el repositorio.

## Evaluación (3 semillas, media ± desviación)

| Modelo | MSE recon. (test) | Fréchet píxeles | Diversidad (vs real) | Clases cubiertas (de 10) | Confianza clasif. | Vecino/real | Casi copias | Entren. (s) | Parámetros | Tamaño (KB) |
|---|---|---|---|---|---|---|---|---|---|---|
| AE-16 con N(0,I) | 0.0206 ± 0.0004 | 95.0 ± 0.1 | 0.46 ± 0.05 | 1.3 ± 0.6 | 0.97 ± 0.01 | 1.79 ± 0.01 | 0.000 ± 0.000 | 7.3 ± 1.3 | 25888 ± 0 | 331 ± 0 |
| AE-16 gaussiana ajustada | 0.0206 ± 0.0004 | 15.3 ± 1.1 | 0.88 ± 0.01 | 10.0 ± 0.0 | 0.68 ± 0.02 | 1.33 ± 0.01 | 0.000 ± 0.001 | 7.3 ± 1.3 | 25888 ± 0 | 331 ± 0 |
| AE-2 gaussiana ajustada | 0.0585 ± 0.0001 | 44.8 ± 0.7 | 0.65 ± 0.01 | 8.3 ± 0.6 | 0.54 ± 0.02 | 1.14 ± 0.01 | 0.000 ± 0.000 | 7.1 ± 0.2 | 3922 ± 0 | 73 ± 0 |
| VAE-2 | 0.0330 ± 0.0001 | 36.8 ± 4.4 | 0.55 ± 0.04 | 9.3 ± 0.6 | 0.55 ± 0.03 | 0.77 ± 0.04 | 0.142 ± 0.032 | 15.4 ± 0.6 | 202516 ± 0 | 833 ± 0 |
| VAE-16 | 0.0165 ± 0.0001 | 17.5 ± 0.7 | 0.68 ± 0.01 | 10.0 ± 0.0 | 0.66 ± 0.01 | 1.09 ± 0.02 | 0.002 ± 0.002 | 14.8 ± 0.4 | 207920 ± 0 | 854 ± 0 |

Piso de ruido del Fréchet (real contra real): 3.8. Los 16 ejemplos de los paneles se fijan con la semilla 2026, sin selección manual.

## Hallazgos

- VAE-16 reconstruye mejor que AE-16, pero con 8 veces más parámetros (comparación confundida por capacidad).
- El AE muestreado con `N(0, I)` colapsa a casi una clase; el VAE no, porque su entrenamiento fuerza el espacio latente hacia esa distribución.
- AE-16 con gaussiana ajustada produce muestras de calidad comparable a VAE-16 con menos costo.
- VAE-2 confunde clases en 2 dimensiones (vestido contra pantalón; camiseta, camisa, suéter y abrigo se superponen).
- Sin evidencia de memorización: ninguna copia exacta, brecha entrenamiento-prueba cercana a cero y vecinos distintos a simple vista. La señal de VAE-2 en la métrica de vecino más cercano podría deberse al brillo (muestras oscuras cerca de prendas oscuras); es una hipótesis plausible, no una causa demostrada, y queda pendiente comprobarla con una distancia que no dependa del brillo.

## Límites y riesgos

- Muestras borrosas, poco diversas (diversidad 0.46 a 0.88 frente a imágenes reales) y lejos del piso de ruido.
- Entrenamiento corto, sin ajuste de hiperparámetros y solo 3 semillas.
- Métricas aproximadas: Fréchet en píxeles, clasificador auxiliar débil (exactitud 0.853 ± 0.004) y memoria por distancia en píxeles.
- La seguridad del clasificador no mide calidad: el AE con `N(0, I)` obtiene 0.97 de confianza con una sola clase.
- No hay auditoría de privacidad (ataques de inferencia de pertenencia o extracción).
- Posible uso indebido: un generador de imágenes sintéticas puede usarse para engañar. Aquí el riesgo es bajo por la baja calidad, pero no se mitiga con ninguna marca de agua.

## Consideraciones éticas y de privacidad

Los datos no contienen personas. Si el mismo código se aplicara a datos personales, habría que evaluar memorización con pruebas más fuertes, minimizar datos, evitar publicar muestras sin revisión y evaluar privacidad diferencial o equivalente.

## Decisión de uso

Como resultado exploratorio de este dominio, en el curso se usa **VAE-16** como generador de referencia, **AE-16 con gaussiana ajustada** como alternativa barata y **VAE-2** únicamente para visualización. Uso académico, no productivo; no se extrapola a otros dominios. Antes de cualquier otro dominio: más semillas, más épocas, métricas con red preentrenada, revisión humana y auditoría de privacidad.

## Reproducibilidad

`uv sync --extra cpu` y `uv run python scripts/e06_evaluacion.py --seeds 42 43 44`. Detalles y pruebas en el README. Los modelos entrenados (3 semillas) se conservan en `models/` con sus hashes en `models/MANIFEST.json`; `uv run python scripts/verificar_modelos.py` comprueba que cargan.

## Referencia

Xiao, H., Rasul, K., y Vollgraf, R. (2017). Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. arXiv:1708.07747.
