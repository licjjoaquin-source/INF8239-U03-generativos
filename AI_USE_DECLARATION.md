# Declaración de uso de inteligencia artificial · U03.E06

**Estudiante:** Johanna Joaquín · INF-8239 Ciencia de Datos II · UASD
*Borrador preparado con ayuda de IA: debe revisarse y ajustarse a lo que realmente se hizo antes de entregarlo.*

## Herramientas

Claude (Anthropic), asistente conversacional, usado en la sesión de trabajo del ejercicio.

## Para qué se usó

- Interpretar el mandato y la rúbrica del ejercicio y planificar los pasos.
- Proponer el diseño de la evaluación (cuatro modelos, tres semillas, métricas de fidelidad, diversidad y memoria).
- Escribir `gen_metrics.py`, `vae.py`, `e06_evaluacion.py` y las pruebas nuevas.
- Redactar borradores del README, la Model Card, el notebook y el informe PDF.

## Prompts relevantes (resumen)

1. Pegué el mandato del ejercicio (rúbrica, entregables) y el manual del LAB10 y pedí los pasos para resolverlo.
2. Pegué las salidas de la terminal (entrenamiento, pruebas) y el código del proyecto base para que se adaptara la evaluación.
3. Pedí una evaluación que cubra reconstrucción, muestreo, interpolación, fidelidad, diversidad, memoria y costo, con varias semillas.
4. Pegué los resultados y las figuras para su interpretación y la decisión final.

## Qué se verificó

- Ejecuté las 16 pruebas y todas pasan.
- Ejecuté la evaluación completa en mi computadora (3 semillas) y revisé las figuras generadas.
- Contrasté las cifras del README, la Model Card y el PDF con `reports/e06_tabla_principal.md` y `e06_resumen.json`.
- Revisé a simple vista los paneles de reconstrucción, muestras, interpolación y vecinos.

## Correcciones hechas a la salida de la IA

- La distancia al vecino más cercano calculada con la fórmula expandida en `float32` no detectaba copias exactas (distancia cercana a 1e-3 en lugar de 0). Una prueba lo reveló y se corrigió recalculando la distancia exacta del vecino ganador.
- La métrica de vecino más cercano sugería memorización en VAE-2. Al revisar las imágenes se vio que eran muestras oscuras cerca de prendas oscuras, y la interpretación se corrigió: el brillo es una explicación plausible, no una causa demostrada, y no se encontró evidencia de memorización.
- Se advirtió que comparar AE y VAE del laboratorio mezcla el objetivo con la capacidad (el AE no tiene capa oculta) y se declaró como límite.
- Se añadió el control AE-2 y el muestreo del AE con una gaussiana ajustada para que la comparación con el VAE sea más justa.
- Tras la devolución del profesor: el brillo se redactó como hipótesis, las recomendaciones quedaron como resultados exploratorios de este dominio y se conservaron los modelos entrenados en `models/`. Reejecuté la evaluación y comprobé con `scripts/verificar_modelos.py` que los modelos coinciden con su manifiesto y cargan.

## Responsabilidad

Las conclusiones y la decisión final las reviso y las puedo explicar. Los datos (Fashion-MNIST, licencia MIT) no contienen información personal y no se compartieron datos privados con la herramienta.
