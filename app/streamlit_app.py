from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/decoder.keras"
st.title("INF-8239 · Explorador del espacio latente")
st.caption("Demostración académica con Fashion-MNIST. Una muestra sintética no garantiza originalidad ni privacidad.")
if not MODEL_PATH.exists():
    st.error("Entrene el modelo antes de abrir la aplicación.")
    st.stop()
decoder = tf.keras.models.load_model(MODEL_PATH)
x = st.slider("Latente X", -3.0, 3.0, 0.0, 0.1)
y = st.slider("Latente Y", -3.0, 3.0, 0.0, 0.1)
image = decoder.predict(np.array([[x, y]]), verbose=0)[0].squeeze()
st.image(image, caption=f"Punto latente ({x:.1f}, {y:.1f})", clamp=True, width=280)
