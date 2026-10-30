import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from inf8239_u03_gen.models import build_autoencoder, build_encoder_decoder
from inf8239_u03_gen.vae import make_vae_class, split_autoencoder


def test_vae_trains_one_epoch_with_finite_loss():
    encoder, decoder = build_encoder_decoder(tf, latent_dim=2)
    vae = make_vae_class(tf)(encoder, decoder)
    vae.compile(optimizer="adam")
    data = np.random.default_rng(0).random((64, 28, 28, 1)).astype("float32")
    history = vae.fit(data, epochs=1, batch_size=32, verbose=0)
    assert np.isfinite(history.history["loss"][-1])
    assert history.history["kl"][-1] >= 0


def test_split_autoencoder_matches_full_model():
    model = build_autoencoder(tf, latent_dim=4)
    encoder, decoder = split_autoencoder(tf, model)
    data = np.random.default_rng(1).random((3, 28, 28, 1)).astype("float32")
    latent = encoder.predict(data, verbose=0)
    assert latent.shape == (3, 4)
    assert np.allclose(decoder.predict(latent, verbose=0), model.predict(data, verbose=0), atol=1e-5)
