import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from inf8239_u03_gen.models import build_autoencoder, build_encoder_decoder


def test_autoencoder_output_contract():
    model = build_autoencoder(tf)
    result = model(np.zeros((2, 28, 28, 1), dtype="float32")).numpy()
    assert result.shape == (2, 28, 28, 1)


def test_decoder_output_contract():
    _, decoder = build_encoder_decoder(tf, latent_dim=2)
    result = decoder(np.zeros((2, 2), dtype="float32")).numpy()
    assert result.shape == (2, 28, 28, 1)
