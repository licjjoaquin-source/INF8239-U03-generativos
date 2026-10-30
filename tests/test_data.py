import numpy as np
import pytest

from inf8239_u03_gen.data import normalize_images, validate_images


def test_normalization_contract():
    raw = np.array([[[0] * 28] * 28, [[255] * 28] * 28], dtype=np.uint8)
    result = normalize_images(raw)
    validate_images(result)
    assert result.shape == (2, 28, 28, 1)
    assert result.min() == 0 and result.max() == 1


def test_rejects_wrong_shape():
    with pytest.raises(ValueError, match="Forma inesperada"):
        validate_images(np.zeros((2, 28, 28)))
