import numpy as np
import pytest

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


def _images(n, seed=0):
    return np.random.default_rng(seed).random((n, 28, 28, 1)).astype("float32")


def test_fixed_indices_reproducible_unique_and_in_range():
    first = fixed_indices(10000, 16, seed=2026)
    second = fixed_indices(10000, 16, seed=2026)
    assert np.array_equal(first, second)
    assert len(set(first.tolist())) == 16
    assert first.min() >= 0 and first.max() < 10000


def test_fixed_indices_rejects_too_many():
    with pytest.raises(ValueError):
        fixed_indices(5, 10)


def test_reconstruction_mse_zero_and_known_value():
    x = _images(4)
    assert reconstruction_mse(x, x) == 0.0
    assert abs(reconstruction_mse(np.zeros((2, 28, 28, 1)), np.ones((2, 28, 28, 1))) - 1.0) < 1e-9


def test_nearest_neighbor_matches_brute_force():
    ref, queries = _images(40, 1), _images(7, 2)
    dist, idx = nearest_neighbor_distances(queries, ref, chunk=3)
    flat_r, flat_q = ref.reshape(40, -1), queries.reshape(7, -1)
    brute = np.linalg.norm(flat_q[:, None, :] - flat_r[None, :, :], axis=2)
    assert np.array_equal(idx, brute.argmin(axis=1))
    assert np.allclose(dist, brute.min(axis=1), atol=1e-3)


def test_memorization_detects_exact_copies():
    train, test = _images(200, 3), _images(100, 4)
    copies = train[:20]  # un "modelo" que memoriza
    fresh = _images(20, 5)
    leaky = memorization_report(copies, train, test)
    clean = memorization_report(fresh, train, test)
    assert leaky["exact_copy_share"] == 1.0
    assert leaky["near_copy_share"] == 1.0
    assert clean["exact_copy_share"] == 0.0
    assert leaky["nn_ratio"] < clean["nn_ratio"]


def test_mean_pairwise_distance_collapse_vs_diverse():
    collapsed = np.repeat(_images(1), 50, axis=0)
    diverse = _images(50, 6)
    assert mean_pairwise_distance(collapsed, n_pairs=500) == 0.0
    assert mean_pairwise_distance(diverse, n_pairs=500) > 1.0


def test_frechet_zero_for_same_and_positive_for_shift():
    x = _images(300, 7)
    assert frechet_distance_pixels(x, x) < 1e-3
    assert frechet_distance_pixels(x, np.clip(x + 0.3, 0, 1)) > 1.0


def test_class_coverage_bounds():
    full = class_coverage(np.repeat(np.arange(10), 10))
    single = class_coverage(np.zeros(100, dtype=int))
    assert full["classes_covered"] == 10 and abs(full["normalized_entropy"] - 1.0) < 1e-9
    assert single["classes_covered"] == 1 and single["normalized_entropy"] == 0.0


def test_interpolation_endpoints_and_shape():
    path = interpolation_path(np.array([0.0, 0.0]), np.array([2.0, 4.0]), steps=5)
    assert path.shape == (5, 2)
    assert np.allclose(path[0], [0, 0]) and np.allclose(path[-1], [2, 4])
    with pytest.raises(ValueError):
        interpolation_path(np.zeros(2), np.ones(2), steps=1)


def test_summarize_runs_mean_and_sample_std():
    rows = [{"model": "a", "x": 1.0}, {"model": "a", "x": 3.0}, {"model": "b", "x": 5.0}]
    summary = summarize_runs(rows)
    assert summary["a"]["x"] == (2.0, pytest.approx(2 ** 0.5))
    assert summary["b"]["x"] == (5.0, 0.0)
