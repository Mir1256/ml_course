import numpy as np
import pytest

from vectors import channel_mean_std, cosine, hwc_uint8_to_chw_float, make_batch, norm, pairwise_cosine, project


def test_projection_residual_is_orthogonal():
    rng = np.random.default_rng(1)
    u, v = rng.normal(size=5), rng.normal(size=5)
    r = v - project(v, u)
    assert abs(r @ u) < 1e-10          # остаток перпендикулярен направлению проекции


def test_cosine_bounds_and_zero_vector():
    assert cosine(np.array([1.0, 0]), np.array([-3.0, 0])) == pytest.approx(-1.0)
    assert cosine(np.zeros(3), np.ones(3)) == 0.0     # не nan


def test_norm_triangle_inequality():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=10), rng.normal(size=10)
    for p in (1, 2, np.inf):
        assert norm(a + b, p) <= norm(a, p) + norm(b, p) + 1e-12


def test_pairwise_cosine_shape_and_diag():
    A = np.random.default_rng(3).normal(size=(7, 4))
    S = pairwise_cosine(A, A)
    assert S.shape == (7, 7)
    assert np.allclose(np.diag(S), 1.0)


def test_image_tensor_pipeline():
    img = np.full((4, 6, 3), 255, dtype=np.uint8)
    chw = hwc_uint8_to_chw_float(img)
    assert chw.shape == (3, 4, 6) and chw.dtype == np.float32 and chw.max() == 1.0
    batch = make_batch([chw, chw])
    mean, std = channel_mean_std(batch)
    assert batch.shape == (2, 3, 4, 6) and mean.shape == (3,) and np.allclose(std, 0)
