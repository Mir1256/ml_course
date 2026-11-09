import numpy as np
import pytest

from svd import homography_dlt, low_rank, pinv_svd, svd_via_eigh


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (4, 4)])
def test_svd_reconstructs(shape):
    A = np.random.default_rng(0).normal(size=shape)
    U, s, Vt = svd_via_eigh(A)
    assert np.allclose((U * s) @ Vt, A)
    assert np.allclose(s, np.linalg.svd(A, compute_uv=False)[: len(s)])


def test_rank_deficient():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 2)) @ rng.normal(size=(2, 5))       # ранг 2
    U, s, Vt = svd_via_eigh(A, tol=1e-6)
    assert len(s) == 2
    assert np.allclose((U * s) @ Vt, A)


def test_eckart_young_beats_random_rank_k():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(20, 15))
    best = np.linalg.norm(A - low_rank(A, 3), "fro")
    for _ in range(50):
        B = rng.normal(size=(20, 3)) @ rng.normal(size=(3, 15))
        assert np.linalg.norm(A - B, "fro") >= best


def test_pinv_min_norm_solution():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(3, 6))                                  # недоопределённая: бесконечно много решений
    b = rng.normal(size=3)
    x = pinv_svd(A) @ b
    assert np.allclose(A @ x, b)
    null_dir = np.linalg.svd(A)[2][-1]                           # направление из ядра A
    assert np.linalg.norm(x) < np.linalg.norm(x + 0.5 * null_dir)


def test_dlt_exact():
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.3]])
    H = np.array([[2, 0.1, 5], [0.2, 1.5, -3], [0.001, 0.002, 1]])
    p = np.c_[src, np.ones(5)] @ H.T
    assert np.allclose(homography_dlt(src, p[:, :2] / p[:, 2:]), H)
