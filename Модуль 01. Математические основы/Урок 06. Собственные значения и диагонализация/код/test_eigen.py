import numpy as np

from eigen import gd_on_quadratic, power_iteration, qr_algorithm


def test_power_iteration_matches_eigh():
    B = np.random.default_rng(1).normal(size=(6, 6))
    S = B @ B.T
    lam, v, _ = power_iteration(S)
    assert np.isclose(lam, np.linalg.eigvalsh(S)[-1])
    assert np.allclose(S @ v, lam * v, atol=1e-6)


def test_qr_algorithm_spectrum():
    B = np.random.default_rng(2).normal(size=(4, 4))
    S = B + B.T
    vals, V = qr_algorithm(S, iters=2000)
    assert np.allclose(np.sort(vals), np.linalg.eigvalsh(S), atol=1e-8)
    assert np.allclose(V.T @ V, np.eye(4))


def test_trace_and_det_from_spectrum():
    B = np.random.default_rng(3).normal(size=(5, 5))
    w = np.linalg.eigvals(B)                          # у несимметричной — возможно комплексные
    assert np.isclose(w.sum().real, np.trace(B))
    assert np.isclose(np.prod(w).real, np.linalg.det(B))


def test_covariance_is_psd():
    X = np.random.default_rng(4).normal(size=(100, 7))
    C = np.cov(X, rowvar=False)
    assert np.all(np.linalg.eigvalsh(C) >= -1e-12)


def test_gd_diverges_above_threshold():
    H = np.diag([1.0, 4.0])                           # 2 / λ_max = 0.5
    assert gd_on_quadratic(H, 0.45)[-1] < 1e-6
    assert gd_on_quadratic(H, 0.55)[-1] > 1e6
