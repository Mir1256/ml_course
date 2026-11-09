import numpy as np
import pytest
from scipy import stats

from distributions import bce, cross_entropy_logits, entropy, kl, kl_gauss, mvn_logpdf


def test_kl_nonnegative_and_zero_iff_equal():
    rng = np.random.default_rng(0)
    for _ in range(100):
        p, q = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))
        assert kl(p, q) >= 0
    assert kl(p, p) == pytest.approx(0)


def test_entropy_max_is_uniform():
    rng = np.random.default_rng(1)
    for _ in range(50):
        assert entropy(rng.dirichlet(np.ones(4))) <= np.log(4) + 1e-12


def test_ce_stable_for_huge_logits():
    logits = np.array([[1000.0, 0.0, -1000.0]])
    assert cross_entropy_logits(logits, np.array([0])) == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(cross_entropy_logits(logits, np.array([2])))


def test_bce_perfect_and_worst():
    y = np.array([1, 0])
    assert bce(y, np.array([1.0, 0.0])) == pytest.approx(0.0, abs=1e-9)
    assert bce(y, np.array([0.5, 0.5])) == pytest.approx(np.log(2))


def test_kl_gauss_zero_and_positive():
    assert kl_gauss(0, 1, 0, 1) == pytest.approx(0.0)
    assert kl_gauss(0, 2, 0, 1) > 0


def test_mvn_matches_scipy_3d():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(3, 3))
    Sigma = A @ A.T + np.eye(3)
    mu, x = rng.normal(size=3), rng.normal(size=(10, 3))
    assert np.allclose(mvn_logpdf(x, mu, Sigma), stats.multivariate_normal(mu, Sigma).logpdf(x))
