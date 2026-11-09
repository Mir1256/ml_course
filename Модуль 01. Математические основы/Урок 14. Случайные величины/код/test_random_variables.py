import numpy as np
from scipy import stats

from random_variables import ecdf, inverse_transform_exponential


def test_ecdf_monotone_and_ends_at_one():
    xs, F = ecdf(np.array([3.0, 1.0, 2.0, 2.0]))
    assert np.all(np.diff(xs) >= 0) and np.all(np.diff(F) > 0) and F[-1] == 1.0


def test_inverse_transform_ks():
    rng = np.random.default_rng(1)
    samples = inverse_transform_exponential(rng.random(20_000), rate=3.0)
    assert stats.kstest(samples, stats.expon(scale=1 / 3).cdf).pvalue > 0.01


def test_linearity_of_expectation_without_independence():
    rng = np.random.default_rng(2)
    x = rng.normal(size=100_000)
    y = 3 * x + rng.normal(size=100_000)            # сильно зависимы
    assert np.isclose((x + y).mean(), x.mean() + y.mean())


def test_variance_of_sum():
    rng = np.random.default_rng(3)
    x = rng.normal(size=500_000)
    y = 0.5 * x + rng.normal(size=500_000)
    lhs = np.var(x + y)
    rhs = np.var(x) + np.var(y) + 2 * np.cov(x, y, ddof=0)[0, 1]
    assert np.isclose(lhs, rhs)


def test_covariance_matrix_psd():
    X = np.random.default_rng(4).normal(size=(1000, 5)) @ np.random.default_rng(5).normal(size=(5, 5))
    assert np.linalg.eigvalsh(np.cov(X, rowvar=False)).min() > 0
