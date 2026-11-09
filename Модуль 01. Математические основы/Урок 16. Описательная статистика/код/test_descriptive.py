import numpy as np
from scipy import stats

from descriptive import iqr_outliers, mad, quantile, robust_z, skewness


def test_quantile_matches_numpy():
    x = np.random.default_rng(0).lognormal(size=1001)
    for q in (0, 0.1, 0.25, 0.5, 0.9, 1):
        assert np.isclose(quantile(x, q), np.quantile(x, q))


def test_mad_and_skew_match_scipy():
    x = np.random.default_rng(1).exponential(size=5000)
    assert np.isclose(mad(x), stats.median_abs_deviation(x))
    assert np.isclose(skewness(x), stats.skew(x))


def test_median_robust_mean_not():
    x = np.ones(100)
    x_bad = np.append(x, 1e6)
    assert np.median(x_bad) == 1.0 and x_bad.mean() > 9000


def test_robust_z_ignores_outlier_scale():
    x = np.random.default_rng(2).normal(size=1000)
    x[0] = 1000
    z_classic = (x - x.mean()) / x.std()
    assert abs(robust_z(x)[0]) > 500 and abs(z_classic[0]) < 32          # классический z «съеден» самим выбросом


def test_iqr_outliers():
    x = np.array([1, 2, 3, 4, 5, 100.0])
    assert iqr_outliers(x).tolist() == [False] * 5 + [True]
