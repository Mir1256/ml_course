import numpy as np
from scipy import stats

from density import fd_bin_width, histogram_density, kde


def test_histogram_integrates_to_one():
    x = np.random.default_rng(0).normal(size=3000)
    h = fd_bin_width(x)
    _, dens = histogram_density(x, h)
    assert np.isclose(np.sum(dens * h), 1.0)


def test_kde_matches_scipy_and_integrates():
    x = np.random.default_rng(1).exponential(size=500)
    grid = np.linspace(-2, 10, 2000)
    f = kde(x, grid)
    assert np.allclose(f, stats.gaussian_kde(x)(grid))
    assert np.isclose(np.trapezoid(f, grid), 1.0, atol=1e-3)


def test_small_bandwidth_is_spiky():
    x = np.array([0.0, 1.0])
    grid = np.linspace(-1, 2, 301)
    assert kde(x, grid, bandwidth=0.01)[150] < 1e-6       # между точками почти ноль
