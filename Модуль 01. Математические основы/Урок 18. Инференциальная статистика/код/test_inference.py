import numpy as np
from scipy import stats

from inference import bootstrap_ci, mcnemar_exact, wald_ci, wilson_ci


def test_wilson_inside_unit_interval():
    lo, hi = wilson_ci(30, 30)
    assert 0 <= lo < hi <= 1                      # Wald дал бы вырожденный интервал [1, 1]
    assert wald_ci(30, 30) == (1.0, 1.0)


def test_wilson_matches_scipy():
    ours = wilson_ci(37, 50)
    ref = stats.binomtest(37, 50).proportion_ci(method="wilson")
    assert np.allclose(ours, (ref.low, ref.high), atol=1e-6)


def test_bootstrap_ci_contains_mean_and_shrinks():
    x = np.random.default_rng(0).normal(size=1000)
    lo, hi = bootstrap_ci(x, lambda s: s.mean(axis=1), n_boot=2000)
    lo2, hi2 = bootstrap_ci(x[:100], lambda s: s.mean(axis=1), n_boot=2000)
    assert lo < x.mean() < hi and (hi - lo) < (hi2 - lo2)


def test_mcnemar_identical_models():
    a = np.random.default_rng(1).random(200) < 0.8
    assert mcnemar_exact(a, a.copy()) == 1.0
