import numpy as np
import pytest
from scipy import stats

from bayes import beta_update, grid_posterior, posterior_positive


def test_base_rate():
    assert posterior_positive(0.01, 0.99, 0.95) == pytest.approx(0.0099 / (0.0099 + 0.0495))
    assert posterior_positive(0.5, 0.9, 0.9) == pytest.approx(0.9)


def test_perfect_specificity_means_certainty():
    assert posterior_positive(0.001, 0.5, 1.0) == pytest.approx(1.0)


def test_sequential_equals_batch_update():
    a, b = beta_update(2, 2, 3, 1)
    a, b = beta_update(a, b, 4, 2)
    assert (a, b) == beta_update(2, 2, 7, 3)          # порядок и разбиение данных не важны


def test_grid_matches_conjugate():
    theta, post = grid_posterior(3, 5, 4, 10)
    assert np.allclose(post, stats.beta.pdf(theta, 7, 11), atol=1e-6)


def test_uniform_prior_map_equals_mle():
    a, b = beta_update(1, 1, 7, 3)                    # Beta(1, 1) — равномерное распределение
    assert (a - 1) / (a + b - 2) == pytest.approx(0.7)
