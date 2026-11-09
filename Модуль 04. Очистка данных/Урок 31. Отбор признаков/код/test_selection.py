import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from selection import entropy_discrete, forward_selection, mutual_information_discrete, permutation_importance_np


def test_mi_bounds():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    assert mutual_information_discrete(y, y) == pytest.approx(entropy_discrete(y))      # I(Y;Y) = H(Y)
    assert mutual_information_discrete(rng.integers(0, 4, 5000), y) < 0.01              # независимые ≈ 0


def test_mi_nonlinear_dependence():
    rng = np.random.default_rng(1)
    x = rng.integers(-3, 4, 5000)
    y = (x**2 > 4).astype(int)                       # корреляция ≈ 0, но связь полная
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.05
    assert mutual_information_discrete(x, y) == pytest.approx(entropy_discrete(y))


def test_forward_selection_finds_signal():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 8))
    y = (X[:, 3] + 0.5 * X[:, 6] > 0).astype(int)
    assert set(forward_selection(LogisticRegression(), X, y, k=2)) == {3, 6}


def test_permutation_importance_noise_near_zero():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(600, 3))
    y = (X[:, 0] > 0).astype(int)
    imp = permutation_importance_np(LogisticRegression().fit(X, y), X, y)
    assert imp[0] > 0.3 and abs(imp[1]) < 0.02 and abs(imp[2]) < 0.02
