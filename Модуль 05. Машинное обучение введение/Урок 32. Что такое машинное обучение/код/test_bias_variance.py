import numpy as np

from bias_variance import SIGMA, decompose, poly


def test_decomposition_adds_up():
    b, v, t = decompose(poly(3), n_sets=2000)
    assert abs((SIGMA**2 + b + v) - t) < 0.01           # σ² + bias² + variance ≈ ожидаемая ошибка


def test_complexity_tradeoff():
    b0, v0, _ = decompose(poly(0), n_sets=300)
    b9, v9, _ = decompose(poly(9), n_sets=300)
    assert b0 > b9 and v0 < v9                           # простая модель: смещение; сложная: разброс
