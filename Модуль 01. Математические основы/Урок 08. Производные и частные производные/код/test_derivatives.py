import numpy as np
import pytest

from derivatives import central_diff, forward_diff, partial, relu_grad, sigmoid, sigmoid_grad, softplus


def test_central_is_second_order():
    # ошибка central ~ h^2: уменьшили h в 10 раз -> ошибка в ~100 раз
    e1 = abs(central_diff(np.exp, 0.5, 1e-2) - np.exp(0.5))
    e2 = abs(central_diff(np.exp, 0.5, 1e-3) - np.exp(0.5))
    assert 80 < e1 / e2 < 120


def test_forward_is_first_order():
    e1 = abs(forward_diff(np.exp, 0.5, 1e-2) - np.exp(0.5))
    e2 = abs(forward_diff(np.exp, 0.5, 1e-3) - np.exp(0.5))
    assert 8 < e1 / e2 < 12


def test_sigmoid_grad_max():
    assert sigmoid_grad(0.0) == pytest.approx(0.25)
    assert sigmoid_grad(10.0) < 1e-4                 # насыщение


def test_softplus_stable_and_derivative():
    assert np.isfinite(softplus(1000.0)) and softplus(1000.0) == pytest.approx(1000.0)
    assert central_diff(softplus, 0.3, 1e-5) == pytest.approx(sigmoid(0.3), abs=1e-9)


def test_relu_grad():
    assert relu_grad(np.array([-1.0, 0.0, 2.0])).tolist() == [0.0, 0.0, 1.0]


def test_partial_derivatives():
    f = lambda p: p[0] ** 3 + 2 * p[0] * p[1]
    p = np.array([1.0, 3.0])
    assert partial(f, p, 0) == pytest.approx(3 + 6, abs=1e-6)
    assert partial(f, p, 1) == pytest.approx(2, abs=1e-6)
