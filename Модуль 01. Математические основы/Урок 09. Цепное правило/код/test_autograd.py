import math

import pytest

from autograd import Value


def numeric_grad(f, xs, i, h=1e-6):
    plus = [x + (h if j == i else 0) for j, x in enumerate(xs)]
    minus = [x - (h if j == i else 0) for j, x in enumerate(xs)]
    return (f(*plus) - f(*minus)) / (2 * h)


@pytest.mark.parametrize("fn", [
    lambda a, b: a * b + b ** 3,
    lambda a, b: (a / b).exp() - a,
    lambda a, b: (a * a + 1).log() * b.tanh(),
    lambda a, b: (a - b).relu() * a + 2 / b,
])
def test_matches_numeric(fn):
    xs = [0.7, 1.3]
    vs = [Value(x) for x in xs]
    out = fn(*vs)
    out.backward()
    as_float = lambda a, b: fn(Value(a), Value(b)).data
    for i, v in enumerate(vs):
        assert v.grad == pytest.approx(numeric_grad(as_float, xs, i), rel=1e-6, abs=1e-8)


def test_fan_out_accumulates():
    a = Value(2.0)
    (a * a * a).backward()                 # d(a^3)/da = 3a^2 = 12, через ветвления
    assert a.grad == pytest.approx(12.0)


def test_backward_twice_resets():
    a = Value(1.0)
    f = a * 5
    f.backward()
    f.backward()                           # повторный вызов не должен удваивать градиент
    assert a.grad == 5.0


def test_chain_of_sigmoid_like():
    x = Value(0.3)
    y = 1 / (1 + (-x).exp())
    y.backward()
    s = 1 / (1 + math.exp(-0.3))
    assert x.grad == pytest.approx(s * (1 - s))
