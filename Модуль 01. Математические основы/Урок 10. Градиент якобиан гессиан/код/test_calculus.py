import numpy as np

from matrix_calculus import linear_backward, logistic_hessian, numeric_jacobian, softmax, softmax_jacobian
from optimization import newton


def test_quadratic_form_gradient():
    rng = np.random.default_rng(0)
    A, x = rng.normal(size=(4, 4)), rng.normal(size=4)
    f = lambda v: v @ A @ v
    assert np.allclose(numeric_jacobian(f, x).ravel(), (A + A.T) @ x)       # ∇(x^T A x) = (A + A^T) x


def test_least_squares_gradient():
    rng = np.random.default_rng(1)
    A, b, x = rng.normal(size=(6, 3)), rng.normal(size=6), rng.normal(size=3)
    f = lambda v: np.sum((A @ v - b) ** 2)
    assert np.allclose(numeric_jacobian(f, x).ravel(), 2 * A.T @ (A @ x - b))


def test_chain_rule_is_jacobian_product():
    rng = np.random.default_rng(2)
    W1, W2 = rng.normal(size=(4, 3)), rng.normal(size=(2, 4))
    g = lambda x: np.tanh(W1 @ x)
    f = lambda u: W2 @ u
    x = rng.normal(size=3)
    J_composite = numeric_jacobian(lambda v: f(g(v)), x)
    J_product = W2 @ np.diag(1 - np.tanh(W1 @ x) ** 2) @ W1                # J_f · J_g
    assert np.allclose(J_composite, J_product)


def test_softmax_jacobian_symmetric_psd_like():
    J = softmax_jacobian(np.array([0.1, 2.0, -1.0]))
    assert np.allclose(J, J.T)
    assert np.allclose(J @ np.ones(3), 0)                                   # softmax не меняется от сдвига всех z


def test_linear_backward_shapes():
    X, W, dY = np.ones((5, 3)), np.ones((2, 3)), np.ones((5, 2))
    dW, dX, db = linear_backward(X, W, dY)
    assert dW.shape == W.shape and dX.shape == X.shape and db.shape == (2,)


def test_logistic_hessian_psd():
    rng = np.random.default_rng(3)
    X, y, w = rng.normal(size=(50, 4)), rng.integers(0, 2, 50).astype(float), rng.normal(size=4)
    assert np.linalg.eigvalsh(logistic_hessian(w, X, y)).min() > -1e-12


def test_newton_one_step_on_quadratic():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, -1.0])
    x, k = newton(lambda x: A @ x - b, lambda x: A, [10.0, -7.0])
    assert k == 1 and np.allclose(A @ x, b)                                 # квадратичная функция — ровно один шаг
