"""Якобианы и матричное дифференцирование: численная проверка, softmax, линейный слой, гессиан логистической потери."""
import numpy as np


def numeric_jacobian(f, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """J[i, j] = d f_i / d x_j центральными разностями: столбец j — реакция всех выходов на толчок x_j."""
    fx = np.atleast_1d(f(x))
    J = np.zeros((fx.size, x.size))
    for j in range(x.size):
        e = np.zeros_like(x)
        e.flat[j] = h
        J[:, j] = (np.atleast_1d(f(x + e)) - np.atleast_1d(f(x - e))).ravel() / (2 * h)
    return J


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())                 # сдвиг на максимум: тот же результат, но без переполнения
    return e / e.sum()


def softmax_jacobian(z: np.ndarray) -> np.ndarray:
    """d p_i / d z_j = p_i (δ_ij - p_j)  ->  J = diag(p) - p p^T."""
    p = softmax(z)
    return np.diag(p) - np.outer(p, p)


def linear_backward(X: np.ndarray, W: np.ndarray, dY: np.ndarray):
    """Y = X W^T + b. Градиенты через «совпадение форм»: dW = dY^T X, dX = dY W, db = sum(dY)."""
    return dY.T @ X, dY @ W, dY.sum(axis=0)


def logistic_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Средняя логистическая потеря, y ∈ {0, 1}."""
    z = X @ w
    return float(np.mean(np.logaddexp(0, z) - y * z))


def logistic_grad(w, X, y):
    p = 1 / (1 + np.exp(-X @ w))
    return X.T @ (p - y) / len(y)            # «ошибка предсказания × вход»


def logistic_hessian(w, X, y):
    p = 1 / (1 + np.exp(-X @ w))
    return (X.T * (p * (1 - p))) @ X / len(y)    # X^T diag(p(1-p)) X — PSD, значит потеря выпукла


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    z = rng.normal(size=5)
    print("якобиан softmax: аналитический == численный:", np.allclose(softmax_jacobian(z), numeric_jacobian(softmax, z)))
    print("суммы строк якобиана softmax ≈ 0:", np.allclose(softmax_jacobian(z).sum(axis=1), 0))
    y_onehot = np.eye(5)[2]
    ce = lambda zz: -np.log(softmax(zz)[2])
    print("grad CE(softmax) = p - y:", np.allclose(numeric_jacobian(ce, z).ravel(), softmax(z) - y_onehot))

    X, W, b = rng.normal(size=(4, 3)), rng.normal(size=(2, 3)), rng.normal(size=2)
    dY = rng.normal(size=(4, 2))                                        # градиент, пришедший сверху
    dW, dX, db = linear_backward(X, W, dY)
    L_of_W = lambda w_flat: np.sum((X @ w_flat.reshape(2, 3).T + b) * dY)   # L = <Y, dY>, так что dL/dY = dY
    print("dW линейного слоя совпадает с численным:", np.allclose(dW.ravel(), numeric_jacobian(L_of_W, W.ravel()).ravel()))

    Xl = np.c_[np.ones(200), rng.normal(size=(200, 2))]
    yl = (Xl @ np.array([0.5, 2.0, -1.0]) + rng.normal(size=200) > 0).astype(float)
    w = rng.normal(size=3) * 0.1
    g_num = numeric_jacobian(lambda ww: logistic_loss(ww, Xl, yl), w).ravel()
    H_num = numeric_jacobian(lambda ww: logistic_grad(ww, Xl, yl), w)
    print("градиент логистической потери верен:", np.allclose(logistic_grad(w, Xl, yl), g_num, atol=1e-8))
    print("гессиан логистической потери верен:", np.allclose(logistic_hessian(w, Xl, yl), H_num, atol=1e-6))
    print("собственные значения гессиана (все ≥ 0 -> выпукла):", np.linalg.eigvalsh(logistic_hessian(w, Xl, yl)).round(4))

    try:
        import torch
        from torch.autograd.functional import hessian, jacobian

        zt = torch.tensor(z)
        print("torch jacobian softmax совпадает:", np.allclose(jacobian(lambda t: torch.softmax(t, 0), zt).numpy(), softmax_jacobian(z)))
        Xt, yt = torch.tensor(Xl), torch.tensor(yl)
        f = lambda ww: torch.mean(torch.nn.functional.softplus(Xt @ ww) - yt * (Xt @ ww))
        print("torch hessian совпадает:", np.allclose(hessian(f, torch.tensor(w)).numpy(), logistic_hessian(w, Xl, yl)))
    except ImportError:
        print("torch не установлен — сверка пропущена")
