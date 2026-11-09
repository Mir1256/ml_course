"""Градиентный спуск против метода Ньютона, седловые точки и множители Лагранжа (PCA как задача с ограничением)."""
import numpy as np
from scipy.optimize import minimize


def rosenbrock(p):
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def rosenbrock_grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x**2), 200 * (y - x**2)])


def rosenbrock_hess(p):
    x, y = p
    return np.array([[2 - 400 * (y - x**2) + 800 * x**2, -400 * x], [-400 * x, 200.0]])


def gradient_descent(grad, x0, lr, tol=1e-8, max_iter=200_000):
    x = np.array(x0, float)
    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            return x, k
        x -= lr * g
    return x, max_iter


def newton(grad, hess, x0, tol=1e-8, max_iter=100):
    """x <- x - H^{-1} g: минимизируем квадратичную модель Тейлора на каждом шаге."""
    x = np.array(x0, float)
    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            return x, k
        x -= np.linalg.solve(hess(x), g)        # solve, а не inv (урок 5)
    return x, max_iter


if __name__ == "__main__":
    x0 = [-1.2, 1.0]
    x_gd, k_gd = gradient_descent(rosenbrock_grad, x0, lr=1e-3)
    x_nt, k_nt = newton(rosenbrock_grad, rosenbrock_hess, x0)
    print(f"Розенброк: GD -> {x_gd.round(6)} за {k_gd} итераций | Ньютон -> {x_nt.round(6)} за {k_nt} итераций")
    H_min = rosenbrock_hess([1.0, 1.0])
    ev = np.linalg.eigvalsh(H_min)
    print(f"гессиан в минимуме: λ = {ev.round(3)}, κ = {ev[-1] / ev[0]:.0f}  <- поэтому GD ползёт по «банановой» долине")

    # Классификация критических точек по спектру гессиана
    f = lambda p: p[0] ** 3 - 3 * p[0] + p[1] ** 2       # критические точки (±1, 0)
    hess_f = lambda p: np.array([[6 * p[0], 0], [0, 2.0]])
    for pt in ([1.0, 0.0], [-1.0, 0.0]):
        ev = np.linalg.eigvalsh(hess_f(pt))
        kind = "минимум" if ev.min() > 0 else ("максимум" if ev.max() < 0 else "седло")
        print(f"точка {pt}: λ(H) = {ev}, это {kind}")

    # Седло f = x^2 - y^2: из точной точки (1, 0) GD сходится в седло, малый шум уводит вдоль y
    saddle_grad = lambda p: np.array([2 * p[0], -2 * p[1]])
    for start in ([1.0, 0.0], [1.0, 1e-8]):
        x = np.array(start)
        for _ in range(60):
            x = x - 0.1 * saddle_grad(x)
        print(f"седло: старт {start} -> через 60 шагов {x.round(6)}")

    # Лагранж: max v^T C v при v^T v = 1  <=>  C v = λ v. Решаем как задачу оптимизации и сравниваем с eigh.
    rng = np.random.default_rng(0)
    data = rng.normal(size=(500, 3)) @ np.array([[3, 0, 0], [1, 1, 0], [0, 0.5, 0.2]])
    C = np.cov(data, rowvar=False)
    res = minimize(lambda v: -v @ C @ v, x0=np.ones(3) / np.sqrt(3),
                   constraints={"type": "eq", "fun": lambda v: v @ v - 1}, method="SLSQP")
    lam, vecs = np.linalg.eigh(C)
    print("Лагранж (SLSQP): v =", res.x.round(4), " max v^T C v =", round(-res.fun, 4))
    print("eigh:            v =", vecs[:, -1].round(4), " λ_max      =", round(lam[-1], 4))
