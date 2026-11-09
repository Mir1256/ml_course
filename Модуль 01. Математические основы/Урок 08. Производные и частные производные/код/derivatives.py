"""Производные с нуля: конечные разности и их ошибка, производные активаций, ряд Тейлора, частные производные."""
import numpy as np


def forward_diff(f, x: float, h: float) -> float:
    """(f(x+h) - f(x)) / h — ошибка O(h)."""
    return (f(x + h) - f(x)) / h


def central_diff(f, x: float, h: float) -> float:
    """(f(x+h) - f(x-h)) / 2h — ошибка O(h^2): члены с f'' сокращаются."""
    return (f(x + h) - f(x - h)) / (2 * h)


def partial(f, x: np.ndarray, i: int, h: float = 1e-5) -> float:
    """Частная производная df/dx_i: меняем только i-ю координату."""
    e = np.zeros_like(x, dtype=float)
    e[i] = h
    return (f(x + e) - f(x - e)) / (2 * h)


# ---------- Активации и их производные (понадобятся в уроке 65) ----------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)                       # σ' = σ(1-σ): максимум 0.25 при x=0 -> затухание градиента


def tanh_grad(x):
    return 1 - np.tanh(x) ** 2


def softplus(x):
    return np.logaddexp(0, x)                # log(1 + e^x) без переполнения при больших x


def relu_grad(x):
    return (x > 0).astype(float)             # в нуле производной нет; по соглашению берём 0 (субградиент)


if __name__ == "__main__":
    f, df = np.sin, np.cos
    x0 = 1.0
    print(" h        forward err   central err")
    for h in (1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11):
        ef = abs(forward_diff(f, x0, h) - df(x0))
        ec = abs(central_diff(f, x0, h) - df(x0))
        print(f"{h:.0e}   {ef:.2e}      {ec:.2e}")

    xs = np.linspace(-6, 6, 13)
    for name, fn, grad in (("sigmoid", sigmoid, sigmoid_grad), ("tanh", np.tanh, tanh_grad), ("softplus", softplus, sigmoid)):
        num = np.array([central_diff(fn, x, 1e-5) for x in xs])
        print(f"{name:<8}: аналитическая == численная: {np.allclose(num, grad(xs), atol=1e-8)}, max f' = {grad(xs).max():.3f}")

    # Ряд Тейлора sigmoid в нуле: σ(x) ≈ 1/2 + x/4 - x^3/48
    for x in (0.1, 0.5, 1.0, 2.0):
        t1, t3 = 0.5 + x / 4, 0.5 + x / 4 - x**3 / 48
        print(f"x={x}: σ={sigmoid(x):.5f}, Тейлор-1 {t1:.5f} (ошибка {abs(t1 - sigmoid(x)):.1e}), Тейлор-3 {t3:.5f} (ошибка {abs(t3 - sigmoid(x)):.1e})")

    # Частные производные f(x, y) = x^2 y + e^{xy}
    g = lambda p: p[0] ** 2 * p[1] + np.exp(p[0] * p[1])
    p = np.array([1.0, 2.0])
    analytic = [2 * p[0] * p[1] + p[1] * np.exp(p[0] * p[1]), p[0] ** 2 + p[0] * np.exp(p[0] * p[1])]
    print("∂f/∂x, ∂f/∂y численно:", np.round([partial(g, p, i) for i in range(2)], 6), "| аналитически:", np.round(analytic, 6))
