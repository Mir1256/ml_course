"""Распределения и откуда берутся функции потерь: MLE -> MSE / BCE / CE; энтропия, кросс-энтропия, KL."""
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)


def gaussian_nll(y: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    """-log p(y | mu, sigma) для независимых нормальных наблюдений."""
    return float(np.sum(0.5 * np.log(2 * np.pi * sigma**2) + (y - mu) ** 2 / (2 * sigma**2)))


def bce(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """-(1/n) Σ [y log p + (1-y) log(1-p)] — это -log-правдоподобие Бернулли, делённое на n."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cross_entropy_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """-(1/n) Σ log softmax(z)_y через log-sum-exp — так считает nn.CrossEntropyLoss."""
    z = logits - logits.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-np.mean(log_probs[np.arange(len(labels)), labels]))


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]                                         # 0 log 0 = 0 по непрерывности
    return float(-np.sum(p * np.log(p)))


def kl(p: np.ndarray, q: np.ndarray) -> float:
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def kl_gauss(mu1, s1, mu2, s2) -> float:
    """KL(N(mu1, s1²) || N(mu2, s2²)) в замкнутой форме — та самая формула из VAE."""
    return float(np.log(s2 / s1) + (s1**2 + (mu1 - mu2) ** 2) / (2 * s2**2) - 0.5)


def mvn_logpdf(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """log N(x; mu, Σ) через разложение Холецкого: без явной обратной матрицы и определителя."""
    d = mu.size
    L = np.linalg.cholesky(Sigma)                        # Σ = L L^T
    z = np.linalg.solve(L, (x - mu).T)                   # z = L^{-1}(x - mu); ||z||² = расстояние Махаланобиса²
    log_det = 2 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2 * np.pi) + log_det + np.sum(z**2, axis=0))


if __name__ == "__main__":
    # 1. MLE нормального распределения = выборочные среднее и (смещённая) дисперсия
    data = rng.normal(3.0, 2.0, size=1000)
    mu_hat, sigma_hat = data.mean(), data.std()          # ddof=0 — это и есть MLE
    print(f"MLE: mu = {mu_hat:.4f}, sigma = {sigma_hat:.4f} | scipy norm.fit: {np.round(stats.norm.fit(data), 4)}")

    # 2. Гауссово NLL = MSE / (2σ²) + const: минимизация по mu та же
    y = rng.normal(size=50)
    for mu in (0.0, 0.5):
        pred = np.full(50, mu)
        nll = gaussian_nll(y, pred, sigma=1.0)
        mse = np.mean((y - pred) ** 2)
        print(f"mu={mu}: NLL = {nll:.4f} = n·MSE/2 + n·log(2π)/2 = {50 * mse / 2 + 25 * np.log(2 * np.pi):.4f}")

    # 3. BCE = -log-правдоподобие Бернулли; CE = -log-правдоподобие категориального
    yb = rng.integers(0, 2, 100)
    pb = rng.random(100)
    print(f"BCE = {bce(yb, pb):.4f} | -mean log Bernoulli.pmf = {-stats.bernoulli.logpmf(yb, pb).mean():.4f}")
    logits, labels = rng.normal(size=(8, 5)), rng.integers(0, 5, 8)
    try:
        import torch
        print(f"CE свой = {cross_entropy_logits(logits, labels):.6f} | torch = {torch.nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(labels)).item():.6f}")
    except ImportError:
        print(f"CE свой = {cross_entropy_logits(logits, labels):.6f}")

    # 4. Энтропия, кросс-энтропия, KL
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.4, 0.4, 0.2])
    print(f"\nH(p) = {entropy(p):.4f} (scipy {stats.entropy(p):.4f}) | равномерное: {entropy(np.ones(3) / 3):.4f} = log 3")
    print(f"KL(p||q) = {kl(p, q):.4f} (scipy {stats.entropy(p, q):.4f}) | KL(q||p) = {kl(q, p):.4f}  <- несимметрична")
    print(f"H(p, q) = {-np.sum(p * np.log(q)):.4f} = H(p) + KL(p||q) = {entropy(p) + kl(p, q):.4f}")

    # 5. KL двух гауссиан: формула против Монте-Карло
    s = rng.normal(1.0, 0.5, size=1_000_000)
    mc = np.mean(stats.norm(1.0, 0.5).logpdf(s) - stats.norm(0, 1).logpdf(s))
    print(f"KL(N(1, 0.5²) || N(0, 1)): формула {kl_gauss(1.0, 0.5, 0, 1):.4f}, Монте-Карло {mc:.4f}")

    # 6. Центральная предельная теорема: среднее 30 экспоненциальных ≈ нормальное
    means = rng.exponential(1.0, size=(100_000, 30)).mean(axis=1)
    print(f"\nЦПТ: асимметрия экспоненциального = 2.0, асимметрия среднего 30 штук = {stats.skew(means):.3f} (теория 2/sqrt(30) = {2 / np.sqrt(30):.3f})")

    # 7. Многомерная нормаль своими руками против scipy
    Sigma = np.array([[2.0, 0.8], [0.8, 1.0]])
    pts = rng.normal(size=(5, 2))
    print("MVN logpdf совпадает со scipy:", np.allclose(mvn_logpdf(pts, np.zeros(2), Sigma), stats.multivariate_normal(np.zeros(2), Sigma).logpdf(pts)))

    # 8. Пуассон — счётные данные: число объектов на кадре; MLE λ = выборочное среднее
    counts = rng.poisson(3.2, size=2000)
    print(f"Пуассон: MLE λ = {counts.mean():.3f}, дисперсия {counts.var():.3f} (у Пуассона среднее = дисперсии)")
