"""Теорема Байеса и байесовский вывод: ловушка базовой частоты, Beta-Bernoulli, MLE против MAP."""
import numpy as np
from scipy import stats


def posterior_positive(prior: float, sensitivity: float, specificity: float) -> float:
    """P(дефект | тревога) = P(тревога | дефект) P(дефект) / P(тревога)."""
    p_alarm = sensitivity * prior + (1 - specificity) * (1 - prior)      # формула полной вероятности
    return sensitivity * prior / p_alarm


def beta_update(a: float, b: float, successes: int, failures: int) -> tuple[float, float]:
    """Сопряжённость: Beta(a, b) prior × Bernoulli likelihood -> Beta(a + k, b + n - k) posterior."""
    return a + successes, b + failures


def grid_posterior(a: float, b: float, k: int, n: int, grid_size: int = 10_001):
    """Тот же апостериор «в лоб»: prior * likelihood на сетке, затем нормировка."""
    theta = np.linspace(0, 1, grid_size)
    unnorm = stats.beta.pdf(theta, a, b) * theta**k * (1 - theta) ** (n - k)
    return theta, unnorm / np.trapezoid(unnorm, theta)


if __name__ == "__main__":
    # 1. Ловушка базовой частоты: редкий дефект, хороший детектор
    for prior in (0.01, 0.1, 0.5):
        print(f"P(дефект) = {prior:<4}: P(дефект | тревога) = {posterior_positive(prior, 0.99, 0.95):.3f}")

    # 2. Precision детектора на новой камере. Старые камеры: ~90% верных срабатываний -> prior Beta(18, 2)
    a0, b0 = 18, 2
    k, n = 7, 10                                             # на новой камере 7 верных из 10 проверенных
    a1, b1 = beta_update(a0, b0, k, n - k)
    mle = k / n
    map_ = (a1 - 1) / (a1 + b1 - 2)                          # мода Beta
    mean = a1 / (a1 + b1)
    lo, hi = stats.beta.ppf([0.025, 0.975], a1, b1)
    print(f"\nMLE = {mle:.3f} | MAP = {map_:.3f} | апостериорное среднее = {mean:.3f} | 95% интервал [{lo:.3f}, {hi:.3f}]")

    theta, post = grid_posterior(a0, b0, k, n)
    print("сетка == формула сопряжённости:", np.allclose(post, stats.beta.pdf(theta, a1, b1), atol=1e-6))

    # 3. С ростом данных prior «забывается»: MAP -> MLE
    rng = np.random.default_rng(0)
    true_p = 0.75
    for n_obs in (10, 100, 1000, 10_000):
        k_obs = int((rng.random(n_obs) < true_p).sum())
        a, b = beta_update(a0, b0, k_obs, n_obs - k_obs)
        print(f"n={n_obs:>6}: MLE {k_obs / n_obs:.3f}, MAP {(a - 1) / (a + b - 2):.3f}, ширина 95% интервала {np.diff(stats.beta.ppf([0.025, 0.975], a, b))[0]:.3f}")
