"""Случайные величины с нуля: PMF/PDF/CDF, обратное преобразование, матожидание, дисперсия, ковариация."""
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)


def ecdf(samples: np.ndarray):
    """Эмпирическая функция распределения: F(x) = доля наблюдений <= x."""
    xs = np.sort(samples)
    return xs, np.arange(1, len(xs) + 1) / len(xs)


def inverse_transform_exponential(u: np.ndarray, rate: float) -> np.ndarray:
    """Если U ~ Uniform(0, 1), то F^{-1}(U) имеет распределение с CDF F.
    Для экспоненциального F(x) = 1 - e^{-λx}  ->  F^{-1}(u) = -ln(1 - u) / λ."""
    return -np.log1p(-u) / rate


if __name__ == "__main__":
    # 1. Дискретная величина: число объектов на кадре. PMF, матожидание через определение и через LOTUS
    values = np.array([0, 1, 2, 3, 4])
    pmf = np.array([0.1, 0.3, 0.3, 0.2, 0.1])
    mean = np.sum(values * pmf)
    var = np.sum(values**2 * pmf) - mean**2                        # Var = E[X²] - (E[X])²
    sample = rng.choice(values, size=100_000, p=pmf)
    print(f"объектов на кадре: E[X] = {mean:.3f} (выборка {sample.mean():.3f}), Var = {var:.3f} (выборка {sample.var():.3f})")
    print(f"E[X²] через LOTUS = {np.sum(values**2 * pmf):.3f} | E[X]² = {mean**2:.3f}  <- E[g(X)] ≠ g(E[X])")

    # 2. Плотность — не вероятность: может быть > 1
    u = stats.uniform(0, 0.25)
    print(f"\nUniform(0, 0.25): плотность {u.pdf(0.1):.1f} > 1, но P(X <= 0.1) = {u.cdf(0.1):.2f}")

    # 3. Метод обратного преобразования
    times = inverse_transform_exponential(rng.random(100_000), rate=2.0)
    xs, F = ecdf(times)
    max_gap = np.max(np.abs(F - stats.expon(scale=1 / 2.0).cdf(xs)))
    print(f"экспоненциальные из равномерных: среднее {times.mean():.4f} (теория 0.5), max|F_эмп - F| = {max_gap:.4f}")

    # 4. Дисперсия среднего: Var(X̄) = σ²/n -> отсюда 1/sqrt(n) из урока 12
    for n in (10, 100, 1000):
        means = rng.normal(5, 2, size=(5_000, n)).mean(axis=1)
        print(f"n={n:>4}: std(среднего) = {means.std():.4f}, теория σ/sqrt(n) = {2 / np.sqrt(n):.4f}")

    # 5. Ковариация линейного преобразования: Cov(AX) = A Σ A^T
    Sigma = np.array([[4.0, 1.5], [1.5, 1.0]])
    X = rng.multivariate_normal([0, 0], Sigma, size=200_000)
    A = np.array([[1.0, 1.0], [1.0, -1.0], [2.0, 0.0]])
    print("\nCov(AX) выборочная:\n", np.cov(X @ A.T, rowvar=False).round(2))
    print("A Σ A^T:\n", (A @ Sigma @ A.T).round(2))

    # 6. Некоррелированы, но зависимы: X ~ N(0,1), Y = X²
    x = rng.normal(size=200_000)
    y = x**2
    print(f"\ncorr(X, X²) = {np.corrcoef(x, y)[0, 1]:.4f}, но Y полностью определяется X")

    # 7. Неравенство Йенсена: для выпуклой f E[f(X)] >= f(E[X]); для вогнутого log — наоборот
    z = rng.uniform(0.5, 2.0, size=100_000)
    print(f"E[log Z] = {np.log(z).mean():.4f} <= log E[Z] = {np.log(z.mean()):.4f}")
