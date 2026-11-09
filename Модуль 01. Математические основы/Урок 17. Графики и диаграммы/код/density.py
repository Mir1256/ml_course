"""Гистограмма и KDE как оценки плотности: правило Фридмана–Диакониса и ядерная оценка с нуля."""
import numpy as np
from scipy import stats


def fd_bin_width(x: np.ndarray) -> float:
    """Фридман–Диаконис: h = 2 IQR / n^(1/3) — робастный выбор ширины столбца."""
    q75, q25 = np.percentile(x, [75, 25])
    return 2 * (q75 - q25) / len(x) ** (1 / 3)


def histogram_density(x: np.ndarray, width: float):
    """Высота столбца = доля точек в нём / ширина -> площадь под гистограммой равна 1."""
    edges = np.arange(x.min(), x.max() + width, width)
    counts, _ = np.histogram(x, bins=edges)
    return edges, counts / (len(x) * width)


def kde(x: np.ndarray, grid: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    """f(t) = 1/(n h) Σ K((t - x_i)/h), K — стандартная нормальная плотность.
    По умолчанию h по правилу Скотта: σ n^(-1/5) (так же, как в scipy.stats.gaussian_kde)."""
    h = bandwidth if bandwidth is not None else x.std(ddof=1) * len(x) ** (-1 / 5)
    u = (grid[:, None] - x[None, :]) / h                  # (len(grid), n) — все пары «точка сетки × наблюдение»
    return np.exp(-0.5 * u**2).sum(axis=1) / (len(x) * h * np.sqrt(2 * np.pi))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.log(rng.lognormal(8, 1.2, size=2000))           # log-площади рамок
    h = fd_bin_width(x)
    edges, dens = histogram_density(x, h)
    print(f"ширина столбца FD: своя {h:.4f} | numpy 'fd': {np.diff(np.histogram_bin_edges(x, 'fd'))[0]:.4f}")
    print(f"площадь под гистограммой: {np.sum(dens * h):.4f}")

    grid = np.linspace(x.min() - 1, x.max() + 1, 400)
    ours = kde(x, grid)
    ref = stats.gaussian_kde(x)(grid)
    print("KDE совпадает со scipy.stats.gaussian_kde:", np.allclose(ours, ref))
    print(f"площадь под KDE: {np.trapezoid(ours, grid):.4f}")
