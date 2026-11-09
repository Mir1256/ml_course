"""PCA-аугментация цвета из AlexNet (Krizhevsky et al., 2012) через собственные векторы ковариации RGB."""
import numpy as np
from sklearn.datasets import load_sample_image

img = load_sample_image("china.jpg").astype(np.float64) / 255.0        # (427, 640, 3)
pixels = img.reshape(-1, 3)                                             # каждая строка — пиксель (R, G, B)

mean = pixels.mean(axis=0)
cov = np.cov(pixels - mean, rowvar=False)                               # (3, 3), симметричная PSD
eigvals, eigvecs = np.linalg.eigh(cov)                                  # по возрастанию

print("ковариация RGB:\n", cov.round(5))
print("собственные значения:", eigvals.round(5))
print("доля дисперсии вдоль главной оси:", round(eigvals[-1] / eigvals.sum(), 3))
print("главная ось (R, G, B):", eigvecs[:, -1].round(3), "<- примерно 'яркость': все компоненты одного знака")

# Проверка: дисперсия проекций пикселей на собственный вектор = собственное значение
proj = (pixels - mean) @ eigvecs
print("дисперсии проекций == собственные значения:", np.allclose(proj.var(axis=0, ddof=1), eigvals))


def fancy_pca(image: np.ndarray, alpha_std: float = 0.1, rng=np.random.default_rng(0)) -> np.ndarray:
    """Сдвигаем все пиксели на sum_i alpha_i * sqrt(λ_i) * v_i, alpha_i ~ N(0, alpha_std^2).

    Меняется освещение/цвет вдоль естественных для этой картинки направлений, а не вдоль осей R, G, B.
    """
    delta = eigvecs @ (rng.normal(0, alpha_std, size=3) * np.sqrt(eigvals))
    return np.clip(image + delta, 0, 1)


aug = fancy_pca(img)
print("средний цвет до / после аугментации:", img.reshape(-1, 3).mean(0).round(3), aug.reshape(-1, 3).mean(0).round(3))
