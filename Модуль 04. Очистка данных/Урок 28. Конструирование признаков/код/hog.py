"""HOG (гистограмма ориентированных градиентов, Dalal & Triggs 2005) с нуля и цветовые гистограммы."""
import numpy as np


def gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Центральные разности [-1, 0, 1] (урок 8); на границах — 0, как в skimage."""
    gx = np.zeros_like(img, dtype=float)
    gy = np.zeros_like(img, dtype=float)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    return gx, gy


def hog(img: np.ndarray, orientations: int = 9, cell: int = 8, block: int = 2, eps: float = 1e-5) -> np.ndarray:
    """1) градиенты; 2) в каждой клетке cell×cell — гистограмма направлений (0..180°), взвешенная модулем;
    3) клетки группируются в блоки block×block, вектор блока нормируется (L2-Hys) — устойчивость к освещению."""
    img = img.astype(float)
    gx, gy = gradients(img)
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180                     # «беззнаковые» направления: край светлое→тёмное = тёмное→светлое
    n_cy, n_cx = img.shape[0] // cell, img.shape[1] // cell
    bins = np.minimum((ang / (180 / orientations)).astype(int), orientations - 1)

    # гистограммы клеток без циклов: индекс (клетка_y, клетка_x, бин) -> bincount с весами-модулями
    ys, xs = np.mgrid[: n_cy * cell, : n_cx * cell]
    idx = ((ys // cell) * n_cx + xs // cell) * orientations + bins[: n_cy * cell, : n_cx * cell]
    hist = np.bincount(idx.ravel(), weights=mag[: n_cy * cell, : n_cx * cell].ravel(),
                       minlength=n_cy * n_cx * orientations).reshape(n_cy, n_cx, orientations) / cell**2

    blocks = []
    for by in range(n_cy - block + 1):
        for bx in range(n_cx - block + 1):
            v = hist[by:by + block, bx:bx + block].ravel()
            v = v / np.sqrt(np.sum(v**2) + eps**2)
            v = np.minimum(v, 0.2)                                  # Hys: обрезаем доминирующие градиенты...
            blocks.append(v / np.sqrt(np.sum(v**2) + eps**2))      # ...и нормируем снова
    return np.concatenate(blocks)


def color_histogram(img_rgb: np.ndarray, bins: int = 8) -> np.ndarray:
    """Совместная 3D-гистограмма RGB (bins^3 значений), нормированная в распределение: глобальный «цветовой отпечаток»."""
    q = (img_rgb.astype(int) * bins // 256).reshape(-1, 3)
    idx = (q[:, 0] * bins + q[:, 1]) * bins + q[:, 2]
    h = np.bincount(idx, minlength=bins**3).astype(float)
    return h / h.sum()
