"""Векторы и тензоры с нуля на NumPy: скалярное произведение, нормы, косинус, проекция, NCHW."""
import numpy as np


# ---------- Векторы: «учебные» версии через цикл и векторизованные ----------
def dot_loop(u: np.ndarray, v: np.ndarray) -> float:
    assert u.shape == v.shape and u.ndim == 1
    s = 0.0
    for a, b in zip(u, v):
        s += a * b
    return s


def norm(v: np.ndarray, p: float = 2) -> float:
    """L_p-норма: (sum |v_i|^p)^(1/p); p=inf — максимум модуля."""
    if p == np.inf:
        return float(np.max(np.abs(v)))
    return float(np.sum(np.abs(v) ** p) ** (1 / p))


def cosine(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    """cos угла между векторами; eps защищает от деления на ноль у нулевого вектора."""
    return float(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), eps))


def project(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Ортогональная проекция v на прямую, натянутую на u: (u·v / u·u) u."""
    return (u @ v) / (u @ u) * u


def pairwise_cosine(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Матрица косинусов (n, m) между строками A (n, d) и B (m, d) — одно матричное умножение."""
    A_n = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), eps)   # keepdims -> (n, 1), иначе broadcasting сломается
    B_n = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), eps)
    return A_n @ B_n.T


# ---------- Тензоры: изображение и батч ----------
def hwc_uint8_to_chw_float(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 [0..255] -> (3, H, W) float32 [0..1] — как делает torchvision ToTensor."""
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
    return img.transpose(2, 0, 1).astype(np.float32) / 255.0


def make_batch(images: list[np.ndarray]) -> np.ndarray:
    """Список (3, H, W) -> батч (N, 3, H, W): формат NCHW, который ждут PyTorch и YOLO."""
    return np.stack(images, axis=0)


def channel_mean_std(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Поканальные mean/std по батчу NCHW: усредняем по осям N, H, W -> вектор длины C."""
    return batch.mean(axis=(0, 2, 3)), batch.std(axis=(0, 2, 3))


if __name__ == "__main__":
    u, v = np.array([1.0, 2.0, 2.0]), np.array([2.0, 1.0, -1.0])
    print("u·v =", dot_loop(u, v), "| ||u||_2 =", norm(u), "| ||u||_1 =", norm(u, 1), "| ||u||_inf =", norm(u, np.inf))
    print("cos(u, v) =", round(cosine(u, v), 4), "| proj_u v =", project(v, u))

    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 256, size=(64, 48, 3), dtype=np.uint8) for _ in range(8)]
    batch = make_batch([hwc_uint8_to_chw_float(im) for im in imgs])
    mean, std = channel_mean_std(batch)
    print("batch shape (N, C, H, W) =", batch.shape, "| dtype", batch.dtype, "| байт:", batch.nbytes)
    print("mean по каналам:", mean.round(3), "| std:", std.round(3))
    normalized = (batch - mean[None, :, None, None]) / std[None, :, None, None]   # broadcasting (1,C,1,1)
    print("после нормализации mean ≈", normalized.mean(axis=(0, 2, 3)).round(6))
