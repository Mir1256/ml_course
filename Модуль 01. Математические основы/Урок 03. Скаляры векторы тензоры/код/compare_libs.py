"""Сверка своих реализаций с NumPy/SciPy/scikit-learn и (если установлен) PyTorch."""
import numpy as np
from scipy.spatial.distance import cosine as scipy_cosine_distance
from sklearn.metrics.pairwise import cosine_similarity

from vectors import cosine, dot_loop, hwc_uint8_to_chw_float, norm, pairwise_cosine

rng = np.random.default_rng(42)
u, v = rng.normal(size=128), rng.normal(size=128)

print("dot:     ", np.isclose(dot_loop(u, v), np.dot(u, v)))
print("L1/L2/inf:", all(np.isclose(norm(u, p), np.linalg.norm(u, p)) for p in (1, 2, np.inf)))
print("cosine:  ", np.isclose(cosine(u, v), 1 - scipy_cosine_distance(u, v)))   # SciPy возвращает РАССТОЯНИЕ 1 - cos

A, B = rng.normal(size=(500, 64)), rng.normal(size=(300, 64))
print("pairwise:", np.allclose(pairwise_cosine(A, B), cosine_similarity(A, B)))

try:
    import torch

    img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255     # то же, что hwc_uint8_to_chw_float
    print("torch CHW:", np.allclose(t.numpy(), hwc_uint8_to_chw_float(img)))
    print("torch cos:", np.isclose(torch.nn.functional.cosine_similarity(
        torch.from_numpy(u), torch.from_numpy(v), dim=0).item(), cosine(u, v)))
except ImportError:
    print("torch не установлен — сверка с PyTorch пропущена")
