"""Сверка с SciPy/torchvision и эмпирическая проверка O-нотации."""
import time

import numpy as np
from scipy import ndimage

from discrete import box_iou, connected_components, nms

rng = np.random.default_rng(0)
mask = ndimage.binary_opening(rng.random((200, 200)) > 0.55)
ours, n_ours = connected_components(mask, 4)
ref, n_ref = ndimage.label(mask)                                   # по умолчанию 4-связность
print(f"компонент: свои {n_ours}, scipy {n_ref}, разбиение совпадает: {np.array_equal(ours > 0, ref > 0) and n_ours == n_ref}")

try:
    import torch
    from torchvision.ops import box_iou as tv_box_iou, nms as tv_nms

    xy = rng.uniform(0, 500, size=(300, 2))
    wh = rng.uniform(20, 80, size=(300, 2))
    boxes = np.c_[xy, xy + wh]
    scores = rng.random(300)
    keep_tv = tv_nms(torch.tensor(boxes), torch.tensor(scores), 0.5).tolist()
    print("NMS совпадает с torchvision:", nms(boxes, scores, 0.5) == keep_tv)
    print("IoU совпадает:", np.isclose(box_iou(boxes[0], boxes[1]), tv_box_iou(torch.tensor(boxes[:1]), torch.tensor(boxes[1:2])).item()))
except ImportError:
    print("torchvision не установлен — сверка пропущена")

# O(n^2) против O(n log n): поиск дубликатов хэшей кадров
for n in (1_000, 2_000, 4_000):
    h = rng.integers(0, 10**9, size=n)
    t0 = time.perf_counter()
    dup_pairs = sum(1 for i in range(n) for j in range(i + 1, n) if h[i] == h[j])
    t_quad = time.perf_counter() - t0
    t0 = time.perf_counter()
    s = np.sort(h)
    dup_sorted = int(np.sum(s[1:] == s[:-1]))
    t_sort = time.perf_counter() - t0
    print(f"n={n}: попарно {t_quad:.2f} с, через сортировку {t_sort * 1e3:.2f} мс")
