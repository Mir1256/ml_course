"""Дискретная математика в CV: множества -> IoU, графы -> компоненты связности, жадный NMS и O-нотация."""
from collections import deque
from itertools import product
from math import comb

import numpy as np


# ---------- Множества: IoU = мера Жаккара ----------
def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """|A ∩ B| / |A ∪ B| для булевых масок."""
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def box_iou(b1, b2) -> float:
    """IoU рамок (x1, y1, x2, y2): пересечение двух прямоугольников — тоже прямоугольник."""
    iw = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    ih = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    inter = iw * ih
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area(b1) + area(b2) - inter)            # формула включений-исключений: |A∪B| = |A|+|B|-|A∩B|


# ---------- Графы: компоненты связности бинарной маски ----------
def connected_components(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    """Обход в ширину (BFS): пиксели — вершины, соседние пиксели маски — рёбра. O(H*W)."""
    labels = np.zeros(mask.shape, dtype=int)
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    current = 0
    H, W = mask.shape
    for i, j in product(range(H), range(W)):
        if mask[i, j] and labels[i, j] == 0:
            current += 1
            labels[i, j] = current
            queue = deque([(i, j)])
            while queue:
                y, x = queue.popleft()
                for dy, dx in steps:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current
                        queue.append((ny, nx))
    return labels, current


# ---------- Жадный NMS: O(n^2) сравнений в худшем случае ----------
def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> list[int]:
    order = list(np.argsort(scores)[::-1])                   # сортировка O(n log n)
    keep = []
    while order:
        best = order.pop(0)
        keep.append(int(best))
        order = [i for i in order if box_iou(boxes[best], boxes[i]) < iou_thr]
    return keep


if __name__ == "__main__":
    a = np.zeros((10, 10), bool); a[2:7, 2:7] = True
    b = np.zeros((10, 10), bool); b[4:9, 4:9] = True
    print(f"IoU масок = {mask_iou(a, b):.4f} (9 / 41)")
    print(f"IoU рамок = {box_iou((2, 2, 7, 7), (4, 4, 9, 9)):.4f}")

    m = np.zeros((6, 8), bool)
    m[0:2, 0:2] = True; m[2, 2] = True; m[4:6, 5:8] = True
    print("компонент при 4-связности:", connected_components(m, 4)[1], "| при 8-связности:", connected_components(m, 8)[1])

    boxes = np.array([[10, 10, 50, 50], [12, 12, 52, 52], [100, 100, 140, 140], [11, 9, 49, 51]], float)
    scores = np.array([0.9, 0.8, 0.7, 0.95])
    print("NMS оставил индексы:", nms(boxes, scores))

    # Комбинаторика: почему полный перебор признаков и гиперпараметров невозможен
    print("подмножеств из 30 признаков:", 2**30, "| выбрать 5 из 30:", comb(30, 5))
    grid = {"lr": 6, "batch": 4, "weight_decay": 5, "augment": 3, "optimizer": 3}
    print("размер сетки гиперпараметров:", np.prod(list(grid.values())), "запусков")
