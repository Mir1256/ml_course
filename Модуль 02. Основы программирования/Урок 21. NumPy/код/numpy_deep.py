"""NumPy глубоко: память и strides, view против copy, broadcasting, fancy indexing, einsum, окна без циклов."""
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

rng = np.random.default_rng(0)

# ---------- 1. Память: ndarray = буфер + dtype + shape + strides ----------
img = np.zeros((480, 640, 3), dtype=np.uint8)                      # HWC
print("HWC strides:", img.strides, "<- шаг в байтах вдоль каждой оси")
chw = img.transpose(2, 0, 1)
print("после transpose -> CHW: shape", chw.shape, "strides", chw.strides, "| C-contiguous:", chw.flags.c_contiguous)
print("transpose — view (общая память):", np.shares_memory(img, chw))
print("ascontiguousarray — копия:", not np.shares_memory(img, np.ascontiguousarray(chw)))

# ---------- 2. View или copy ----------
a = np.arange(12).reshape(3, 4)
checks = {
    "a[1:, ::2] (срез)": a[1:, ::2],
    "a.T (транспонирование)": a.T,
    "a.reshape(4, 3)": a.reshape(4, 3),
    "a[[0, 2]] (fancy)": a[[0, 2]],
    "a[a > 5] (маска)": a[a > 5],
    "a.T.reshape(-1) (не-contiguous)": a.T.reshape(-1),
}
for name, arr in checks.items():
    print(f"{name:<32} view: {np.shares_memory(a, arr)}")

# ---------- 3. Broadcasting: выравнивание форм справа ----------
batch = rng.random((8, 3, 32, 32)).astype(np.float32)               # NCHW
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)           # ImageNet-статистики
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
normed = (batch - mean[:, None, None]) / std[:, None, None]        # (3,1,1) растягивается на (8,3,32,32)
print("\nнормализация NCHW:", normed.shape)
try:
    batch - mean                                                   # (8,3,32,32) против (3,): 32 != 3
except ValueError as e:
    print("ошибка broadcasting:", str(e)[:70])

# ---------- 4. Векторизация: попарные IoU без циклов ----------
def iou_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(n,4) x (m,4) -> (n,m). A[:, None] имеет форму (n,1,4), B[None] — (1,m,4)."""
    lt = np.maximum(A[:, None, :2], B[None, :, :2])
    rb = np.minimum(A[:, None, 2:], B[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area(A)[:, None] + area(B)[None, :] - inter)


def iou_loops(A, B):
    out = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = iw * ih
            out[i, j] = inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return out


xy = rng.uniform(0, 500, size=(800, 2)); wh = rng.uniform(10, 100, size=(800, 2))
boxes = np.c_[xy, xy + wh]
t0 = time.perf_counter(); M1 = iou_loops(boxes[:300], boxes[300:]); t_loop = time.perf_counter() - t0
t0 = time.perf_counter(); M2 = iou_matrix(boxes[:300], boxes[300:]); t_vec = time.perf_counter() - t0
print(f"\nIoU 300x500: циклы {t_loop:.2f} с, векторизация {t_vec * 1e3:.1f} мс, ускорение x{t_loop / t_vec:.0f}, совпадают: {np.allclose(M1, M2)}")

# ---------- 5. Fancy indexing ----------
logits = rng.normal(size=(5, 4))
labels = np.array([0, 3, 1, 1, 2])
print("\nлогит истинного класса для каждого объекта:", logits[np.arange(5), labels].round(2))
print("one-hot через np.eye:\n", np.eye(4, dtype=int)[labels])

counts_bad = np.zeros(4, int)
counts_bad[labels] += 1                                            # повторяющиеся индексы: += применяется ОДИН раз
counts_ok = np.zeros(4, int)
np.add.at(counts_ok, labels, 1)                                    # небуферизованное накопление
print("подсчёт классов: a[idx] += 1 ->", counts_bad, "| np.add.at ->", counts_ok, "| bincount ->", np.bincount(labels, minlength=4))

# ---------- 6. einsum: индексная запись операций ----------
Q, K = rng.normal(size=(2, 6, 8)), rng.normal(size=(2, 6, 8))      # (batch, токены, размерность)
scores = np.einsum("bqd,bkd->bqk", Q, K)                           # батчевое Q K^T — attention, урок 83
print("\neinsum attention scores:", scores.shape, "== Q @ K^T:", np.allclose(scores, Q @ K.transpose(0, 2, 1)))
A = rng.normal(size=(4, 4))
print("след:", np.isclose(np.einsum("ii->", A), np.trace(A)), "| внешнее произведение:", np.einsum("i,j->ij", [1, 2], [3, 4]).tolist())

# ---------- 7. Окна без циклов: max-pooling 2x2 ----------
x = rng.integers(0, 10, size=(4, 6))
pooled_reshape = x.reshape(2, 2, 3, 2).max(axis=(1, 3))            # неперекрывающиеся окна — через reshape
windows = sliding_window_view(x, (2, 2))[::2, ::2]                 # (2, 3, 2, 2) — view, без копирования
print("\nmax-pool 2x2 через reshape и через sliding_window_view совпадают:", np.array_equal(pooled_reshape, windows.max(axis=(2, 3))))

# ---------- 8. dtype-ловушки ----------
pix = np.array([250, 10], dtype=np.uint8)
print("\nuint8: 250 + 10 =", (pix + np.uint8(10))[0], "(переполнение по модулю 256) | через int16:", (pix.astype(np.int16) + 10)[0])
big = np.full(10_000_000, 0.1, dtype=np.float32)
print(f"сумма 10^7 раз 0.1 в float32: np.sum {big.sum():.1f} (попарное суммирование), cumsum[-1] {big.cumsum()[-1]:.1f} (последовательное)")
