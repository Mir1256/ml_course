"""Сжатие изображения через SVD и PCA через SVD против sklearn."""
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits, load_sample_image
from sklearn.decomposition import PCA, TruncatedSVD

from svd import low_rank

# ---------- 1. Сжатие изображения ----------
gray = load_sample_image("china.jpg").mean(axis=2) / 255.0          # (427, 640)
m, n = gray.shape
s = np.linalg.svd(gray, compute_uv=False)
energy = np.cumsum(s**2) / np.sum(s**2)
print(f"изображение {m}x{n}, ранг {np.linalg.matrix_rank(gray)}")
for k in (5, 20, 50, 100):
    approx = low_rank(gray, k)
    mse = np.mean((gray - approx) ** 2)
    psnr = 10 * np.log10(1.0 / mse)
    stored = k * (m + n + 1) / (m * n)
    print(f"k={k:>3}: энергия {energy[k - 1]:.4f}, PSNR {psnr:5.2f} дБ, хранится {stored:.1%} чисел")

fig, axes = plt.subplots(1, 4, figsize=(14, 3))
for ax, k in zip(axes, (5, 20, 50, 100)):
    ax.imshow(np.clip(low_rank(gray, k), 0, 1), cmap="gray")
    ax.set_title(f"ранг {k}")
    ax.axis("off")
fig.tight_layout()
fig.savefig("svd_compression.png", dpi=80)

# ---------- 2. PCA = SVD центрированных данных ----------
X = load_digits().data                                              # (1797, 64)
Xc = X - X.mean(axis=0)
U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
k = 10
our_components = Vt[:k]                                             # главные направления — строки V^T
our_variance = sv[:k] ** 2 / (len(X) - 1)                           # дисперсии вдоль них
our_scores = U[:, :k] * sv[:k]                                      # координаты объектов = U Σ = Xc V

pca = PCA(n_components=k, svd_solver="full").fit(X)
signs = np.sign(np.sum(our_components * pca.components_, axis=1))   # SVD определено с точностью до знака
print("\nPCA: направления совпадают (с учётом знака):", np.allclose(our_components * signs[:, None], pca.components_, atol=1e-8))
print("PCA: дисперсии совпадают:", np.allclose(our_variance, pca.explained_variance_))
print("PCA: проекции совпадают:", np.allclose(our_scores * signs, pca.transform(X), atol=1e-8))
print("доля объяснённой дисперсии 10 компонент:", pca.explained_variance_ratio_.sum().round(3))

# ---------- 3. Полное SVD против усечённого (рандомизированного) ----------
rng = np.random.default_rng(0)
big = rng.normal(size=(3000, 50)) @ rng.normal(size=(50, 2000)) + 0.01 * rng.normal(size=(3000, 2000))
t0 = time.perf_counter(); s_full = np.linalg.svd(big, compute_uv=False); t_full = time.perf_counter() - t0
t0 = time.perf_counter(); tsvd = TruncatedSVD(n_components=20, random_state=0).fit(big); t_tr = time.perf_counter() - t0
print(f"\nполное SVD 3000x2000: {t_full:.2f} с | TruncatedSVD(k=20): {t_tr:.2f} с")
print("первые 5 σ, полное:   ", s_full[:5].round(2))
print("первые 5 σ, TruncSVD: ", tsvd.singular_values_[:5].round(2))
