"""Снижение размерности: проклятие размерности, PCA как инструмент, случайные проекции (Джонсон–Линденштраусс), t-SNE и UMAP."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import GaussianRandomProjection, johnson_lindenstrauss_min_dim

rng = np.random.default_rng(0)

# ---------- 1. Проклятие размерности: расстояния «концентрируются» ----------
print("отношение (самое далёкое − самое близкое) / самое близкое для 500 случайных точек в кубе:")
for d in (2, 10, 100, 1000):
    X = rng.random((500, d))
    dist = np.linalg.norm(X[1:] - X[0], axis=1)
    print(f"  d = {d:>4}: {(dist.max() - dist.min()) / dist.min():.2f}")

# ---------- 2. PCA: сколько компонент оставить ----------
X, y = load_digits(return_X_y=True)
pca = PCA().fit(X)
cum = np.cumsum(pca.explained_variance_ratio_)
for share in (0.8, 0.9, 0.95, 0.99):
    print(f"PCA: для {share:.0%} дисперсии нужно {np.searchsorted(cum, share) + 1} компонент из 64")
for k in (5, 10, 20, 40):
    Z = PCA(k).fit(X)
    rec_err = np.mean((X - Z.inverse_transform(Z.transform(X))) ** 2)
    acc = cross_val_score(KNeighborsClassifier(3), Z.transform(X), y, cv=5).mean()
    print(f"  k = {k:>2}: ошибка реконструкции {rec_err:6.2f}, KNN accuracy {acc:.3f}")
print(f"  все 64: KNN accuracy {cross_val_score(KNeighborsClassifier(3), X, y, cv=5).mean():.3f}")

# ---------- 3. Случайные проекции: расстояния почти сохраняются при k ~ log(n) / ε² ----------
E = rng.normal(size=(1000, 10_000))                   # 1000 «эмбеддингов» размерности 10 000
print(f"\nДжонсон–Линденштраусс: для n=1000, ε=0.2 нужно k ≥ {johnson_lindenstrauss_min_dim(1000, eps=0.2)}")
i, j = rng.integers(0, 1000, (2, 3000))
mask = i != j
orig = np.linalg.norm(E[i[mask]] - E[j[mask]], axis=1)
for k in (50, 500, 1500):
    P = GaussianRandomProjection(k, random_state=0).fit_transform(E)
    ratio = np.linalg.norm(P[i[mask]] - P[j[mask]], axis=1) / orig
    print(f"  k = {k:>4}: отношение расстояний после/до в [{ratio.min():.3f}, {ratio.max():.3f}]")

# ---------- 4. Визуализация: PCA против t-SNE против UMAP ----------
emb = {"PCA": PCA(2).fit_transform(X),
       "t-SNE": TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(X)}
try:
    import umap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb["UMAP"] = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X)
except ImportError:
    print("umap-learn не установлен — UMAP пропущен")

print("\n2D-вложение | trustworthiness (сохранение соседей) | KNN accuracy в 2D")
fig, axes = plt.subplots(1, len(emb), figsize=(5 * len(emb), 4.5))
for ax, (name, Z) in zip(axes, emb.items()):
    tw = trustworthiness(X, Z, n_neighbors=10)
    acc = cross_val_score(KNeighborsClassifier(5), Z, y, cv=5).mean()
    print(f"  {name:<6} {tw:.3f}  {acc:.3f}")
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=4)
    ax.set_title(f"{name}: trustworthiness {tw:.2f}")
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(sc, ax=axes, ticks=range(10), label="цифра")
fig.savefig("embeddings.png", dpi=80, bbox_inches="tight"); plt.close(fig)
