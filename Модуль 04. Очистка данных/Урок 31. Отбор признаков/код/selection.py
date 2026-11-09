"""Отбор признаков: filter (взаимная информация), wrapper (жадный прямой отбор), embedded (L1, важности) — с нуля и через sklearn."""
import numpy as np
from sklearn.model_selection import cross_val_score


def entropy_discrete(x: np.ndarray) -> float:
    _, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def mutual_information_discrete(x: np.ndarray, y: np.ndarray) -> float:
    """I(X;Y) = H(X) + H(Y) - H(X,Y) — сколько неопределённости о Y снимает знание X (урок 15)."""
    joint = np.unique(np.c_[x, y], axis=0, return_inverse=True)[1].ravel()
    return entropy_discrete(x) + entropy_discrete(y) - entropy_discrete(joint)


def forward_selection(model, X: np.ndarray, y: np.ndarray, k: int, cv: int = 5) -> list[int]:
    """Жадный прямой отбор: на каждом шаге добавляем признак, лучше всего улучшающий CV-качество.
    Полный перебор — 2^d моделей (урок 11), жадный — O(k·d)."""
    chosen: list[int] = []
    rest = list(range(X.shape[1]))
    for _ in range(k):
        scores = [cross_val_score(model, X[:, chosen + [j]], y, cv=cv).mean() for j in rest]
        best = rest[int(np.argmax(scores))]
        chosen.append(best)
        rest.remove(best)
    return chosen


def permutation_importance_np(model, X: np.ndarray, y: np.ndarray, n_repeats: int = 10, seed: int = 0) -> np.ndarray:
    """Падение качества при перемешивании столбца: связь признака с целью разрушена, распределение сохранено."""
    rng = np.random.default_rng(seed)
    base = model.score(X, y)
    drops = np.zeros((n_repeats, X.shape[1]))
    for r in range(n_repeats):
        for j in range(X.shape[1]):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops[r, j] = base - model.score(Xp, y)
    return drops.mean(axis=0)
