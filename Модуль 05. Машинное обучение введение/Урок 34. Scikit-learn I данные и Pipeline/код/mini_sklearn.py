"""Как устроен scikit-learn изнутри: API оценщиков, свой Pipeline, стратифицированный и групповой сплит."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MiniPipeline:
    """Цепочка шагов: fit проходит по трансформерам (fit_transform) и обучает последний шаг на результате.
    Ключевое свойство: на новых данных вызывается только transform — статистики берутся из train."""

    def __init__(self, steps: list[tuple[str, object]]):
        self.steps = steps

    def fit(self, X, y=None):
        for _, step in self.steps[:-1]:
            X = step.fit_transform(X, y)
        self.steps[-1][1].fit(X, y)
        return self

    def _transform(self, X):
        for _, step in self.steps[:-1]:
            X = step.transform(X)
        return X

    def predict(self, X):
        return self.steps[-1][1].predict(self._transform(X))

    def score(self, X, y):
        return self.steps[-1][1].score(self._transform(X), y)


class CyclicEncoder(TransformerMixin, BaseEstimator):
    """Свой трансформер в стиле sklearn: столбец «час» -> (sin, cos) (урок 28). Обучаемых параметров нет, но fit обязателен."""

    def __init__(self, period: float = 24.0):
        self.period = period                       # параметры только сохраняются в __init__ — правило sklearn для clone/GridSearch

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        a = 2 * np.pi * np.asarray(X, float) / self.period
        return np.hstack([np.sin(a), np.cos(a)])

    def get_feature_names_out(self, input_features=None):
        names = input_features if input_features is not None else [f"x{i}" for i in range(self.n_features_in_)]
        return np.array([f"{n}_sin" for n in names] + [f"{n}_cos" for n in names])


def stratified_split(y: np.ndarray, test_size: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Внутри каждого класса берём одинаковую долю в test — доли классов в train и test совпадают."""
    rng = np.random.default_rng(seed)
    test = []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        test.extend(idx[: int(round(test_size * len(idx)))])
    test = np.sort(np.array(test))
    return np.setdiff1d(np.arange(len(y)), test), test


def group_split(groups: np.ndarray, test_size: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Целые группы (видео, сцены, пациенты) уходят либо в train, либо в test — без почти-копий по разные стороны."""
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(groups))
    test_groups = uniq[: int(round(test_size * len(uniq)))]
    is_test = np.isin(groups, test_groups)
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)
