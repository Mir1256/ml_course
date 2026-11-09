"""Подбор гиперпараметров: свой grid search, grid против random, вложенная CV, successive halving, сохранение модели."""
import itertools
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.experimental import enable_halving_search_cv  # noqa: F401 — включает HalvingRandomSearchCV
from sklearn.model_selection import (GridSearchCV, HalvingRandomSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def my_grid_search(estimator, grid: dict, X, y, cv):
    """Полный перебор: для каждой комбинации — средний CV-score; лучшая переобучается на всех данных (refit)."""
    results = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        model = clone(estimator).set_params(**params)
        results.append((cross_val_score(model, X, y, cv=cv).mean(), params))
    best_score, best_params = max(results, key=lambda r: r[0])
    return clone(estimator).set_params(**best_params).fit(X, y), best_params, best_score


if __name__ == "__main__":
    X, y = load_digits(return_X_y=True)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pipe = make_pipeline(StandardScaler(), SVC())
    grid = {"svc__C": [0.1, 1, 10], "svc__gamma": [1e-3, 1e-2, 1e-1]}

    _, p_ours, s_ours = my_grid_search(pipe, grid, X, y, cv)
    gs = GridSearchCV(pipe, grid, cv=cv).fit(X, y)
    print(f"свой grid: {p_ours} {s_ours:.4f} | GridSearchCV: {gs.best_params_} {gs.best_score_:.4f}")

    # Grid против random при одинаковом бюджете 9 запусков
    rs = RandomizedSearchCV(pipe, {"svc__C": loguniform(1e-2, 1e2), "svc__gamma": loguniform(1e-4, 1)},
                            n_iter=9, cv=cv, random_state=0).fit(X, y)
    print(f"random 9 запусков: C={rs.best_params_['svc__C']:.3g}, gamma={rs.best_params_['svc__gamma']:.3g}, score {rs.best_score_:.4f}"
          f" | различных значений gamma: grid 3, random {len(set(np.round(rs.cv_results_['param_svc__gamma'].astype(float), 8)))}")

    # Successive halving: плохие кандидаты отсеиваются на малых выборках
    t0 = time.perf_counter()
    big_rs = RandomizedSearchCV(pipe, {"svc__C": loguniform(1e-2, 1e2), "svc__gamma": loguniform(1e-4, 1)},
                                n_iter=60, cv=cv, random_state=0).fit(X, y)
    t_rs = time.perf_counter() - t0
    t0 = time.perf_counter()
    hs = HalvingRandomSearchCV(pipe, {"svc__C": loguniform(1e-2, 1e2), "svc__gamma": loguniform(1e-4, 1)},
                               n_candidates=60, factor=3, cv=cv, random_state=0).fit(X, y)
    t_hs = time.perf_counter() - t0
    print(f"60 кандидатов: random search {t_rs:.1f} с, score {big_rs.best_score_:.4f} | halving {t_hs:.1f} с, score {hs.best_score_:.4f}")

    # Вложенная CV: оценка лучшей модели по той же CV, на которой выбирали, — оптимистична
    Xb, yb = load_breast_cancer(return_X_y=True)
    Xb, yb = Xb[:150], yb[:150]                                     # маленькая выборка — эффект заметнее
    inner, outer = StratifiedKFold(3, shuffle=True, random_state=1), StratifiedKFold(5, shuffle=True, random_state=2)
    space = {"svc__C": np.logspace(-3, 3, 13), "svc__gamma": np.logspace(-4, 1, 11)}
    search = GridSearchCV(pipe, space, cv=inner)
    non_nested = search.fit(Xb, yb).best_score_
    nested = cross_val_score(search, Xb, yb, cv=outer).mean()
    print(f"\nоценка лучшей из {13 * 11} моделей: невложенная {non_nested:.4f} | вложенная (честная) {nested:.4f}")

    # Предсказание, вероятности, сохранение
    final = gs.best_estimator_
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "digits_svc.joblib"
        joblib.dump(final, path)                                    # сохраняется ВЕСЬ пайплайн: скейлер + модель
        loaded = joblib.load(path)                                  # joblib = pickle: загружать только свои файлы (урок 26)
        print(f"\nсохранено {path.stat().st_size / 1e3:.0f} КБ, предсказания после загрузки совпадают: {np.array_equal(loaded.predict(X[:50]), final.predict(X[:50]))}")
    print("decision_function первого объекта:", final.decision_function(X[:1]).round(2))
