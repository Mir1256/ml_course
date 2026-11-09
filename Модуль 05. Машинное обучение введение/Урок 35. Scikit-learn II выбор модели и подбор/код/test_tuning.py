import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from tuning import my_grid_search


def test_my_grid_matches_sklearn():
    X, y = make_classification(300, 10, random_state=0)
    cv = StratifiedKFold(4, shuffle=True, random_state=0)
    grid = {"C": [0.001, 0.1, 10]}
    model, params, score = my_grid_search(LogisticRegression(max_iter=1000), grid, X, y, cv)
    gs = GridSearchCV(LogisticRegression(max_iter=1000), grid, cv=cv).fit(X, y)
    assert params == gs.best_params_ and np.isclose(score, gs.best_score_)
    assert np.allclose(model.coef_, gs.best_estimator_.coef_)          # refit на всех данных одинаковый


def test_random_search_hit_probability():
    # P(хотя бы один из n случайных запусков попал в лучшие 5% пространства) = 1 - 0.95^n
    n = 60
    assert 1 - 0.95**n > 0.95
