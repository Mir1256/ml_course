import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mini_sklearn import CyclicEncoder, MiniPipeline, group_split, stratified_split


def test_pipeline_uses_train_statistics():
    Xtr, Xte = np.array([[0.0], [2.0]]), np.array([[100.0]])
    scaler = StandardScaler()
    MiniPipeline([("s", scaler), ("m", LogisticRegression())]).fit(Xtr, np.array([0, 1]))
    assert scaler.mean_[0] == 1.0                       # test (100) не повлиял на статистики


def test_cyclic_encoder_is_sklearn_compatible():
    enc = CyclicEncoder(24)
    assert clone(enc).period == 24                      # clone работает -> параметры в __init__ сохранены правильно
    out = enc.fit_transform(np.array([[0], [6], [12]]))
    assert np.allclose(out, [[0, 1], [1, 0], [0, -1]], atol=1e-12)
    gs = GridSearchCV(make_pipeline(CyclicEncoder(), StandardScaler(), LogisticRegression()),
                      {"cyclicencoder__period": [12, 24]}, cv=2).fit(np.arange(40)[:, None] % 24, np.arange(40) % 2)
    assert gs.best_params_["cyclicencoder__period"] in (12, 24)


def test_stratified_split_preserves_proportions():
    y = np.array([0] * 90 + [1] * 10)
    tr, te = stratified_split(y, 0.2)
    assert len(te) == 20 and y[te].sum() == 2 and len(np.intersect1d(tr, te)) == 0


def test_group_split_no_overlap():
    groups = np.repeat(np.arange(10), 5)
    tr, te = group_split(groups, 0.3)
    assert set(groups[tr]).isdisjoint(groups[te]) and len(tr) + len(te) == 50
