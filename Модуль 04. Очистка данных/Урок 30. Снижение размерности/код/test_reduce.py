import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection


def test_pca_full_rank_reconstructs_exactly():
    X = np.random.default_rng(0).normal(size=(50, 6))
    p = PCA(6).fit(X)
    assert np.allclose(p.inverse_transform(p.transform(X)), X)


def test_pca_error_equals_dropped_variance():
    X = np.random.default_rng(1).normal(size=(200, 8)) @ np.diag([5, 4, 3, 2, 1, 0.5, 0.2, 0.1])
    p_full = PCA().fit(X)
    p = PCA(3).fit(X)
    err = np.sum((X - p.inverse_transform(p.transform(X))) ** 2) / (len(X) - 1)
    assert np.isclose(err, p_full.explained_variance_[3:].sum())     # Эккарт–Юнг (урок 7)


def test_random_projection_preserves_norms_on_average():
    X = np.random.default_rng(2).normal(size=(300, 5000))
    P = GaussianRandomProjection(800, random_state=0).fit_transform(X)
    ratio = np.linalg.norm(P, axis=1) / np.linalg.norm(X, axis=1)
    assert abs(ratio.mean() - 1) < 0.02 and ratio.std() < 0.05
