import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

from pseudo_label import self_training


def test_self_training_labels_everything_on_easy_data():
    X, y = make_blobs(300, centers=3, cluster_std=0.5, random_state=0)
    y_semi = np.full(300, -1)
    for c in range(3):
        y_semi[np.flatnonzero(y == c)[:3]] = c                   # по 3 метки на класс
    model, y_pseudo, hist = self_training(lambda: LogisticRegression(), X, y_semi, threshold=0.8)
    assert hist[0] == 9 and hist[-1] > 250
    assert np.mean(y_pseudo[y_pseudo != -1] == y[y_pseudo != -1]) > 0.98


def test_no_unlabeled_means_plain_supervised():
    X, y = make_blobs(60, centers=2, random_state=1)
    _, y_out, hist = self_training(lambda: LogisticRegression(), X, y)
    assert hist == [60] and (y_out == y).all()
