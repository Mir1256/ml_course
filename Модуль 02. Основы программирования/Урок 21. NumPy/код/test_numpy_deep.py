import numpy as np
import pytest

from numpy_deep import iou_loops, iou_matrix


def test_iou_matrix_matches_loops():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 100, size=(30, 2))
    boxes = np.c_[xy, xy + rng.uniform(1, 30, size=(30, 2))]
    assert np.allclose(iou_matrix(boxes[:10], boxes[10:]), iou_loops(boxes[:10], boxes[10:]))


def test_iou_matrix_diag_is_one():
    b = np.array([[0, 0, 2, 2], [1, 1, 5, 4.0]])
    assert np.allclose(np.diag(iou_matrix(b, b)), 1)


def test_view_semantics():
    a = np.arange(6)
    v = a[::2]
    v[0] = 100
    assert a[0] == 100                       # срез — view: изменения видны в оригинале
    c = a[[0, 2]]
    c[0] = -1
    assert a[0] == 100                       # fancy indexing — копия


def test_add_at_vs_buffered():
    idx = np.array([0, 0, 0])
    a = np.zeros(1); a[idx] += 1
    b = np.zeros(1); np.add.at(b, idx, 1)
    assert a[0] == 1 and b[0] == 3


def test_broadcast_error():
    with pytest.raises(ValueError):
        np.ones((2, 3)) + np.ones(2)
    assert (np.ones((2, 3)) + np.ones((2, 1))).shape == (2, 3)
