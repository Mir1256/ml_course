import numpy as np
import pytest

from linsys import det, homography_from_4_points, inverse, rank, solve


@pytest.mark.parametrize("n", [1, 2, 3, 6])
def test_against_numpy(n):
    rng = np.random.default_rng(n)
    A, b = rng.normal(size=(n, n)), rng.normal(size=n)
    assert np.isclose(det(A), np.linalg.det(A))
    assert np.allclose(solve(A, b), np.linalg.solve(A, b))
    assert np.allclose(inverse(A), np.linalg.inv(A))


def test_det_properties():
    rng = np.random.default_rng(7)
    A, B = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    assert np.isclose(det(A @ B), det(A) * det(B))            # мультипликативность
    assert np.isclose(det(A.T), det(A))
    assert np.isclose(det(A[[1, 0, 2]]), -det(A))              # перестановка строк меняет знак


def test_pivoting_needed():
    A = np.array([[0.0, 1.0], [1.0, 1.0]])                     # без выбора главного элемента — деление на 0
    assert np.allclose(solve(A, np.array([1.0, 2.0])), [1.0, 1.0])


def test_rank():
    assert rank(np.eye(3)) == 3
    assert rank(np.array([[1, 2], [2, 4.0]])) == 1
    assert rank(np.zeros((2, 3))) == 0
    assert rank(np.random.default_rng(0).normal(size=(5, 3))) == 3


def test_homography_maps_points():
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    dst = np.array([[10, 10], [50, 12], [48, 60], [8, 55]], float)
    H = homography_from_4_points(src, dst)
    pts = np.c_[src, np.ones(4)] @ H.T
    assert np.allclose(pts[:, :2] / pts[:, 2:], dst)
