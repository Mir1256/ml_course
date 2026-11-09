import numpy as np
import pytest

from matrices import conv2d_im2col, matmul_loop, rotation, scaling, transform_bbox, translation


def test_matmul_shapes_mismatch():
    with pytest.raises(AssertionError):
        matmul_loop(np.ones((2, 3)), np.ones((2, 3)))


def test_matmul_not_commutative():
    A, B = np.array([[0, 1], [0, 0]]), np.array([[0, 0], [1, 0]])
    assert not np.allclose(matmul_loop(A, B), matmul_loop(B, A))


def test_rotation_is_orthogonal():
    R = rotation(0.7)[:2, :2]
    assert np.allclose(R.T @ R, np.eye(2))
    assert np.isclose(np.linalg.det(R), 1.0)


def test_bbox_identity_and_scale():
    box = (10, 20, 30, 60)
    assert np.allclose(transform_bbox(box, np.eye(3)), box)
    assert np.allclose(transform_bbox(box, scaling(2, 0.5)), (20, 10, 60, 30))
    assert np.allclose(transform_bbox(box, translation(5, -5)), (15, 15, 35, 55))


def test_conv_identity_kernel():
    img = np.arange(25.0).reshape(5, 5)
    ident = np.zeros((3, 3)); ident[1, 1] = 1
    assert np.allclose(conv2d_im2col(img, ident), img[1:-1, 1:-1])
