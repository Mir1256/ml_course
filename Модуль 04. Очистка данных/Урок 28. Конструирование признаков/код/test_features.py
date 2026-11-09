import numpy as np
from skimage.feature import hog as sk_hog

from hog import color_histogram, gradients, hog


def test_gradients_of_ramp():
    img = np.tile(np.arange(10.0), (5, 1))          # яркость растёт вправо на 1 за пиксель
    gx, gy = gradients(img)
    assert np.allclose(gx[:, 1:-1], 2) and np.allclose(gy, 0)


def test_hog_shape_matches_skimage():
    img = np.random.default_rng(0).random((32, 48))
    ours = hog(img)
    ref = sk_hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")
    assert ours.shape == ref.shape


def test_hog_brightness_invariance():
    img = np.random.default_rng(1).random((32, 32)) * 100
    assert np.allclose(hog(img), hog(img * 3 + 50), atol=1e-6)     # блочная нормировка убирает контраст и сдвиг яркости


def test_color_histogram_is_distribution():
    img = np.random.default_rng(2).integers(0, 256, (20, 30, 3)).astype(np.uint8)
    h = color_histogram(img, bins=4)
    assert h.shape == (64,) and np.isclose(h.sum(), 1)
    assert np.allclose(h, color_histogram(img[::-1, ::-1], bins=4))  # не зависит от расположения пикселей
