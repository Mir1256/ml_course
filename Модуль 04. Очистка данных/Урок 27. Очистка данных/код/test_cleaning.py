import numpy as np
from PIL import Image

from image_cleaning import dhash, hamming_matrix, is_readable


def test_broken_file_detected(tmp_path):
    p = tmp_path / "a.png"
    Image.fromarray(np.zeros((50, 50, 3), np.uint8)).save(p)
    assert is_readable(p)[0]
    (tmp_path / "b.png").write_bytes(p.read_bytes()[:40])
    assert not is_readable(tmp_path / "b.png")[0]


def test_dhash_invariances():
    rng = np.random.default_rng(0)
    img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    img = np.asarray(Image.fromarray(img).resize((256, 256)))                 # плавная картинка
    H = np.stack([dhash(img), dhash(np.clip(img.astype(int) + 20, 0, 255).astype(np.uint8)),
                  dhash(np.asarray(Image.fromarray(img).resize((128, 128))))])
    D = hamming_matrix(H)
    assert D[0, 1] <= 4 and D[0, 2] <= 6            # яркость и масштаб почти не меняют хэш


def test_hamming_matrix_symmetric_zero_diag():
    H = np.random.default_rng(1).random((5, 64)) > 0.5
    D = hamming_matrix(H)
    assert (D == D.T).all() and (np.diag(D) == 0).all()
