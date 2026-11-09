"""Матрицы с нуля: умножение, линейный слой, аффинные преобразования bbox, свёртка через im2col."""
import numpy as np


def matmul_loop(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """C[i, j] = sum_k A[i, k] * B[k, j] — определение матричного произведения, O(n*m*p)."""
    n, k = A.shape
    k2, m = B.shape
    assert k == k2, f"несовместимые формы {A.shape} @ {B.shape}"
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            for t in range(k):
                C[i, j] += A[i, t] * B[t, j]
    return C


def linear_layer(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Батч X (N, d_in), W (d_out, d_in), b (d_out,) -> (N, d_out). Ровно как nn.Linear: X W^T + b."""
    return X @ W.T + b


# ---------- Аффинные преобразования в однородных координатах ----------
def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def scaling(sx: float, sy: float) -> np.ndarray:
    return np.diag([sx, sy, 1.0])


def translation(tx: float, ty: float) -> np.ndarray:
    T = np.eye(3)
    T[:2, 2] = tx, ty
    return T


def transform_bbox(box: tuple[float, float, float, float], M: np.ndarray) -> tuple[float, ...]:
    """Преобразует bbox (x1, y1, x2, y2): 4 угла -> M -> снова охватывающий прямоугольник.

    Так аугментации (поворот, масштаб) пересчитывают разметку детекции.
    """
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]]).T   # (3, 4): углы — столбцы
    moved = M @ corners
    xs, ys = moved[0], moved[1]
    return xs.min(), ys.min(), xs.max(), ys.max()


# ---------- Свёртка как матричное умножение (превью урока 70) ----------
def im2col(img: np.ndarray, k: int) -> np.ndarray:
    """(H, W) -> (out_h*out_w, k*k): каждая строка — развёрнутое окно k×k (stride 1, без padding)."""
    H, W = img.shape
    out_h, out_w = H - k + 1, W - k + 1
    cols = np.empty((out_h * out_w, k * k), dtype=img.dtype)
    for i in range(out_h):
        for j in range(out_w):
            cols[i * out_w + j] = img[i:i + k, j:j + k].ravel()
    return cols


def conv2d_im2col(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Свёртка (в DL-смысле, т. е. кросс-корреляция) = im2col @ vec(kernel)."""
    k = kernel.shape[0]
    H, W = img.shape
    return (im2col(img, k) @ kernel.ravel()).reshape(H - k + 1, W - k + 1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(3, 4)), rng.normal(size=(4, 2))
    print("matmul_loop == A @ B:", np.allclose(matmul_loop(A, B), A @ B))
    print("(AB)^T == B^T A^T:   ", np.allclose((A @ B).T, B.T @ A.T))
    print("||A||_F через след:  ", np.isclose(np.linalg.norm(A, "fro"), np.sqrt(np.trace(A.T @ A))))

    X = rng.normal(size=(5, 3))
    W, b = rng.normal(size=(2, 3)), rng.normal(size=2)
    print("linear layer:", linear_layer(X, W, b).shape)

    # Поворот вокруг центра кадра 640x480: перенос в начало -> поворот -> перенос обратно.
    # Матрицы применяются справа налево!
    M = translation(320, 240) @ rotation(np.deg2rad(15)) @ translation(-320, -240)
    print("bbox после поворота на 15°:", np.round(transform_bbox((100, 100, 200, 160), M), 1))

    img = rng.normal(size=(6, 6))
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    print("im2col shape:", im2col(img, 3).shape, "-> conv out:", conv2d_im2col(img, sobel_x).shape)
