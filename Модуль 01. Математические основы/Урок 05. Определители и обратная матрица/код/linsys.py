"""Определитель, решение СЛАУ, обратная матрица и ранг — с нуля через метод Гаусса."""
import numpy as np


def gauss_eliminate(A: np.ndarray, b: np.ndarray | None = None):
    """Прямой ход метода Гаусса с выбором главного элемента по столбцу (partial pivoting).

    Возвращает верхнетреугольную U, преобразованную правую часть и число перестановок строк.
    """
    U = A.astype(float).copy()
    rhs = None if b is None else b.astype(float).copy()
    n = U.shape[0]
    swaps = 0
    for col in range(n):
        pivot = col + np.argmax(np.abs(U[col:, col]))     # самый большой по модулю элемент — меньше ошибка округления
        if pivot != col:
            U[[col, pivot]] = U[[pivot, col]]
            if rhs is not None:
                rhs[[col, pivot]] = rhs[[pivot, col]]
            swaps += 1
        if abs(U[col, col]) < 1e-15:
            continue                                     # столбец нулевой — матрица вырождена
        factors = U[col + 1:, col] / U[col, col]
        U[col + 1:] -= factors[:, None] * U[col]         # вычитаем строку col из всех строк ниже разом
        if rhs is not None:
            rhs[col + 1:] -= factors[:, None] * rhs[col] if rhs.ndim == 2 else factors * rhs[col]
    return U, rhs, swaps


def det(A: np.ndarray) -> float:
    """det = (-1)^(число перестановок) * произведение диагонали треугольной матрицы."""
    U, _, swaps = gauss_eliminate(A)
    return float((-1) ** swaps * np.prod(np.diag(U)))


def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Решение Ax = b: прямой ход Гаусса + обратная подстановка."""
    U, y, _ = gauss_eliminate(A, b)
    n = U.shape[0]
    x = np.zeros_like(y)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def inverse(A: np.ndarray) -> np.ndarray:
    """A^{-1} — решение A X = I сразу для всех столбцов единичной матрицы."""
    return solve(A, np.eye(A.shape[0]))


def rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """Ранг = число ненулевых (с допуском!) строк после приведения к ступенчатому виду."""
    M = A.astype(float).copy()
    rows, cols = M.shape
    r = 0
    for col in range(cols):
        if r == rows:
            break
        pivot = r + np.argmax(np.abs(M[r:, col]))
        if abs(M[pivot, col]) <= tol * max(1.0, np.abs(A).max()):
            continue
        M[[r, pivot]] = M[[pivot, r]]
        M[r + 1:] -= (M[r + 1:, col] / M[r, col])[:, None] * M[r]
        r += 1
    return r


def homography_from_4_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Перспективное преобразование по 4 парам точек: СЛАУ 8x8 (как cv2.getPerspectiveTransform).

    x' = (h11 x + h12 y + h13) / (h31 x + h32 y + 1), аналогично y'. Домножаем на знаменатель —
    получаем по 2 линейных уравнения на каждую пару точек относительно 8 неизвестных h.
    """
    rows, rhs = [], []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y]); rhs.append(v)
    h = solve(np.array(rows, float), np.array(rhs, float))
    return np.append(h, 1.0).reshape(3, 3)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4))
    b = rng.normal(size=4)
    print(f"det: свой {det(A):.6f} | numpy {np.linalg.det(A):.6f}")
    print("solve совпадает:", np.allclose(solve(A, b), np.linalg.solve(A, b)))
    print("inverse: A @ A^-1 = I:", np.allclose(A @ inverse(A), np.eye(4)))
    S = np.array([[1, 2, 3], [2, 4, 6], [0, 1, 0]])
    print("rank вырожденной:", rank(S), "| det:", round(det(S), 12))

    # Определитель аффинного преобразования = во сколько раз меняется площадь
    M = np.array([[2.0, 0.5], [0.0, 1.5]])
    print("|det M| =", det(M), "-> площадь квадрата 1x1 после M:", abs(det(M)))

    # Документ сфотографирован под углом: 4 угла на фото -> прямоугольник 400x300
    src = np.array([[55, 40], [430, 75], [460, 380], [30, 340]], float)
    dst = np.array([[0, 0], [400, 0], [400, 300], [0, 300]], float)
    H = homography_from_4_points(src, dst)
    p = H @ np.array([430, 75, 1.0])
    print("H переводит угол (430, 75) в", np.round(p[:2] / p[2], 6))
