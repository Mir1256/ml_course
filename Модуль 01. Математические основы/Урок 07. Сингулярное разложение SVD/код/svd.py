"""SVD с нуля через собственное разложение A^T A; псевдообратная; малоранговое приближение; DLT-гомография."""
import numpy as np


def svd_via_eigh(A: np.ndarray, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Тонкое SVD A = U diag(s) V^T «по определению» (учебная версия, численно хуже np.linalg.svd).

    A^T A = V diag(s^2) V^T  ->  s_i = sqrt(λ_i),  u_i = A v_i / s_i.
    Допуск грубый (1e-6), потому что A^T A квадрирует обусловленность: шум округления ~1e-16 * λ_max
    в собственных значениях превращается в «сингулярные числа» ~1e-8 * σ_max, которым верить нельзя.
    """
    lam, V = np.linalg.eigh(A.T @ A)
    order = np.argsort(lam)[::-1]                       # по убыванию, как принято для SVD
    lam, V = np.clip(lam[order], 0, None), V[:, order]
    s = np.sqrt(lam)
    r = int(np.sum(s > tol * s[0]))                     # численный ранг
    U = A @ V[:, :r] / s[:r]                            # делим каждый столбец на своё σ (broadcasting по строке)
    return U, s[:r], V[:, :r].T


def low_rank(A: np.ndarray, k: int) -> np.ndarray:
    """Лучшее приближение ранга k (теорема Эккарта–Юнга): сумма первых k слагаемых σ_i u_i v_i^T."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k]


def pinv_svd(A: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """A^+ = V Σ^+ U^T: обращаем только «ненулевые» сингулярные числа."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_inv = np.where(s > rcond * s.max(), 1 / s, 0.0)
    return (Vt.T * s_inv) @ U.T


def homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """DLT: Ah = 0, ||h|| = 1 -> h = правый сингулярный вектор с наименьшим σ (решение МНК для однородной системы)."""
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.array(rows, float))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 4))
    U, s, Vt = svd_via_eigh(A)
    print("свой SVD: A = U Σ V^T:", np.allclose((U * s) @ Vt, A), "| U^T U = I:", np.allclose(U.T @ U, np.eye(4)))
    print("σ свои :", s.round(6))
    print("σ numpy:", np.linalg.svd(A, compute_uv=False).round(6))
    print(f"||A||_2 = σ1: {np.isclose(np.linalg.norm(A, 2), s[0])} | ||A||_F^2 = Σσ²: {np.isclose(np.linalg.norm(A, 'fro')**2, (s**2).sum())}")
    print(f"κ(A) = {s[0] / s[-1]:.3f} | κ(A^T A) = {np.linalg.cond(A.T @ A):.3f} = κ(A)^2 = {(s[0] / s[-1])**2:.3f}")

    # Эккарт–Юнг: ошибка лучшего ранга-k приближения = хвост сингулярных чисел
    B = rng.normal(size=(50, 30))
    sb = np.linalg.svd(B, compute_uv=False)
    for k in (1, 5, 20):
        err_f = np.linalg.norm(B - low_rank(B, k), "fro")
        err_2 = np.linalg.norm(B - low_rank(B, k), 2)
        print(f"k={k:>2}: ||B-B_k||_F = {err_f:.4f} (теория {np.sqrt((sb[k:]**2).sum()):.4f}), ||B-B_k||_2 = {err_2:.4f} (σ_{k+1} = {sb[k]:.4f})")

    X = np.column_stack([rng.normal(size=8), rng.normal(size=8)])
    X = np.column_stack([X, X[:, 0] + X[:, 1]])        # третий столбец — сумма первых двух: ранг 2
    print("pinv через SVD == np.linalg.pinv:", np.allclose(pinv_svd(X), np.linalg.pinv(X)))

    src = rng.uniform(0, 1, size=(10, 2))
    H_true = np.array([[1.2, 0.1, 0.3], [-0.05, 0.9, 0.2], [0.1, 0.05, 1.0]])
    p = np.c_[src, np.ones(10)] @ H_true.T
    dst = p[:, :2] / p[:, 2:]
    print("DLT по 10 точкам восстановил H:", np.allclose(homography_dlt(src, dst), H_true))
