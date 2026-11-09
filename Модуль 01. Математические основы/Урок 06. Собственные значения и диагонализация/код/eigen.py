"""Собственные значения с нуля: степенной метод, QR-алгоритм, связь спектра со скоростью градиентного спуска."""
import numpy as np


def power_iteration(A: np.ndarray, iters: int = 1000, tol: float = 1e-12, seed: int = 0):
    """Наибольшее по модулю собственное значение и его вектор.

    v_{k+1} = A v_k / ||A v_k||: компонента вдоль главного вектора растёт как |λ1|^k,
    остальные — как |λi|^k, поэтому их доля убывает как |λ2/λ1|^k.
    """
    v = np.random.default_rng(seed).normal(size=A.shape[0])
    v /= np.linalg.norm(v)
    lam = 0.0
    for k in range(iters):
        w = A @ v
        v_new = w / np.linalg.norm(w)
        lam_new = v_new @ A @ v_new                    # отношение Рэлея (v^T A v, ||v|| = 1)
        if abs(lam_new - lam) < tol:
            return lam_new, v_new, k + 1
        v, lam = v_new, lam_new
    return lam, v, iters


def qr_algorithm(A: np.ndarray, iters: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Все собственные пары симметричной матрицы: A_k = Q_k R_k -> A_{k+1} = R_k Q_k.

    A_{k+1} = Q_k^T A_k Q_k подобна A_k (тот же спектр) и сходится к диагональной;
    произведение всех Q_k — матрица собственных векторов.
    """
    Ak = A.astype(float).copy()
    V = np.eye(A.shape[0])
    for _ in range(iters):
        Q, R = np.linalg.qr(Ak)
        Ak = R @ Q
        V = V @ Q
    return np.diag(Ak), V


def gd_on_quadratic(A: np.ndarray, lr: float, steps: int = 200) -> np.ndarray:
    """Градиентный спуск на f(x) = 1/2 x^T A x (минимум в 0). Возвращает ||x_k|| по шагам.

    grad f = A x, поэтому x_{k+1} = (I - lr A) x_k: вдоль i-го собственного вектора
    координата умножается на (1 - lr λ_i) на каждом шаге.
    """
    x = np.ones(A.shape[0])
    norms = []
    for _ in range(steps):
        x = x - lr * A @ x
        norms.append(np.linalg.norm(x))
    return np.array(norms)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B = rng.normal(size=(5, 5))
    S = B @ B.T                                         # симметричная положительно полуопределённая
    lam, v, k = power_iteration(S)
    ref = np.linalg.eigh(S)                             # eigh — для симметричных: вещественный спектр по возрастанию
    print(f"степенной метод: λ_max = {lam:.6f} за {k} итераций | eigh: {ref.eigenvalues[-1]:.6f}")
    vals, V = qr_algorithm(S)
    print("QR-алгоритм:", np.sort(vals).round(6))
    print("eigh       :", ref.eigenvalues.round(6))
    print("V ортогональна:", np.allclose(V.T @ V, np.eye(5)), "| S = V Λ V^T:", np.allclose(V @ np.diag(vals) @ V.T, S))

    # Диагонализация упрощает степени: A^k = P Λ^k P^{-1}
    A = np.array([[0.9, 0.2], [0.1, 0.8]])
    w, P = np.linalg.eig(A)
    A50 = P @ np.diag(w ** 50) @ np.linalg.inv(P)
    print("A^50 через диагонализацию совпадает:", np.allclose(A50, np.linalg.matrix_power(A, 50)))

    # Спектр гессиана и выбор learning rate: сходимость iff lr < 2 / λ_max
    H = np.diag([1.0, 10.0, 100.0])                     # число обусловленности 100
    for lr in (0.005, 0.019, 0.021):
        print(f"lr = {lr}: ||x|| после 200 шагов = {gd_on_quadratic(H, lr)[-1]:.3e}  (граница 2/λ_max = 0.02)")
