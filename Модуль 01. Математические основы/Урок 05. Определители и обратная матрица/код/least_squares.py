"""Нормальное уравнение МНК, мультиколлинеарность, псевдообратная и число обусловленности."""
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
n = 200
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
X = np.column_stack([np.ones(n), x1, x2])            # столбец единиц = свободный член
y = 1.0 + 2.0 * x1 - 3.0 * x2 + rng.normal(0, 0.1, size=n)

# 1. Нормальное уравнение: w = (X^T X)^{-1} X^T y. Через solve, а НЕ через inv — так точнее и быстрее.
w_normal = np.linalg.solve(X.T @ X, X.T @ y)
w_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
lr = LinearRegression().fit(X[:, 1:], y)
print("нормальное ур.:", w_normal.round(4))
print("lstsq        :", w_lstsq.round(4))
print("sklearn      :", np.r_[lr.intercept_, lr.coef_].round(4))

# 2. Мультиколлинеарность: третий признак почти копирует второй (например, ширина bbox в px и в см)
x3 = x1 * 2.54 + rng.normal(0, 1e-6, size=n)
X_bad = np.column_stack([X, x3])
print(f"\ncond(X^T X) хорошая: {np.linalg.cond(X.T @ X):.1e} | почти вырожденная: {np.linalg.cond(X_bad.T @ X_bad):.1e}")
w_bad = np.linalg.solve(X_bad.T @ X_bad, X_bad.T @ y)
print("веса при коллинеарности:", w_bad.round(2), "<- огромные и противоположные по знаку")

# 3. Псевдообратная даёт решение с минимальной нормой — устойчиво даже при вырожденности
X_exact = np.column_stack([X, x1 * 2.54])              # точная копия -> X^T X вырождена
w_pinv = np.linalg.pinv(X_exact) @ y
print("pinv при точной копии:", w_pinv.round(4), "| вес x1 разделён между x1 и его копией")

# 4. Гребневая регрессия (превью урока 40): X^T X + λI всегда обратима
lam = 1e-3
w_ridge = np.linalg.solve(X_bad.T @ X_bad + lam * np.eye(4), X_bad.T @ y)
print(f"ridge λ={lam}:", w_ridge.round(4), f"| cond = {np.linalg.cond(X_bad.T @ X_bad + lam * np.eye(4)):.1e}")

# 5. Число обусловленности = во сколько раз относительная ошибка входа может вырасти в ответе
for k in (4, 8, 12):
    Hk = 1.0 / (np.arange(k)[:, None] + np.arange(k)[None, :] + 1)    # матрица Гильберта
    x_true = np.ones(k)
    x_hat = np.linalg.solve(Hk, Hk @ x_true)
    print(f"Гильберт {k:>2}: cond = {np.linalg.cond(Hk):.1e}, ошибка решения = {np.abs(x_hat - x_true).max():.1e}")
