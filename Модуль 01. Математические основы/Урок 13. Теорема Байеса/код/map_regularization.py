"""MAP = регуляризация: гауссов prior -> Ridge (L2), лапласов prior -> Lasso (L1). Плюс наивный Байес с нуля."""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

rng = np.random.default_rng(0)
n, d = 50, 8
X = rng.normal(size=(n, d))
w_true = np.array([3.0, -2.0, 0, 0, 1.5, 0, 0, 0])
sigma = 1.0                                                     # шум наблюдений: y = Xw + N(0, σ²)
y = X @ w_true + rng.normal(0, sigma, n)

# Гауссов prior w ~ N(0, τ² I): -log posterior = ||y - Xw||²/(2σ²) + ||w||²/(2τ²) + const
tau = 0.5
w_map = np.linalg.solve(X.T @ X + (sigma**2 / tau**2) * np.eye(d), X.T @ y)
ridge = Ridge(alpha=sigma**2 / tau**2, fit_intercept=False).fit(X, y)
print("MAP с гауссовым prior == Ridge(alpha = σ²/τ²):", np.allclose(w_map, ridge.coef_))
print("MLE (МНК):", np.linalg.lstsq(X, y, rcond=None)[0].round(2))
print("MAP (L2): ", w_map.round(2), "<- веса стянуты к нулю")

# Лапласов prior p(w_j) ∝ exp(-|w_j|/b): -log posterior = ||y - Xw||²/(2σ²) + ||w||_1 / b.
# sklearn Lasso минимизирует ||y - Xw||²/(2n) + α||w||_1  ->  α = σ²/(n b)
b = 0.05
lasso = Lasso(alpha=sigma**2 / (n * b), fit_intercept=False, tol=1e-10, max_iter=100_000).fit(X, y)
w = lasso.coef_
g = X.T @ (y - X @ w) / sigma**2                                # -градиент гладкой части
nz = w != 0
kkt = np.allclose(g[nz], np.sign(w[nz]) / b, atol=1e-6) and np.all(np.abs(g[~nz]) <= 1 / b + 1e-6)
print("MAP (L1): ", w.round(2), "<- точные нули: разреженность")
print("условия оптимальности MAP-задачи с лапласовым prior выполнены:", kkt)

# Наивный Байес: P(y | x) ∝ P(y) Π_j P(x_j | y), каждое P(x_j | y) — одномерная нормаль
Xi, yi = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xi, yi, test_size=0.3, random_state=0, stratify=yi)
classes = np.unique(ytr)
log_prior = np.log(np.array([np.mean(ytr == c) for c in classes]))
mu = np.array([Xtr[ytr == c].mean(axis=0) for c in classes])                    # (K, d)
var = np.array([Xtr[ytr == c].var(axis=0) for c in classes]) + 1e-9 * Xtr.var(axis=0).max()
log_lik = -0.5 * (np.log(2 * np.pi * var)[None] + (Xte[:, None, :] - mu[None]) ** 2 / var[None]).sum(axis=2)
pred = np.argmax(log_lik + log_prior, axis=1)                                   # сумма логарифмов вместо произведения
sk = GaussianNB().fit(Xtr, ytr)
print(f"\nнаивный Байес с нуля: accuracy {np.mean(pred == yte):.3f} | sklearn GaussianNB: {sk.score(Xte, yte):.3f} | предсказания совпадают: {np.array_equal(pred, sk.predict(Xte))}")
