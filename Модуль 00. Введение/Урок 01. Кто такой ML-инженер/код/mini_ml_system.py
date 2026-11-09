"""Мини-ML-система: данные -> обучение -> оценка -> «деплой» -> мониторинг.

Всё на NumPy, чтобы было видно каждую стадию, за которую отвечает ML-инженер.
"""
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(42)


def make_data(n: int, shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Синтетика: два признака кадра (x1 — яркость, x2 — размытость) -> качество детекции.

    shift сдвигает яркость: так имитируем смену условий съёмки (день -> вечер).
    Зависимость слегка нелинейная — линейная модель хороша только около «обычной» яркости.
    """
    x1 = rng.normal(0.5 + shift, 0.1, size=n)
    x2 = rng.normal(0.3, 0.1, size=n)
    y = 2.0 * x1 - 1.0 * x2 + 3.0 * (x1 - 0.5) ** 2 + 0.3 + rng.normal(0, 0.05, size=n)
    return np.column_stack([x1, x2]), y


# ---------- 1. Данные: разбиение ДО любой статистики (иначе утечка) ----------
X, y = make_data(1000)
idx = rng.permutation(len(X))
train_idx, val_idx = idx[:800], idx[800:]
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

# ---------- 2. Предобработка: статистики считаем ТОЛЬКО по train ----------
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)


def standardize(X: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


Z_train, Z_val = standardize(X_train), standardize(X_val)

# ---------- 3. Обучение: градиентный спуск по MSE ----------
w, b = np.zeros(2), 0.0
lr, n = 0.1, len(Z_train)
for epoch in range(500):
    err = Z_train @ w + b - y_train          # остатки, shape (n,)
    grad_w = 2.0 / n * Z_train.T @ err       # dMSE/dw
    grad_b = 2.0 / n * err.sum()             # dMSE/db
    w -= lr * grad_w
    b -= lr * grad_b


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


# ---------- 4. Оценка: всегда сравниваем с тривиальным бейзлайном ----------
val_pred = Z_val @ w + b
baseline = np.full_like(y_val, y_train.mean())
print(f"val MSE модели   = {mse(y_val, val_pred):.5f}")
print(f"val MSE бейзлайна = {mse(y_val, baseline):.5f}")

# ---------- 5. «Деплой»: артефакт = веса + ВСЯ предобработка ----------
artifact = Path("model.json")
artifact.write_text(json.dumps({"w": w.tolist(), "b": b, "mu": mu.tolist(), "sigma": sigma.tolist()}))


def load_predictor(path: Path):
    p = json.loads(path.read_text())
    w_, b_, mu_, sigma_ = np.array(p["w"]), p["b"], np.array(p["mu"]), np.array(p["sigma"])
    return lambda X_new: ((X_new - mu_) / sigma_) @ w_ + b_


predict = load_predictor(artifact)
assert np.allclose(predict(X_val), val_pred), "training-serving skew: сервинг считает иначе, чем обучение"

# ---------- 6. Мониторинг: дрейф входных данных и деградация ----------
for shift in (0.0, 0.2, 0.4):
    X_new, y_new = make_data(500, shift=shift)
    z_mean = np.abs(standardize(X_new).mean(axis=0)).max()  # насколько «уехали» признаки, в сигмах train
    alert = "ДРЕЙФ!" if z_mean > 1.0 else "ok"
    print(f"shift={shift:.1f}: |mean z|={z_mean:.2f} [{alert}]  MSE={mse(y_new, predict(X_new)):.5f}")
