"""Разложение ошибки на смещение и разброс (bias–variance) симуляцией: много обучающих выборок из одного распределения."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)                # истинная зависимость (её мы обычно не знаем)
SIGMA = 0.3                                         # шум: неустранимая ошибка
x_test = np.linspace(0.05, 0.95, 50)


def decompose(fit_predict, n_train=30, n_sets=500):
    """Для каждой из n_sets обучающих выборок обучаем модель и предсказываем в точках x_test."""
    preds = np.empty((n_sets, x_test.size))
    for s in range(n_sets):
        x = rng.uniform(0, 1, n_train)
        y = f(x) + rng.normal(0, SIGMA, n_train)
        preds[s] = fit_predict(x, y, x_test)
    mean_pred = preds.mean(axis=0)
    bias2 = np.mean((mean_pred - f(x_test)) ** 2)                 # насколько «в среднем» модель промахивается
    variance = np.mean(preds.var(axis=0))                         # насколько модель скачет от выборки к выборке
    y_new = f(x_test) + rng.normal(0, SIGMA, (n_sets, x_test.size))
    total = np.mean((y_new - preds) ** 2)                         # ожидаемая ошибка на новых точках
    return bias2, variance, total


def poly(degree):
    return lambda x, y, xt: np.polyval(np.polyfit(x, y, degree), xt)


if __name__ == "__main__":
    print(f"неустранимый шум σ² = {SIGMA**2:.3f}")
    print("степень | bias²  | variance | σ²+bias²+var | средняя ошибка (симуляция)")
    rows = []
    for d in (0, 1, 3, 5, 9, 12):
        b, v, t = decompose(poly(d))
        rows.append((d, b, v, t))
        print(f"{d:>7} | {b:.3f}  | {v:8.3f} | {SIGMA**2 + b + v:12.3f} | {t:.3f}")

    # То же для KNN: маленькое k — малое смещение и большой разброс
    print("\nKNN k  | bias²  | variance")
    for k in (1, 3, 10, 25):
        knn = lambda x, y, xt, k=k: KNeighborsRegressor(k).fit(x[:, None], y).predict(xt[:, None])
        b, v, _ = decompose(knn)
        print(f"{k:>6} | {b:.3f}  | {v:.3f}")

    # Библиотечный аналог: кривая валидации по степени полинома на одной выборке
    x = rng.uniform(0, 1, 60)
    y = f(x) + rng.normal(0, SIGMA, 60)
    degrees = np.arange(0, 13)
    model = make_pipeline(PolynomialFeatures(), StandardScaler(), LinearRegression())
    train_s, val_s = validation_curve(model, x[:, None], y, param_name="polynomialfeatures__degree",
                                      param_range=degrees, cv=5, scoring="neg_mean_squared_error")
    best = degrees[np.argmax(val_s.mean(axis=1))]
    print(f"\nvalidation_curve: лучшая степень по CV = {best}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    d, b, v, t = map(np.array, zip(*rows))
    axes[0].plot(d, b, "o-", label="bias²"); axes[0].plot(d, v, "o-", label="variance"); axes[0].plot(d, t, "o-", label="ошибка")
    axes[0].axhline(SIGMA**2, ls="--", c="gray", label="σ²"); axes[0].set_yscale("log")
    axes[0].set_xlabel("степень полинома"); axes[0].legend(); axes[0].set_title("Разложение ошибки (симуляция)")
    axes[1].plot(degrees, -train_s.mean(1), "o-", label="train"); axes[1].plot(degrees, -val_s.mean(1), "o-", label="CV")
    axes[1].set_yscale("log"); axes[1].set_xlabel("степень полинома"); axes[1].set_ylabel("MSE"); axes[1].legend()
    axes[1].set_title("Кривая валидации (sklearn)")
    fig.tight_layout(); fig.savefig("bias_variance.png", dpi=80); plt.close(fig)
