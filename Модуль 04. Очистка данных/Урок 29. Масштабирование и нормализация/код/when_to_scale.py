"""Кому нужен масштаб признаков: KNN, SVM, градиентный спуск — да; деревья — нет. Плюс выбросы против min-max."""
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from scalers import MinMaxScalerNP, RobustScalerNP, StandardScalerNP

X, y = load_wine(return_X_y=True)                 # 13 признаков в разных единицах: пролин ~1000, оттенок ~1
print("диапазоны признаков (min..max) у трёх из них:", [f"{X[:, j].min():g}..{X[:, j].max():g}" for j in (12, 7, 10)])

# 1. Сверка своих скейлеров со sklearn
for ours, ref in ((StandardScalerNP(), StandardScaler()), (MinMaxScalerNP(), MinMaxScaler()), (RobustScalerNP(), RobustScaler())):
    print(f"{type(ref).__name__:<15} совпадает: {np.allclose(ours.fit(X).transform(X), ref.fit_transform(X))}")

# 2. Кому важен масштаб
models = {"KNN (k=5)": KNeighborsClassifier(5), "SVM (RBF)": SVC(), "дерево решений": DecisionTreeClassifier(random_state=0)}
print()
for name, m in models.items():
    raw = cross_val_score(m, X, y, cv=5).mean()
    scaled = cross_val_score(make_pipeline(StandardScaler(), m), X, y, cv=5).mean()
    print(f"{name:<15} без масштаба {raw:.3f} | со StandardScaler {scaled:.3f}")

# 3. Градиентный спуск: число шагов до сходимости (связь с уроком 6 — число обусловленности)
def gd_steps(A, y, tol=1e-6, max_iter=200_000):
    H = 2 * A.T @ A / len(A)
    lr = 1 / np.linalg.eigvalsh(H)[-1]
    w = np.zeros(A.shape[1])
    for k in range(max_iter):
        g = 2 * A.T @ (A @ w - y) / len(A)
        if np.linalg.norm(g) < tol:
            return k, np.linalg.cond(H)
        w -= lr * g
    return max_iter, np.linalg.cond(H)

yc = y - y.mean()
for name, A in (("сырые (центрированные)", X - X.mean(0)), ("стандартизованные", StandardScaler().fit_transform(X))):
    k, cond = gd_steps(A, yc)
    print(f"GD на {name:<24}: cond(H) = {cond:.1e}, шагов {k}")

# 4. Выброс ломает min-max, но не robust
rng = np.random.default_rng(0)
x = rng.normal(50, 10, 1000)
x_out = np.append(x, 5000.0)[:, None]                  # одна ошибка датчика или разметки
mm = MinMaxScalerNP().fit(x_out).transform(x_out)[:-1]
rb = RobustScalerNP().fit(x_out).transform(x_out)[:-1]
print(f"\nс одним выбросом: min-max сжал нормальные значения в [{mm.min():.3f}, {mm.max():.3f}], robust — std {rb.std():.2f}")
