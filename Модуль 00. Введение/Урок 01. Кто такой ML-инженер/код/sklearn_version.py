"""Та же задача через scikit-learn: Pipeline сам хранит предобработку вместе с моделью."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
x1 = rng.normal(0.5, 0.1, 1000)
x2 = rng.normal(0.3, 0.1, 1000)
y = 2.0 * x1 - 1.0 * x2 + 3.0 * (x1 - 0.5) ** 2 + 0.3 + rng.normal(0, 0.05, 1000)
X = np.column_stack([x1, x2])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(StandardScaler(), LinearRegression())  # scaler.fit увидит только train
model.fit(X_train, y_train)
print(f"val MSE sklearn = {mean_squared_error(y_val, model.predict(X_val)):.5f}")

# Сверка с NumPy-версией: градиентный спуск должен сойтись к тому же решению,
# что и аналитическое решение МНК внутри LinearRegression.
Z = (X_train - X_train.mean(0)) / X_train.std(0)
w, b = np.zeros(2), 0.0
for _ in range(500):
    err = Z @ w + b - y_train
    w -= 0.1 * 2 / len(Z) * Z.T @ err
    b -= 0.1 * 2 / len(Z) * err.sum()
lr = model[-1]
print("веса GD     :", np.round(w, 4), round(b, 4))
print("веса sklearn:", np.round(lr.coef_, 4), round(lr.intercept_, 4))
print("совпадают:", np.allclose(w, lr.coef_, atol=1e-4) and np.isclose(b, lr.intercept_, atol=1e-4))
