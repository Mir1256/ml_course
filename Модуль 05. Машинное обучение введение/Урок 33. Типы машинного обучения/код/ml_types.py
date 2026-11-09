"""Пять типов ML на одном датасете digits: сколько меток нужно и что делать, когда их мало."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier

rng = np.random.default_rng(0)
X, y = load_digits(return_X_y=True)
X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
labeled = rng.choice(len(Xtr), 50, replace=False)                 # размечено всего 50 картинок из 1257
logreg = lambda: LogisticRegression(max_iter=3000)

# ---------- 1. Supervised: учимся на парах (x, y) ----------
full = logreg().fit(Xtr, ytr).score(Xte, yte)
few = logreg().fit(Xtr[labeled], ytr[labeled]).score(Xte, yte)
print(f"SUPERVISED: все {len(Xtr)} меток -> {full:.3f} | только 50 меток -> {few:.3f}")

# ---------- 2. Unsupervised: меток нет вообще, ищем структуру ----------
km = KMeans(10, n_init=10, random_state=0).fit(Xtr)
C = np.zeros((10, 10), int)
np.add.at(C, (km.labels_, ytr), 1)                                # кластер × истинный класс
rows, cols = linear_sum_assignment(-C)                            # лучшее соответствие кластер -> класс (венгерский алгоритм)
mapping = dict(zip(rows, cols))
print(f"UNSUPERVISED: k-means без единой метки, после сопоставления кластеров с классами -> "
      f"{np.mean([mapping[c] == t for c, t in zip(km.predict(Xte), yte)]):.3f}")

# ---------- 3. Semi-supervised: 50 меток + 1207 неразмеченных ----------
y_semi = np.full(len(ytr), -1)                                    # -1 = «метки нет» (соглашение sklearn)
y_semi[labeled] = ytr[labeled]
st = SelfTrainingClassifier(logreg(), threshold=0.9).fit(Xtr, y_semi)
with np.errstate(invalid="ignore", divide="ignore"):              # внутри sklearn делит на 0 для изолированных точек графа
    ls_score = LabelSpreading(kernel="knn", n_neighbors=7).fit(Xtr, y_semi).score(Xte, yte)
print(f"SEMI-SUPERVISED: self-training (псевдоразметка) -> {st.score(Xte, yte):.3f} | label spreading по графу соседей -> {ls_score:.3f}")

# ---------- 4. Self-supervised: «метки» берём из самих данных ----------
# Предтекстовая задача: по левой половине цифры восстановить правую. Метки не нужны — «ответ» есть в картинке.
left, right = X.reshape(-1, 8, 8)[:, :, :4].reshape(-1, 32), X.reshape(-1, 8, 8)[:, :, 4:].reshape(-1, 32)
Xtr_l, Xte_l = train_test_split(left, test_size=0.3, random_state=0, stratify=y)
Xtr_r, _ = train_test_split(right, test_size=0.3, random_state=0, stratify=y)
pretext = MLPRegressor(hidden_layer_sizes=(256,), max_iter=1000, random_state=0).fit(Xtr_l, Xtr_r)
embed = lambda A: np.maximum(0, A @ pretext.coefs_[0] + pretext.intercepts_[0])   # скрытый слой (ReLU) = выученные признаки
raw_scores, ssl_scores = [], []
for seed in range(3):                                              # разные наборы из 50 меток: результат на 50 метках шумный
    lab = np.random.default_rng(seed).choice(len(Xtr_l), 50, replace=False)
    raw_scores.append(logreg().fit(Xtr_l[lab], ytr[lab]).score(Xte_l, yte))
    ssl_scores.append(logreg().fit(embed(Xtr_l)[lab], ytr[lab]).score(embed(Xte_l), yte))
print(f"SELF-SUPERVISED: линейный классификатор на 50 метках по ЛЕВОЙ половине (3 набора меток): "
      f"сырые пиксели -> {np.round(raw_scores, 3).tolist()} | признаки из предтекстовой задачи -> {np.round(ssl_scores, 3).tolist()}")

# ---------- 5. Reinforcement learning: ни меток, ни «ответов» — только награда за действие ----------
true_p = np.array([0.30, 0.55, 0.45])                              # три порога уверенности детектора: доля «полезных» тревог
counts, values, total = np.zeros(3), np.zeros(3), 0.0
for t in range(5000):
    a = rng.integers(3) if rng.random() < 0.1 else int(np.argmax(values))     # ε-жадно: 10% исследуем, 90% используем
    r = float(rng.random() < true_p[a])                                         # награда — единственный сигнал
    counts[a] += 1
    values[a] += (r - values[a]) / counts[a]                                    # скользящее среднее награды действия
    total += r
print(f"RL (бандит): выбирал действия {counts.astype(int).tolist()}, оценки ценности {values.round(3).tolist()}, "
      f"средняя награда {total / 5000:.3f} (лучшее возможное 0.55)")
