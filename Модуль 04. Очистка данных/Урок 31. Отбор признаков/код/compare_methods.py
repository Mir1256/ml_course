"""Сравнение методов отбора на данных с известной правдой: 5 информативных, 5 избыточных, 30 шумовых + «id» с высокой кардинальностью."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (RFE, SelectKBest, SequentialFeatureSelector, VarianceThreshold, f_classif,
                                       mutual_info_classif)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from selection import forward_selection, mutual_information_discrete, permutation_importance_np

X, y = make_classification(n_samples=1500, n_features=40, n_informative=5, n_redundant=5, n_repeated=0,
                           shuffle=False, random_state=0)               # первые 5 — информативные, 5–9 — их комбинации
rng = np.random.default_rng(0)
X = np.c_[X, rng.integers(0, 1500, len(y)),                            # столбец 40: «id кадра» — шум с огромной кардинальностью
          rng.integers(0, 2, (len(y), 5))]                              # 41–45: бинарный шум (2 значения)
informative = set(range(5))
top = lambda scores, k=5: sorted(np.argsort(scores)[::-1][:k].tolist())
print("правда: информативные", sorted(informative), "| избыточные 5–9 | шум 10–39 | id = 40 | бинарный шум 41–45")

# Filter
print("\nFILTER")
print("  ANOVA F             :", top(f_classif(X, y)[0]))
print("  взаимная информация :", top(mutual_info_classif(X, y, random_state=0)))
xq = np.digitize(X[:, 0], np.quantile(X[:, 0], [0.25, 0.5, 0.75]))
print(f"  своя MI (признак 0, квартили) = {mutual_information_discrete(xq, y):.3f} | у шумового 20: "
      f"{mutual_information_discrete(np.digitize(X[:, 20], np.quantile(X[:, 20], [0.25, 0.5, 0.75])), y):.3f}")
print("  VarianceThreshold(0.5) оставил признаков:", VarianceThreshold(0.5).fit(X).get_support().sum(), "из 46 — выкинул только бинарные (дисперсия 0.25), на цель не смотрит")

# Wrapper
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
print("\nWRAPPER")
print("  свой прямой отбор   :", sorted(forward_selection(lr, X[:, :20], y, k=5)))
sfs = SequentialFeatureSelector(lr, n_features_to_select=5, cv=5).fit(X[:, :20], y)
print("  SequentialFeatureSelector:", np.flatnonzero(sfs.get_support()).tolist())
print("  RFE (логрег)        :", np.flatnonzero(RFE(LogisticRegression(max_iter=1000), n_features_to_select=5).fit(StandardScaler().fit_transform(X), y).support_).tolist())

# Embedded
print("\nEMBEDDED")
l1 = LogisticRegression(l1_ratio=1, C=0.05, solver="liblinear").fit(StandardScaler().fit_transform(X), y)   # l1_ratio=1 — чистый L1
print("  L1-логрег, ненулевые веса:", np.flatnonzero(l1.coef_[0]).tolist())
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
rf = RandomForestClassifier(300, random_state=0).fit(Xtr, ytr)
imp = rf.feature_importances_
print(f"  RF impurity:    id {imp[40]:.4f} | бинарный шум (среднее) {imp[41:].mean():.4f}  <- смещение к признакам с многими значениями")
perm = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=0).importances_mean
print(f"  RF permutation: id {perm[40]:+.4f} | бинарный шум (среднее) {perm[41:].mean():+.4f}  <- оба ≈ 0, как и должно быть")
ours = permutation_importance_np(rf, Xte, yte, n_repeats=10)
print("  своя permutation importance, топ-10:", top(ours, 10))

# Отбор внутри CV против снаружи (урок 27)
Xn = rng.normal(size=(400, 3000)); yn = rng.integers(0, 2, 400)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)            # повторы уменьшают шум оценки
outside = cross_val_score(LogisticRegression(max_iter=1000), SelectKBest(f_classif, k=20).fit_transform(Xn, yn), yn, cv=cv).mean()
inside = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000)), Xn, yn, cv=cv).mean()
print(f"\nчистый шум: отбор ДО CV {outside:.2f} | отбор ВНУТРИ Pipeline {inside:.2f}")
