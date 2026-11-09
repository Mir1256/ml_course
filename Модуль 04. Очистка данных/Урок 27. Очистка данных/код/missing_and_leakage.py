"""Пропуски (MCAR / MAR / MNAR), импутация без утечки и три вида утечек данных."""
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
n = 4000

# ---------- 1. Механизмы пропусков: от них зависит, можно ли просто выбросить строки ----------
exposure = rng.normal(0, 1, n)                 # экспозиция кадра
blur = 0.6 * exposure + 0.8 * rng.normal(0, 1, n)   # размытость, коррелирует с экспозицией (тёмные кадры — длинная выдержка)
y = (0.8 * exposure - 1.2 * blur + rng.normal(0, 1, n) > 0).astype(int)   # «детектор справился»
df = pd.DataFrame({"exposure": exposure, "blur": blur, "y": y})

mcar = rng.random(n) < 0.3                                  # MCAR: сбой записи — случайно
mar = rng.random(n) < np.where(df.exposure < 0, 0.5, 0.1)   # MAR: зависит от НАБЛЮДАЕМОГО признака
mnar = rng.random(n) < np.where(df.blur > 0.5, 0.6, 0.05)   # MNAR: размытость не измерена, когда она большая

for name, mask in (("MCAR", mcar), ("MAR", mar), ("MNAR", mnar)):
    observed = df.blur.where(~mask)
    print(f"{name}: пропусков {mask.mean():.0%} | среднее blur по наблюдаемым {observed.mean():+.3f} (истинное {df.blur.mean():+.3f})")

# ---------- 2. Индикатор пропуска — сам по себе информативный признак при MNAR ----------
X = df[["exposure", "blur"]].copy()
X.loc[mnar, "blur"] = np.nan
for name, imputer in (("медиана", SimpleImputer(strategy="median")),
                      ("медиана + индикатор", SimpleImputer(strategy="median", add_indicator=True)),
                      ("KNN (k=5)", KNNImputer(n_neighbors=5))):
    pipe = make_pipeline(imputer, StandardScaler(), LogisticRegression())    # импутер ВНУТРИ пайплайна: учится на train-фолде
    print(f"импутация «{name}»: accuracy CV = {cross_val_score(pipe, X, df.y, cv=5).mean():.3f}")

# ---------- 3. Утечка: предобработка на всём датасете до разбиения ----------
# Шумные признаки без связи с целью: честная accuracy ≈ 0.5. Отбор «лучших» признаков на ВСЕХ данных её завышает.
Xn = rng.normal(size=(200, 2000))
yn = rng.integers(0, 2, 200)
corr = np.abs((Xn - Xn.mean(0)).T @ (yn - yn.mean())) / 200
top = np.argsort(corr)[-20:]                                 # отбор по корреляции с целью на всём датасете — утечка
leaky = cross_val_score(LogisticRegression(max_iter=1000), Xn[:, top], yn, cv=5).mean()
print(f"\nшум + отбор признаков ДО разбиения: accuracy {leaky:.2f}  <- иллюзия: связи нет, должна быть ≈ 0.5")

# ---------- 4. Утечка через группы: почти-копии кадров одного видео в train и test ----------
videos = np.repeat(np.arange(40), 50)                        # 40 видео по 50 кадров
video_feat = rng.normal(size=(40, 10))[videos]               # кадры одного видео почти одинаковы
Xv = video_feat + 0.05 * rng.normal(size=(2000, 10))
yv = rng.integers(0, 2, 40)[videos]                          # метка — свойство видео, со зрительными признаками не связана
knn = KNeighborsClassifier(1)
print(f"1-NN, случайный KFold: {cross_val_score(knn, Xv, yv, cv=KFold(5, shuffle=True, random_state=0)).mean():.2f}"
      f" | GroupKFold по видео: {cross_val_score(knn, Xv, yv, cv=GroupKFold(5), groups=videos).mean():.2f}")
