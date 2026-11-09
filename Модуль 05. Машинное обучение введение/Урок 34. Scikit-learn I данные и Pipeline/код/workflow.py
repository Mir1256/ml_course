"""Полный «правильный» поток scikit-learn на смешанной таблице: загрузка -> сплит -> ColumnTransformer -> Pipeline."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_digits
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mini_sklearn import CyclicEncoder, MiniPipeline, group_split, stratified_split

rng = np.random.default_rng(0)

# ---------- 1. Свой Pipeline == sklearn Pipeline ----------
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
ours = MiniPipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]).fit(Xtr, ytr)
ref = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]).fit(Xtr, ytr)
print(f"MiniPipeline {ours.score(Xte, yte):.4f} | sklearn Pipeline {ref.score(Xte, yte):.4f} | предсказания совпадают: {np.array_equal(ours.predict(Xte), ref.predict(Xte))}")

# ---------- 2. Стратификация: доли редкого класса в маленьком test ----------
y_rare = (rng.random(300) < 0.05).astype(int)                       # 5% положительных — как редкий дефект
plain = [y_rare[train_test_split(np.arange(300), test_size=0.2, random_state=s)[1]].mean() for s in range(200)]
strat = [y_rare[stratified_split(y_rare, 0.2, seed=s)[1]].mean() for s in range(200)]
print(f"\nдоля редкого класса в test (истинная {y_rare.mean():.3f}): без стратификации {np.mean(plain):.3f} ± {np.std(plain):.3f} "
      f"(min {min(plain):.3f}), со стратификацией {np.mean(strat):.3f} ± {np.std(strat):.3f}")

# ---------- 3. Смешанная таблица «кадры детектора» с пропусками и категориями ----------
n = 3000
df = pd.DataFrame({
    "camera": rng.choice(["gate", "dock", "yard", "roof"], n),
    "hour": rng.integers(0, 24, n),
    "brightness": rng.normal(120, 40, n),
    "blur": rng.gamma(2, 1.5, n),
    "video_id": rng.integers(0, 60, n),
})
df.loc[rng.random(n) < 0.1, "brightness"] = np.nan
night = (df.hour < 6) | (df.hour >= 21)
logit = -1 + 1.5 * night - 0.02 * (df.brightness.fillna(120) - 120) + 0.4 * df.blur + (df.camera == "roof") * 1.0
df["missed"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)        # «детектор пропустил объект»

features = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["camera"]),
    ("hour", CyclicEncoder(24), ["hour"]),
    ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]),
     ["brightness", "blur"]),
])
model = Pipeline([("features", features), ("clf", LogisticRegression(max_iter=2000))])
X_df, y_df = df.drop(columns=["missed", "video_id"]), df.missed
scores = cross_val_score(model, X_df, y_df, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
print(f"\nPipeline на сыром DataFrame: ROC-AUC {scores.mean():.3f} ± {scores.std():.3f}")
model.fit(X_df, y_df)
print("признаки после ColumnTransformer:", model[:-1].get_feature_names_out().tolist())

# ---------- 4. Групповой сплит ----------
tr, te = group_split(df.video_id.to_numpy(), 0.2)
print(f"\ngroup_split: видео в train ∩ test = {len(set(df.video_id.iloc[tr]) & set(df.video_id.iloc[te]))}; "
      f"GroupKFold: {cross_val_score(model, X_df, y_df, cv=GroupKFold(5), groups=df.video_id, scoring='roc_auc').mean():.3f}")
