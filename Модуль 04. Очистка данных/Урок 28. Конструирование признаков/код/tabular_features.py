"""Табличные признаки: кодирование категорий (и утечка в target encoding), циклическое время, log, признаки из рамок."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

rng = np.random.default_rng(0)

# ---------- 1. Target encoding: «среднее цели по категории» — мощно и опасно ----------
# 2000 камер-«шумов» без связи с целью: честная accuracy ≈ 0.5
n = 3000
camera = rng.integers(0, 2000, n).astype(str)
y = rng.integers(0, 2, n)
naive = pd.Series(y).groupby(camera).transform("mean").to_numpy()[:, None]      # среднее по ВСЕМ строкам, включая свою
print(f"наивный target encoding: accuracy {cross_val_score(LogisticRegression(), naive, y, cv=5).mean():.2f}  <- утечка: строка видит свою же метку")
pipe = make_pipeline(TargetEncoder(target_type="binary", cv=KFold(5, shuffle=True, random_state=0)), LogisticRegression())
print(f"TargetEncoder (кросс-фиттинг внутри fit): accuracy {cross_val_score(pipe, camera[:, None], y, cv=5).mean():.2f}")

# ---------- 2. Циклическое время: 23:00 и 01:00 должны быть рядом ----------
hours = np.array([0, 1, 12, 23])
enc = np.c_[np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)]
d = lambda a, b: np.linalg.norm(enc[a] - enc[b])
print(f"\nрасстояние 23ч–1ч: как число {abs(23 - 1)}, в (sin, cos) {d(3, 1):.3f} | 0ч–12ч: {d(0, 2):.3f}")

# ---------- 3. Признаки из рамок детекции (доменное знание важнее алгоритма) ----------
boxes = pd.DataFrame({"x1": [10, 600, 300], "y1": [20, 5, 200], "x2": [60, 635, 420], "y2": [140, 30, 260]})
W, H = 640, 480
feats = boxes.assign(
    w=lambda b: b.x2 - b.x1, h=lambda b: b.y2 - b.y1,
    log_area=lambda b: np.log((b.x2 - b.x1) * (b.y2 - b.y1)),                # площадь логнормальна (урок 16)
    aspect=lambda b: (b.x2 - b.x1) / (b.y2 - b.y1),
    rel_cy=lambda b: (b.y1 + b.y2) / 2 / H,                                     # объекты внизу кадра обычно ближе
    border_dist=lambda b: np.minimum.reduce([b.x1, b.y1, W - b.x2, H - b.y2]) / min(W, H),   # у края — часто обрезаны
)
print("\n", feats.round(3).to_string(index=False))

# ---------- 4. One-hot для редких категорий ----------
cats = np.array(["cam_a"] * 50 + ["cam_b"] * 30 + ["cam_rare"] * 1)[:, None]
ohe = OneHotEncoder(min_frequency=5, handle_unknown="infrequent_if_exist", sparse_output=False).fit(cats)
print("one-hot с объединением редких:", ohe.get_feature_names_out().tolist(), "| новая камера ->", ohe.transform([["cam_new"]]).tolist())
