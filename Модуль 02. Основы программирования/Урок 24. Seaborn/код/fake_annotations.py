"""Синтетические аннотации детекции, похожие на COCO: дисбаланс классов, логнормальные площади рамок, ошибки разметки.

Используется в уроках 16-17, пока мы не дошли до настоящего COCO (мини-проект 2).
"""
import numpy as np
import pandas as pd


def make_annotations(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    classes = np.array(["person", "car", "chair", "bottle", "dog", "traffic light"])
    class_p = np.array([0.45, 0.2, 0.12, 0.1, 0.08, 0.05])                  # сильный дисбаланс
    median_side = {"person": 90, "car": 70, "chair": 50, "bottle": 25, "dog": 80, "traffic light": 18}

    cls = rng.choice(classes, size=n, p=class_p)
    side = np.array([median_side[c] for c in cls]) * rng.lognormal(0, 0.7, size=n)   # множитель масштаба
    aspect = rng.lognormal(0, 0.35, size=n)                                  # w / h
    w = side * np.sqrt(aspect)
    h = side / np.sqrt(aspect)
    img_w, img_h = 640, 480
    x = rng.uniform(0, img_w - np.minimum(w, img_w - 1))
    y = rng.uniform(0, img_h - np.minimum(h, img_h - 1))

    df = pd.DataFrame({"image_id": rng.integers(0, n // 7, size=n), "category": cls,
                       "x": x, "y": y, "w": w, "h": h})
    bad = rng.choice(n, size=n // 200, replace=False)                       # 0.5% испорченной разметки
    df.loc[bad, "w"] = df.loc[bad, "w"] * rng.choice([0.02, 15.0], size=bad.size)
    df["area"] = df.w * df.h
    df["aspect"] = df.w / df.h
    df["night"] = rng.random(n) < 0.3
    df["score"] = np.clip(rng.normal(0.75 - 0.15 * df.night - 0.1 * (df.area < 32**2), 0.12), 0, 1)
    return df
