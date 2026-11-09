"""Описательная статистика с нуля и через pandas/scipy на синтетических аннотациях детекции."""
import numpy as np
import pandas as pd
from scipy import stats

from fake_annotations import make_annotations


def quantile(x: np.ndarray, q: float) -> float:
    """Квантиль с линейной интерполяцией между соседними порядковыми статистиками (как numpy по умолчанию)."""
    s = np.sort(x)
    pos = q * (len(s) - 1)
    lo, hi = int(np.floor(pos)), int(np.ceil(pos))
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def mad(x: np.ndarray) -> float:
    """Медианное абсолютное отклонение — робастная мера разброса."""
    return float(np.median(np.abs(x - np.median(x))))


def skewness(x: np.ndarray) -> float:
    z = (x - x.mean()) / x.std()
    return float(np.mean(z**3))


def iqr_outliers(x: np.ndarray, k: float = 1.5) -> np.ndarray:
    q1, q3 = quantile(x, 0.25), quantile(x, 0.75)
    return (x < q1 - k * (q3 - q1)) | (x > q3 + k * (q3 - q1))


def robust_z(x: np.ndarray) -> np.ndarray:
    """(x - медиана) / (1.4826 MAD): для нормальных данных совпадает с обычным z, но не «ломается» от выбросов."""
    return (x - np.median(x)) / (1.4826 * mad(x))


if __name__ == "__main__":
    df = make_annotations()
    area = df.area.to_numpy()

    print("доли классов:\n", df.category.value_counts(normalize=True).round(3).to_string())
    print(f"\nплощадь рамки: среднее {area.mean():.0f}, медиана {np.median(area):.0f}  <- среднее >> медианы: правый хвост")
    print(f"асимметрия: своя {skewness(area):.2f}, scipy {stats.skew(area):.2f} | у log(площади): {stats.skew(np.log(area)):.2f}")
    print(f"квантили 25/50/75/99%: {[round(quantile(area, q)) for q in (0.25, 0.5, 0.75, 0.99)]} | numpy: {np.quantile(area, [0.25, 0.5, 0.75, 0.99]).round().astype(int).tolist()}")
    print(f"MAD своя {mad(area):.1f} | scipy {stats.median_abs_deviation(area):.1f}")

    small = (area < 32**2).mean()
    print(f"\nмелких объектов (< 32x32, как в COCO) {small:.1%}")

    # Выбросы и устойчивость статистик
    aspect = df.aspect.to_numpy()
    print(f"\nвыбросов aspect по IQR: {iqr_outliers(np.log(aspect)).sum()}, по |robust z| > 5: {(np.abs(robust_z(np.log(aspect))) > 5).sum()}")
    clean = aspect[~iqr_outliers(np.log(aspect))]
    print(f"aspect: среднее {aspect.mean():.3f} -> без выбросов {clean.mean():.3f} | медиана {np.median(aspect):.3f} -> {np.median(clean):.3f}")

    # Описание по группам — основной инструмент EDA разметки
    summary = df.groupby("category").agg(n=("area", "size"), area_median=("area", "median"),
                                         small_share=("area", lambda a: (a < 32**2).mean()),
                                         score_mean=("score", "mean")).sort_values("n", ascending=False)
    print("\n", summary.round(3).to_string())

    # Пирсон против Спирмена: нелинейная монотонная связь и выбросы
    r_p = np.corrcoef(df.area, df.score)[0, 1]
    r_s = stats.spearmanr(df.area, df.score).statistic
    print(f"\ncorr(площадь, уверенность): Пирсон {r_p:.3f}, Спирмен {r_s:.3f}")
    print("\npandas describe() по площади:\n", df.area.describe().round(1).to_string())
