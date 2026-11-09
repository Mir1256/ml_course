"""groupby с нуля на NumPy: split-apply-combine через np.unique(return_inverse) и np.bincount."""
import time

import numpy as np
import pandas as pd


def group_mean(keys: np.ndarray, values: np.ndarray):
    """1) split: каждому ключу — номер группы; 2) apply: суммы и счётчики через bincount; 3) combine: деление."""
    uniq, inverse = np.unique(keys, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    return uniq, sums / counts


def group_median(keys: np.ndarray, values: np.ndarray):
    """Медиану через bincount не получить: сортируем по (группа, значение) и берём середину каждой группы."""
    uniq, inverse = np.unique(keys, return_inverse=True)
    order = np.lexsort((values, inverse))
    v, g = values[order], inverse[order]
    starts = np.searchsorted(g, np.arange(len(uniq)))
    counts = np.bincount(g)
    lo = v[starts + (counts - 1) // 2]
    hi = v[starts + counts // 2]
    return uniq, (lo + hi) / 2


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 2_000_000
    keys = rng.integers(0, 1000, size=n)
    vals = rng.lognormal(size=n)

    t0 = time.perf_counter(); u, m = group_mean(keys, vals); t_np = time.perf_counter() - t0
    t0 = time.perf_counter(); ref = pd.Series(vals).groupby(keys).mean(); t_pd = time.perf_counter() - t0
    print(f"group mean: NumPy {t_np:.3f} с, pandas {t_pd:.3f} с, совпадают: {np.allclose(m, ref.to_numpy()) and np.array_equal(u, ref.index)}")

    u, med = group_median(keys, vals)
    print("group median совпадает с pandas:", np.allclose(med, pd.Series(vals).groupby(keys).median().to_numpy()))
