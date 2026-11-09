"""Доверительные интервалы, тесты и бутстрап для сравнения моделей."""
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)


def wald_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """p̂ ± z sqrt(p̂(1-p̂)/n) — нормальное приближение (ЦПТ). Плохо при малых n и p̂ около 0 или 1."""
    p = k / n
    half = z * np.sqrt(p * (1 - p) / n)
    return p - half, p + half


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Интервал Уилсона: обращение z-теста, где дисперсия берётся при гипотетическом p, а не при p̂."""
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def bootstrap_ci(data: np.ndarray, stat, n_boot: int = 10_000, alpha: float = 0.05) -> tuple[float, float]:
    """Перцентильный бутстрап: пересэмплируем выборку с возвращением и берём квантили статистики."""
    idx = rng.integers(0, len(data), size=(n_boot, len(data)))
    boots = stat(data[idx])
    return tuple(np.quantile(boots, [alpha / 2, 1 - alpha / 2]))


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Сравнение двух моделей на ОДНОМ тесте: важны только несогласные пары.
    При H0 (модели одинаковы) каждая несогласная пара — «монетка» 50/50 -> биномиальный тест."""
    b = int(np.sum(correct_a & ~correct_b))            # A права, B нет
    c = int(np.sum(~correct_a & correct_b))            # B права, A нет
    if b + c == 0:
        return 1.0                                     # модели нигде не расходятся — различий нет
    return stats.binomtest(b, b + c, 0.5).pvalue


if __name__ == "__main__":
    # 1. Интервалы для accuracy: 92 верных из 100
    k, n = 92, 100
    print(f"Wald     : {np.round(wald_ci(k, n), 4)}")
    print(f"Wilson   : {np.round(wilson_ci(k, n), 4)}")
    print(f"Clopper  : {np.round(stats.binomtest(k, n).proportion_ci(method='exact'), 4)}")
    print(f"bootstrap: {np.round(bootstrap_ci(np.r_[np.ones(k), np.zeros(n - k)], lambda x: x.mean(axis=1)), 4)}")

    # 2. Покрытие: как часто 95%-интервал реально накрывает истинное p
    p_true, n_small, trials = 0.95, 30, 20_000
    ks = rng.binomial(n_small, p_true, size=trials)
    cover = lambda ci: np.mean([lo <= p_true <= hi for lo, hi in (ci(kk, n_small) for kk in ks)])
    print(f"\nпокрытие при n={n_small}, p={p_true}: Wald {cover(wald_ci):.3f}, Wilson {cover(wilson_ci):.3f} (цель 0.95)")

    # 3. Две модели на одних и тех же 500 картинках
    n_test = 500
    hard = rng.random(n_test) < 0.2                    # трудные картинки — обе модели чаще ошибаются на них
    correct_a = rng.random(n_test) < np.where(hard, 0.55, 0.95)
    correct_b = np.where(rng.random(n_test) < 0.9, correct_a, rng.random(n_test) < np.where(hard, 0.65, 0.96))
    print(f"\naccuracy A = {correct_a.mean():.3f}, B = {correct_b.mean():.3f}")
    print(f"McNemar (парный тест) p-value = {mcnemar_exact(correct_a, correct_b):.4f}")
    two_prop = stats.ttest_ind(correct_a.astype(float), correct_b.astype(float)).pvalue
    print(f"непарный t-тест (НЕВЕРНО для одного теста) p-value = {two_prop:.4f}")
    diff = (correct_b.astype(float) - correct_a.astype(float))[:, None]
    lo, hi = bootstrap_ci(diff[:, 0], lambda x: x.mean(axis=1))
    print(f"парный бутстрап, 95% ДИ разницы B - A: [{lo:.4f}, {hi:.4f}]")

    # 4. p-value при верной H0 распределено равномерно -> 5% ложных «открытий» при α = 0.05
    pvals = [stats.ttest_ind(rng.normal(size=30), rng.normal(size=30)).pvalue for _ in range(5_000)]
    print(f"\nдоля p < 0.05 при верной H0: {np.mean(np.array(pvals) < 0.05):.3f}")
    print(f"20 независимых тестов: P(хотя бы одно ложное открытие) = {1 - 0.95**20:.3f}; поправка Бонферрони α/20 = {0.05 / 20}")

    # 5. Перестановочный тест для разницы средних (например, IoU днём и ночью)
    iou_day, iou_night = rng.beta(8, 2, size=60), rng.beta(7, 2.5, size=40)
    res = stats.permutation_test((iou_day, iou_night), lambda a, b: a.mean() - b.mean(), n_resamples=20_000, random_state=0)
    pooled = np.r_[iou_day, iou_night]
    obs = iou_day.mean() - iou_night.mean()
    perm = np.array([(lambda p: p[:60].mean() - p[60:].mean())(rng.permutation(pooled)) for _ in range(20_000)])
    print(f"\nперестановочный тест: свой p = {np.mean(np.abs(perm) >= abs(obs)):.4f}, scipy p = {res.pvalue:.4f}")

    # 6. Сколько тестовых картинок нужно: ширина ДИ ±1 п.п. при accuracy ≈ 0.9
    print(f"нужно картинок для ±1 п.п.: {int(np.ceil(1.96**2 * 0.9 * 0.1 / 0.01**2))}")
