"""Основы вероятности: точные формулы против Монте-Карло, условная вероятность из таблицы, независимость."""
from math import comb, prod

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)


def birthday_exact(n: int, N: int) -> float:
    """P(хотя бы одно совпадение среди n равновероятных значений из N) = 1 - N(N-1)...(N-n+1)/N^n."""
    return 1 - prod((N - i) / N for i in range(n))


def birthday_mc(n: int, N: int, trials: int = 20_000) -> float:
    draws = rng.integers(0, N, size=(trials, n))
    s = np.sort(draws, axis=1)
    return float(np.mean(np.any(s[:, 1:] == s[:, :-1], axis=1)))


def at_least_one(p: float, n: int) -> float:
    """P(хотя бы одно событие из n независимых, каждое с вероятностью p) = 1 - (1-p)^n."""
    return 1 - (1 - p) ** n


if __name__ == "__main__":
    for n in (23, 50):
        print(f"дни рождения, n={n}: точно {birthday_exact(n, 365):.4f}, Монте-Карло {birthday_mc(n, 365):.4f}")
    print(f"хэши 32 бита, 100 000 кадров: P(коллизия) = {birthday_exact(100_000, 2**32):.3f}")

    # Ложные срабатывания: детектор ошибается на кадре с p = 0.001, 10 000 кадров за смену
    p, n = 0.001, 10_000
    sim = np.mean(np.any(rng.random((2_000, n)) < p, axis=1))
    print(f"P(хотя бы одно ложное срабатывание за смену): формула {at_least_one(p, n):.5f}, симуляция {sim:.5f}")
    print(f"ожидаемое число ложных срабатываний: {p * n:.0f}")

    # Закон больших чисел: частота -> вероятность. Одна выборка может случайно «попасть»,
    # поэтому усредняем |ошибку| по 200 повторам; теория: sqrt(p(1-p)/n) * sqrt(2/pi)
    for k in (100, 1_000, 10_000, 100_000):
        freqs = (rng.random((200, k)) < 0.3).mean(axis=1)
        theory = np.sqrt(0.3 * 0.7 / k) * np.sqrt(2 / np.pi)
        print(f"n={k:>7}: средняя |ошибка| частоты {np.abs(freqs - 0.3).mean():.5f}, теория {theory:.5f}")

    # Условная вероятность из таблицы результатов детектора
    n_det = 5_000
    night = rng.random(n_det) < 0.3
    small = rng.random(n_det) < 0.4
    p_correct = 0.92 - 0.25 * night - 0.15 * small                  # ночью и на мелких объектах хуже
    df = pd.DataFrame({"night": night, "small": small, "correct": rng.random(n_det) < p_correct})
    print("\nP(correct) =", round(df.correct.mean(), 3))
    print("P(correct | night):\n", df.groupby("night").correct.mean().round(3).to_string())
    print("P(correct | night, small):\n", df.groupby(["night", "small"]).correct.mean().round(3).to_string())

    # Формула полной вероятности: P(correct) = Σ P(correct | night=v) P(night=v)
    total = sum(df[df.night == v].correct.mean() * (df.night == v).mean() for v in (False, True))
    print("полная вероятность:", round(total, 3), "== прямой подсчёт:", round(df.correct.mean(), 3))

    # Независимость: P(A ∩ B) == P(A) P(B)?
    print(f"\nP(night ∩ small) = {(df.night & df.small).mean():.4f}, P(night) P(small) = {df.night.mean() * df.small.mean():.4f}")
    print("сочетаний 5 кадров из 100:", comb(100, 5))
