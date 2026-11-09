"""Очистка картинок и разметки: битые файлы, почти-дубликаты (перцептивный хэш), шум меток (confident learning)."""
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits, load_sample_images
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


# ---------- 1. Битые картинки: обрезанный при скачивании JPEG ----------
def is_readable(path: Path) -> tuple[bool, str]:
    """verify() проверяет структуру файла, load() — что пиксели действительно декодируются."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
        return True, "ok"
    except Exception as e:                       # PIL бросает разные исключения: OSError, SyntaxError, ...
        return False, f"{type(e).__name__}: {str(e)[:50]}"


# ---------- 2. Перцептивный хэш dHash: похожие картинки -> близкие хэши ----------
def dhash(img: np.ndarray, size: int = 8) -> np.ndarray:
    """Уменьшаем до (size, size+1) в оттенках серого и сравниваем соседей по горизонтали: 64 бита «градиентов»."""
    gray = Image.fromarray(img).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(gray, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel()


def hamming_matrix(H: np.ndarray) -> np.ndarray:
    """Попарные расстояния Хэмминга между битовыми хэшами (n, 64) — без циклов: XOR через неравенство + сумма."""
    return (H[:, None, :] != H[None, :, :]).sum(axis=2)


if __name__ == "__main__":
    china, flower = load_sample_images().images
    with tempfile.TemporaryDirectory() as tmp:
        good, broken = Path(tmp) / "good.jpg", Path(tmp) / "broken.jpg"
        Image.fromarray(china).save(good, quality=90)
        broken.write_bytes(good.read_bytes()[: good.stat().st_size // 2])      # «недокачанный» файл
        for p in (good, broken):
            print(f"{p.name}: {is_readable(p)}")

    rng = np.random.default_rng(0)
    variants = {
        "china": china, "china_jpeg_q30": None, "china_brighter": np.clip(china.astype(int) + 25, 0, 255).astype(np.uint8),
        "china_crop95": china[10:-10, 15:-15], "china_noise": np.clip(china + rng.normal(0, 8, china.shape), 0, 255).astype(np.uint8),
        "china_flipped": china[:, ::-1], "flower": flower,
    }
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "q30.jpg"
        Image.fromarray(china).save(p, quality=30)
        variants["china_jpeg_q30"] = np.asarray(Image.open(p))
    names = list(variants)
    D = hamming_matrix(np.stack([dhash(variants[k]) for k in names]))
    print("\nрасстояние Хэмминга dHash до оригинала (из 64 бит):")
    for k, d in zip(names, D[0]):
        print(f"  {k:<16} {d:>2}  {'почти-дубликат' if d <= 10 else ''}")

    # ---------- 3. Шум меток: 5% неверных меток на digits; ищем их по вероятностям out-of-fold ----------
    X, y_true = load_digits(return_X_y=True)
    y_noisy = y_true.copy()
    flip = rng.random(len(y_true)) < 0.05
    y_noisy[flip] = (y_true[flip] + rng.integers(1, 10, flip.sum())) % 10      # гарантированно другой класс
    proba = cross_val_predict(LogisticRegression(max_iter=2000), X / 16, y_noisy, cv=5, method="predict_proba")
    # Confident learning (упрощённо): порог класса = средняя уверенность модели на объектах, помеченных этим классом
    thresholds = np.array([proba[y_noisy == k, k].mean() for k in range(10)])
    confident = proba >= thresholds                                            # (n, 10): класс k «уверенно предсказан»
    best_other = np.where(confident, proba, -1)
    best_other[np.arange(len(y_noisy)), y_noisy] = -1
    suspect = (best_other.max(axis=1) > proba[np.arange(len(y_noisy)), y_noisy]) & confident.any(axis=1)
    tp = (suspect & flip).sum()
    print(f"\nиспорчено меток: {flip.sum()} | помечено подозрительными: {suspect.sum()} | из них действительно ошибки: {tp}"
          f" (precision {tp / suspect.sum():.2f}, recall {tp / flip.sum():.2f})")
