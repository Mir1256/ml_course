"""Одна таблица аннотаций — семь форматов: размер, скорость, сохранность типов; плюс сильные стороны Parquet."""
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

rng = np.random.default_rng(0)
n = 200_000
df = pd.DataFrame({
    "image_id": [f"{i:06d}" for i in rng.integers(0, 30_000, n)],               # строка с ведущими нулями!
    "category": pd.Categorical(rng.choice(["person", "car", "helmet", "forklift"], n, p=[0.5, 0.2, 0.2, 0.1])),
    "x": rng.uniform(0, 1280, n).astype(np.float32), "y": rng.uniform(0, 720, n).astype(np.float32),
    "w": rng.lognormal(4, 0.6, n).astype(np.float32), "h": rng.lognormal(4, 0.6, n).astype(np.float32),
    "score": rng.beta(5, 2, n), "verified": rng.random(n) < 0.3,
    "captured_at": pd.Timestamp("2026-10-01", tz="UTC") + pd.to_timedelta(rng.integers(0, 30 * 86400, n), unit="s"),
    "comment": np.where(rng.random(n) < 0.001, 'перекрыт, "частично"; см. кадр 12', None),   # кавычки и ; в тексте
})

writers = {
    "csv": (lambda p: df.to_csv(p, index=False), lambda p: pd.read_csv(p)),
    "jsonl": (lambda p: df.to_json(p, orient="records", lines=True, date_format="iso"), lambda p: pd.read_json(p, lines=True)),
    "parquet (snappy)": (lambda p: df.to_parquet(p, compression="snappy"), pd.read_parquet),
    "parquet (zstd)": (lambda p: df.to_parquet(p, compression="zstd"), pd.read_parquet),
    "feather": (lambda p: df.to_feather(p), pd.read_feather),
    "pickle": (lambda p: df.to_pickle(p), pd.read_pickle),
    # Excel не умеет даты с часовым поясом (ValueError) — приходится терять tz перед записью
    "excel (10k строк)": (lambda p: df.head(10_000).assign(captured_at=lambda d: d.captured_at.dt.tz_localize(None))
                          .to_excel(p, index=False), lambda p: pd.read_excel(p)),
}

with tempfile.TemporaryDirectory() as tmp:
    rows = []
    for name, (write, read) in writers.items():
        path = Path(tmp) / name.split()[0]
        t0 = time.perf_counter(); write(path); t_w = time.perf_counter() - t0
        t0 = time.perf_counter(); back = read(path); t_r = time.perf_counter() - t0
        ref = df.head(len(back))
        rows.append({"формат": name, "МБ": path.stat().st_size / 1e6, "запись, с": t_w, "чтение, с": t_r,
                     "image_id цел": back.image_id.astype(str).equals(ref.image_id.astype(str)),
                     "типы целы": bool((back.dtypes.astype(str) == ref.dtypes.astype(str)).all())})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # Почему типы «плывут» в CSV: всё — текст, тип угадывается заново
    back = pd.read_csv(Path(tmp) / "csv")
    print("\nCSV после чтения: image_id ->", back.image_id.dtype, "(ведущие нули потеряны:", back.image_id.iloc[0], ")",
          "| category ->", back.category.dtype, "| captured_at ->", back.captured_at.dtype)
    print("лечение: dtype и parse_dates при чтении:",
          pd.read_csv(Path(tmp) / "csv", dtype={"image_id": str, "category": "category"}, parse_dates=["captured_at"]).dtypes.to_dict()["image_id"])

    # Сильные стороны Parquet: читаем только нужные колонки и только нужные строки (predicate pushdown)
    p = Path(tmp) / "big.parquet"
    df.sort_values("captured_at").to_parquet(p, row_group_size=20_000)
    meta = pq.ParquetFile(p).metadata
    print(f"\nParquet: {meta.num_row_groups} row groups, у каждой в метаданных min/max колонок")
    t0 = time.perf_counter(); full = pd.read_parquet(p); t_full = time.perf_counter() - t0
    t0 = time.perf_counter()
    part = ds.dataset(p).to_table(columns=["image_id", "score"],
                                  filter=ds.field("captured_at") >= pd.Timestamp("2026-10-29", tz="UTC")).to_pandas()
    t_part = time.perf_counter() - t0
    print(f"всё: {t_full * 1e3:.0f} мс | 2 колонки и последние 2 дня: {t_part * 1e3:.0f} мс, строк {len(part)}")

# Pickle исполняет код при загрузке — никогда не открывай чужие .pkl
class Evil:
    def __reduce__(self):
        return (print, ("!!! этот код выполнился при pickle.loads — мог быть любой !!!",))

pickle.loads(pickle.dumps(Evil()))
