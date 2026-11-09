"""Pandas 3.x на аннотациях в формате COCO: JSON -> три таблицы -> merge -> groupby -> отчёты -> Parquet."""
import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def make_coco_like(n_images: int = 2000, seed: int = 0) -> dict:
    """Структура как у instances_val2017.json: images, annotations, categories."""
    rng = np.random.default_rng(seed)
    cats = [{"id": 1, "name": "person", "supercategory": "person"},
            {"id": 3, "name": "car", "supercategory": "vehicle"},
            {"id": 18, "name": "dog", "supercategory": "animal"},
            {"id": 44, "name": "bottle", "supercategory": "kitchen"}]
    images = [{"id": i, "file_name": f"{i:012d}.jpg", "width": int(w), "height": int(h)}
              for i, (w, h) in enumerate(zip(rng.choice([640, 500, 427], n_images), rng.choice([480, 375, 640], n_images)))]
    anns, ann_id = [], 0
    for img in images:
        for _ in range(rng.poisson(3)):                                   # число объектов на картинке — Пуассон (урок 15)
            cat = int(rng.choice([1, 3, 18, 44], p=[0.55, 0.2, 0.1, 0.15]))
            w, h = rng.lognormal(3.8, 0.8, size=2)
            x, y = rng.uniform(0, img["width"] - min(w, img["width"] - 1)), rng.uniform(0, img["height"] - min(h, img["height"] - 1))
            anns.append({"id": ann_id, "image_id": img["id"], "category_id": cat,
                         "bbox": [round(x, 1), round(y, 1), round(w, 1), round(h, 1)], "iscrowd": int(rng.random() < 0.01)})
            ann_id += 1
    return {"images": images, "annotations": anns, "categories": cats}


coco = make_coco_like()

# 1. JSON -> DataFrame. bbox — список в одной колонке: раскладываем в четыре
images = pd.DataFrame(coco["images"])
cats = pd.DataFrame(coco["categories"])
anns = pd.DataFrame(coco["annotations"])
anns[["x", "y", "w", "h"]] = pd.DataFrame(anns["bbox"].tolist(), index=anns.index)
anns = anns.drop(columns="bbox")
print(f"images {images.shape}, annotations {anns.shape}, categories {cats.shape}")
print("dtype строковой колонки (pandas 3: новый str-dtype):", images.file_name.dtype)

# 2. merge — аналог JOIN; validate ловит неожиданные дубли ключей
df = (anns
      .merge(cats.rename(columns={"id": "category_id", "name": "category"}), on="category_id", how="left", validate="many_to_one")
      .merge(images.rename(columns={"id": "image_id"}), on="image_id", how="left", validate="many_to_one"))

# 3. Method chaining: новые колонки через assign, без промежуточных переменных и без цепочечного присваивания
df = (df
      .assign(area=lambda d: d.w * d.h,
              rel_area=lambda d: d.w * d.h / (d.width * d.height),
              size=lambda d: pd.cut(d.w * d.h, [0, 32**2, 96**2, np.inf], labels=["small", "medium", "large"], right=False),
              out_of_frame=lambda d: (d.x + d.w > d.width) | (d.y + d.h > d.height))
      .astype({"category": "category", "supercategory": "category"}))
print("\nразмер по категориям COCO (доли):\n", df["size"].value_counts(normalize=True).round(3).to_string())

# 4. groupby + именованные агрегаты
per_class = (df.groupby("category", observed=True)
               .agg(n=("id", "size"), images=("image_id", "nunique"), median_area=("area", "median"),
                    small_share=("size", lambda s: (s == "small").mean()))
               .sort_values("n", ascending=False))
print("\n", per_class.round(3).to_string())

# 5. pivot_table: класс x размер
print("\n", pd.pivot_table(df, index="category", columns="size", values="id", aggfunc="count", observed=True).to_string())

# 6. Картинки без разметки — anti-join через indicator
per_image = images.rename(columns={"id": "image_id"}).merge(anns[["image_id"]].drop_duplicates(), how="left", indicator=True)
print(f"\nкартинок без объектов: {(per_image._merge == 'left_only').sum()} из {len(images)}")

# 7. Copy-on-Write: цепочечное присваивание больше НЕ меняет исходник
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    df["w"][df.out_of_frame] = 0                                     # так нельзя
print("цепочечное присваивание изменило df?", (df.loc[df.out_of_frame, "w"] == 0).all(), "| предупреждение:", type(w[0].message).__name__)
df.loc[df.out_of_frame, "w"] = (df.width - df.x)[df.out_of_frame]   # так правильно: один .loc
print("после .loc рамок за кадром по ширине:", int((df.x + df.w > df.width + 1e-9).sum()))

# 8. Память: category против str
print(f"\nпамять колонки category: str {df.category.astype(str).memory_usage(deep=True) / 1e3:.0f} КБ, category {df.category.memory_usage(deep=True) / 1e3:.0f} КБ")

# 9. Parquet: типы сохраняются, файл меньше JSON
with tempfile.TemporaryDirectory() as tmp:
    pq, js = Path(tmp) / "anns.parquet", Path(tmp) / "anns.json"
    df.to_parquet(pq)
    js.write_text(json.dumps(coco))
    back = pd.read_parquet(pq)
    print(f"Parquet {pq.stat().st_size / 1e3:.0f} КБ против JSON {js.stat().st_size / 1e3:.0f} КБ; dtype category сохранился: {back.category.dtype == 'category'}")
