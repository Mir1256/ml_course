"""SQL на базе разметки: JOIN, GROUP BY / HAVING, анти-JOIN, CTE, оконные функции, индексы. Сверка с pandas."""
import sqlite3
import time

import numpy as np
import pandas as pd

SCHEMA = """
CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE annotators (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE images (
    id INTEGER PRIMARY KEY, file_name TEXT NOT NULL, width INTEGER, height INTEGER,
    camera TEXT, captured_at TEXT                                  -- ISO-дата; в SQLite нет отдельного типа даты
);
CREATE TABLE annotations (
    id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images(id),
    category_id INTEGER NOT NULL REFERENCES categories(id),
    annotator_id INTEGER REFERENCES annotators(id),
    x REAL, y REAL, w REAL, h REAL,
    score REAL                                                     -- уверенность модели-предразметчика
);
"""


def build_db(n_images: int = 3000, seed: int = 0) -> sqlite3.Connection:
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO categories VALUES (?, ?)", [(1, "person"), (2, "car"), (3, "helmet"), (4, "forklift")])
    con.executemany("INSERT INTO annotators VALUES (?, ?)", [(1, "Аня"), (2, "Борис"), (3, "Вика")])
    cams = np.array(["cam_gate", "cam_dock", "cam_yard"])
    images = [(i, f"{i:06d}.jpg", 1280, 720, str(rng.choice(cams)), f"2026-10-{1 + i % 30:02d}T{i % 24:02d}:00:00")
              for i in range(n_images)]
    con.executemany("INSERT INTO images VALUES (?, ?, ?, ?, ?, ?)", images)
    rows, aid = [], 0
    for img_id in range(n_images):
        for _ in range(rng.poisson(2.5)):
            w, h = rng.lognormal(4, 0.6, size=2)
            rows.append((aid, img_id, int(rng.choice([1, 2, 3, 4], p=[0.5, 0.2, 0.2, 0.1])), int(rng.integers(1, 4)),
                         float(rng.uniform(0, 1280 - w)), float(rng.uniform(0, 720 - h)), float(w), float(h), float(rng.beta(5, 2))))
            aid += 1
    con.executemany("INSERT INTO annotations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.commit()
    return con


QUERIES = {
    # JOIN + GROUP BY + HAVING: классы, у которых больше 500 объектов
    "per_class": """
        SELECT c.name AS category, COUNT(*) AS n, ROUND(AVG(a.w * a.h), 1) AS mean_area
        FROM annotations a JOIN categories c ON c.id = a.category_id
        GROUP BY c.name HAVING COUNT(*) > 500 ORDER BY n DESC""",
    # LEFT JOIN + IS NULL — анти-JOIN: картинки без единой аннотации
    "empty_images": """
        SELECT i.id FROM images i LEFT JOIN annotations a ON a.image_id = i.id
        WHERE a.id IS NULL ORDER BY i.id""",
    # CTE + оконная функция: топ-2 объекта по уверенности на каждой картинке
    "top2_per_image": """
        WITH ranked AS (
            SELECT image_id, id, score,
                   ROW_NUMBER() OVER (PARTITION BY image_id ORDER BY score DESC) AS rnk
            FROM annotations)
        SELECT image_id, id, score FROM ranked WHERE rnk <= 2 ORDER BY image_id, rnk""",
    # Оконная агрегация: доля каждого разметчика в объёме работы и нарастающий итог по дням
    "annotator_share": """
        SELECT an.name, COUNT(*) AS n,
               ROUND(1.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 3) AS share
        FROM annotations a JOIN annotators an ON an.id = a.annotator_id
        GROUP BY an.name ORDER BY n DESC""",
    "daily_cumulative": """
        SELECT substr(i.captured_at, 1, 10) AS day, COUNT(*) AS n,
               SUM(COUNT(*)) OVER (ORDER BY substr(i.captured_at, 1, 10)) AS cumulative
        FROM annotations a JOIN images i ON i.id = a.image_id
        GROUP BY day ORDER BY day LIMIT 5""",
    # Каски на людях: картинки, где есть person, но нет helmet (аудит правил техники безопасности)
    "person_without_helmet": """
        SELECT COUNT(DISTINCT a.image_id) AS n_images FROM annotations a
        WHERE a.category_id = 1
          AND NOT EXISTS (SELECT 1 FROM annotations b WHERE b.image_id = a.image_id AND b.category_id = 3)""",
}


if __name__ == "__main__":
    con = build_db()
    for name in ("per_class", "annotator_share", "daily_cumulative", "person_without_helmet"):
        print(f"\n-- {name}\n", pd.read_sql_query(QUERIES[name], con).to_string(index=False))

    # Сверка SQL и pandas на тех же данных
    anns = pd.read_sql_query("SELECT * FROM annotations", con)
    imgs = pd.read_sql_query("SELECT * FROM images", con)
    sql_empty = pd.read_sql_query(QUERIES["empty_images"], con).id.to_numpy()
    pd_empty = np.sort(imgs.loc[~imgs.id.isin(anns.image_id), "id"].to_numpy())
    print("\nпустые картинки: SQL == pandas:", np.array_equal(sql_empty, pd_empty), f"({len(sql_empty)} шт.)")
    sql_top = pd.read_sql_query(QUERIES["top2_per_image"], con)
    pd_top = (anns.sort_values(["image_id", "score"], ascending=[True, False])
                  .groupby("image_id").head(2)[["image_id", "id", "score"]].reset_index(drop=True))
    print("топ-2 на картинку: SQL == pandas:", sql_top.equals(pd_top))

    # Индекс: поиск по image_id без индекса — полный просмотр таблицы, с индексом — B-дерево, O(log n)
    q = "SELECT COUNT(*) FROM annotations WHERE image_id = ?"
    plan = lambda: con.execute("EXPLAIN QUERY PLAN " + q, (5,)).fetchall()[0][-1]
    t0 = time.perf_counter(); [con.execute(q, (i,)).fetchone() for i in range(500)]; t_scan = time.perf_counter() - t0
    print(f"\nбез индекса: план «{plan()}», 500 запросов за {t_scan * 1e3:.0f} мс")
    con.execute("CREATE INDEX idx_ann_image ON annotations(image_id)")
    t0 = time.perf_counter(); [con.execute(q, (i,)).fetchone() for i in range(500)]; t_idx = time.perf_counter() - t0
    print(f"с индексом:  план «{plan()}», 500 запросов за {t_idx * 1e3:.1f} мс (x{t_scan / t_idx:.0f})")
