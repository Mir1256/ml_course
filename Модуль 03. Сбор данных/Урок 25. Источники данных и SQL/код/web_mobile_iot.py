"""Интернет, мобильные приложения, IoT: robots.txt, извлечение картинок из HTML, EXIF-ориентация, поток датчиков."""
from html.parser import HTMLParser
from io import BytesIO
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# ---------- 1. Интернет: сначала robots.txt и лицензия, потом скачивание ----------
ROBOTS = """
User-agent: *
Disallow: /private/
Crawl-delay: 2
"""
rp = RobotFileParser()
rp.parse(ROBOTS.splitlines())
for path in ("/gallery/cats.html", "/private/faces.html"):
    print(f"можно ли качать {path}: {rp.can_fetch('ml-course-bot', 'https://example.org' + path)}")
print("пауза между запросами, с:", rp.crawl_delay("ml-course-bot"))


class ImageLinks(HTMLParser):
    """Собираем src картинок и их alt — alt часто служит слабой (шумной) меткой класса."""
    def __init__(self, base: str):
        super().__init__()
        self.base, self.items = base, []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "img" and "src" in a:
            self.items.append({"url": urljoin(self.base, a["src"]), "alt": a.get("alt", "")})


html = '<html><body><img src="/img/1.jpg" alt="red car"><p>text</p><img src="2.png"><img alt="no src"></body></html>'
parser = ImageLinks("https://example.org/gallery/")
parser.feed(html)
print("найдено картинок:", parser.items)

# ---------- 2. Мобильные приложения: EXIF-ориентация ----------
# Телефон хранит пиксели «как снял сенсор» и пишет в EXIF тег Orientation, как их повернуть при показе.
# Если читать пиксели без учёта тега, рамки разметки (сделанные в программе, которая тег учла) не совпадут с картинкой.
pixels = np.zeros((40, 60, 3), np.uint8); pixels[:, :20] = 255         # «альбомный» снимок 60x40
exif = Image.Exif(); exif[0x0112] = 6                                   # 6 = «повернуть на 90° по часовой при показе»
buf = BytesIO(); Image.fromarray(pixels).save(buf, "JPEG", exif=exif); buf.seek(0)
img = Image.open(buf)
print(f"\nфото с телефона: сырые пиксели {img.size} (ш x в), Orientation = {img.getexif().get(0x0112)}")
print(f"после ImageOps.exif_transpose: {ImageOps.exif_transpose(img).size} — так картинку видел разметчик")

# ---------- 3. IoT: поток датчиков -> агрегаты по окнам ----------
rng = np.random.default_rng(0)
t = pd.date_range("2026-11-01 06:00", periods=6 * 60 * 60, freq="s")   # 6 часов, 1 замер в секунду
lux = np.clip(50 + 900 * np.sin(np.linspace(0, np.pi / 2, t.size)) + rng.normal(0, 30, t.size), 0, None)
lux[rng.random(t.size) < 0.002] = np.nan                               # одиночные пропуски передачи
lux[8000:8100] = np.nan                                                 # обрыв связи на 100 с
lux[5000:5030] = 0                                                      # датчик «залип» на 30 с
stream = pd.Series(lux, index=t, name="lux")
per_min = stream.resample("1min").agg(["mean", "min", "max", "count"])
print("\nосвещённость, окна по 1 минуте:\n", per_min.head(3).round(1).to_string())
print("минут с потерей > 5% замеров:", int((per_min["count"] < 57).sum()))
stuck = stream.rolling("10s").std() == 0
print("секунд с «залипшим» датчиком (std за 10 с = 0):", int(stuck.sum()))
