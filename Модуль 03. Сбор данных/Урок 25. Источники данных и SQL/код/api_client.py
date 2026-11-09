"""Надёжный клиент HTTP API: таймауты, повторы с экспоненциальной задержкой, пагинация, кэш на диске."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import requests

RETRY_STATUS = {429, 500, 502, 503, 504}          # «попробуй позже»: перегрузка, лимит запросов, сбой шлюза


def get_json(session: requests.Session, url: str, params: dict | None = None, *, retries: int = 4,
             backoff: float = 0.5, timeout: float = 10.0, sleep=time.sleep) -> dict:
    """GET с повторами. Задержка base * 2^попытка + случайный джиттер, чтобы клиенты не «били» сервер синхронно."""
    for attempt in range(retries + 1):
        try:
            r = session.get(url, params=params, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries:
                raise
        else:
            if r.status_code not in RETRY_STATUS:
                r.raise_for_status()                     # 4xx (кроме 429) — ошибка запроса, повторять бессмысленно
                return r.json()
            if attempt == retries:
                r.raise_for_status()
            if "Retry-After" in r.headers:               # сервер сам говорит, сколько ждать
                sleep(float(r.headers["Retry-After"]))
                continue
        sleep(backoff * 2**attempt + random.uniform(0, backoff))
    raise RuntimeError("недостижимо")


def fetch_all_pages(session: requests.Session, url: str, page_size: int = 100, **kw) -> list[dict]:
    """Пагинация по offset/limit: запрашиваем страницы, пока не придёт неполная."""
    items, offset = [], 0
    while True:
        page = get_json(session, url, {"offset": offset, "limit": page_size}, **kw)["items"]
        items.extend(page)
        if len(page) < page_size:
            return items
        offset += page_size


def cached(cache_dir: Path):
    """Кэш ответов на диске: ключ — хэш URL и параметров. Повторный запуск ноутбука не долбит API."""
    def wrap(fn):
        def inner(session, url, params=None, **kw):
            key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()[:16]
            path = cache_dir / f"{key}.json"
            if path.exists():
                return json.loads(path.read_text())
            data = fn(session, url, params, **kw)
            cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
            return data
        return inner
    return wrap


def open_meteo_hourly(session: requests.Session, lat: float, lon: float, days: int = 1) -> dict:
    """Пример настоящего публичного API без ключа: почасовая температура и облачность (освещённость кадров зависит от погоды)."""
    return get_json(session, "https://api.open-meteo.com/v1/forecast",
                    {"latitude": lat, "longitude": lon, "hourly": "temperature_2m,cloud_cover", "forecast_days": days})


if __name__ == "__main__":
    import pandas as pd

    with requests.Session() as s:
        s.headers["User-Agent"] = "ml-course-lesson-25"
        try:
            data = open_meteo_hourly(s, 55.75, 37.62)
            df = pd.DataFrame(data["hourly"]).assign(time=lambda d: pd.to_datetime(d.time))
            print(df.head(4).to_string(index=False))
            print("единицы:", data["hourly_units"])
        except requests.RequestException as e:
            print("сеть недоступна, пример пропущен:", type(e).__name__)
