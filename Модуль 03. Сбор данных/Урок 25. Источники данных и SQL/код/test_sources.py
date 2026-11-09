"""Тесты клиента API на локальном фейковом сервере — без интернета, детерминированно."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
import requests

from api_client import cached, fetch_all_pages, get_json
from sql_practice import QUERIES, build_db


class FakeAPI(BaseHTTPRequestHandler):
    calls = {"flaky": 0}

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: int(v[0]) for k, v in parse_qs(url.query).items()}
        if url.path == "/flaky":                       # первые два запроса — 503
            FakeAPI.calls["flaky"] += 1
            if FakeAPI.calls["flaky"] <= 2:
                return self._send(503, {"error": "busy"})
            return self._send(200, {"ok": True})
        if url.path == "/items":                       # 250 элементов, отдаём страницами
            data = list(range(250))[q["offset"]:q["offset"] + q["limit"]]
            return self._send(200, {"items": data})
        return self._send(404, {"error": "nope"})

    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, *args):                      # не засоряем вывод pytest
        pass


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), FakeAPI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_retry_then_success(server):
    sleeps = []
    with requests.Session() as s:
        assert get_json(s, server + "/flaky", sleep=sleeps.append) == {"ok": True}
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]           # экспоненциальный рост задержки


def test_404_is_not_retried(server):
    sleeps = []
    with requests.Session() as s, pytest.raises(requests.HTTPError):
        get_json(s, server + "/missing", sleep=sleeps.append)
    assert sleeps == []


def test_pagination(server):
    with requests.Session() as s:
        assert fetch_all_pages(s, server + "/items", page_size=100) == list(range(250))


def test_cache(tmp_path, server):
    calls = []
    fn = cached(tmp_path)(lambda s, url, params=None: calls.append(1) or {"v": 1})
    with requests.Session() as s:
        fn(s, server, {"a": 1}); fn(s, server, {"a": 1}); fn(s, server, {"a": 2})
    assert len(calls) == 2


def test_sql_anti_join_and_window():
    con = build_db(n_images=200)
    empty = {r[0] for r in con.execute(QUERIES["empty_images"])}
    with_ann = {r[0] for r in con.execute("SELECT DISTINCT image_id FROM annotations")}
    assert empty.isdisjoint(with_ann) and len(empty | with_ann) == 200
    top = con.execute(QUERIES["top2_per_image"]).fetchall()
    per_image = {}
    for img, _, _ in top:
        per_image[img] = per_image.get(img, 0) + 1
    assert max(per_image.values()) <= 2
