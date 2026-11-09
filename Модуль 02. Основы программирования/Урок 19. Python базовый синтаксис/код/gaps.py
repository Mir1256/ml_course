"""Пробелы, типичные для C#-разработчика в Python. Каждый блок — короткая демонстрация с проверкой через assert."""
from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
import sys
import time
from typing import Iterator


# 1. Изменяемый аргумент по умолчанию создаётся ОДИН раз при определении функции
def add_box_bad(box, boxes=[]):
    boxes.append(box)
    return boxes


def add_box(box, boxes: list | None = None) -> list:
    boxes = [] if boxes is None else boxes
    boxes.append(box)
    return boxes


assert add_box_bad(1) == [1] and add_box_bad(2) == [1, 2]       # «утечка» состояния между вызовами
assert add_box(1) == [1] and add_box(2) == [2]

# 2. Присваивание — это новая ссылка на тот же объект, а не копия (как у ссылочных типов C#, но для ВСЕГО)
a = [[0, 0], [0, 0]]
b = a
shallow = copy.copy(a)                                           # новый внешний список, те же внутренние
deep = copy.deepcopy(a)
a[0][0] = 99
assert b[0][0] == 99 and shallow[0][0] == 99 and deep[0][0] == 0
grid = [[0] * 3] * 3                                             # ловушка: три ссылки на ОДИН список
grid[0][0] = 1
assert grid[1][0] == 1
grid_ok = [[0] * 3 for _ in range(3)]

# 3. Comprehension и генераторы: генератор ленив и одноразов
squares = [i * i for i in range(5)]
by_class = {c: len(c) for c in ("person", "car")}
gen = (i * i for i in range(5))
assert sum(gen) == 30 and sum(gen) == 0                          # второй проход — пусто: генератор исчерпан


def read_lines_lazily(n: int) -> Iterator[str]:
    """yield делает функцию генератором: строки выдаются по одной, память O(1)."""
    for i in range(n):
        yield f"line {i}"


assert next(read_lines_lazily(3)) == "line 0"
assert sys.getsizeof(i for i in range(10**6)) < 1000 < sys.getsizeof([i for i in range(10**6)])


# 4. *args / **kwargs и распаковка
def make_box(x, y, w, h, *, label="obj", **extra):              # после * — только именованные аргументы
    return {"bbox": (x, y, w, h), "label": label, **extra}


coords = (10, 20, 30, 40)
box = make_box(*coords, label="car", score=0.9)
assert box == {"bbox": coords, "label": "car", "score": 0.9}
first, *middle, last = [1, 2, 3, 4]
assert middle == [2, 3]


# 5. Исключения: EAFP («проще попросить прощения») вместо проверок заранее; else/finally
def parse_score(s: str) -> float | None:
    try:
        value = float(s)
    except ValueError:
        return None
    else:                                                        # выполняется, если исключения не было
        return min(max(value, 0.0), 1.0)


assert parse_score("0.7") == 0.7 and parse_score("abc") is None and parse_score("5") == 1.0


class AnnotationError(ValueError):
    """Свои исключения наследуют от встроенных — их можно ловить и по базовому классу."""


try:
    raise AnnotationError("bbox вне кадра")
except ValueError as e:
    assert isinstance(e, AnnotationError)


# 6. Контекстные менеджеры (аналог using в C#)
@contextmanager
def timer(name: str, log: list):
    t0 = time.perf_counter()
    try:
        yield
    finally:                                                     # выполнится даже при исключении внутри with
        log.append((name, time.perf_counter() - t0))


log: list = []
with timer("sum", log):
    sum(range(10_000))
assert log[0][0] == "sum"

# 7. pathlib вместо склейки строк
root = Path("data") / "coco" / "images"
assert root.name == "images" and root.parent.name == "coco" and Path("img_001.jpg").suffix == ".jpg"
assert Path("img_001.jpg").with_suffix(".txt").name == "img_001.txt"      # имя файла разметки YOLO

# 8. Встроенные функции, которые заменяют циклы
scores = [0.3, 0.9, 0.5]
labels = ["a", "b", "c"]
assert sorted(zip(scores, labels), reverse=True)[0] == (0.9, "b")
assert max(range(len(scores)), key=scores.__getitem__) == 1
assert any(s > 0.8 for s in scores) and not all(s > 0.4 for s in scores)
assert list(enumerate(labels, start=1))[0] == (1, "a")

# 9. Сравнения: == сравнивает значения, is — идентичность объекта
assert [1, 2] == [1, 2] and [1, 2] is not [1, 2]
x = None
assert x is None                                                 # для None — всегда is

# 10. match (структурное сопоставление, Python 3.10+)
def describe(ann: dict) -> str:
    match ann:
        case {"bbox": (x, y, w, h)} if w <= 0 or h <= 0:
            return "пустая рамка"
        case {"bbox": _, "label": str(label)}:
            return f"рамка {label}"
        case {"segmentation": _}:
            return "маска"
        case _:
            return "неизвестно"


assert describe({"bbox": (0, 0, 0, 5)}) == "пустая рамка"
assert describe({"bbox": (0, 0, 5, 5), "label": "dog"}) == "рамка dog"

# 11. Целочисленное деление и float: // округляет к минус бесконечности (в C# — к нулю)
assert -7 // 2 == -4 and int(-7 / 2) == -3 and 0.1 + 0.2 != 0.3
assert abs((0.1 + 0.2) - 0.3) < 1e-12

print("все 11 проверок пройдены")
