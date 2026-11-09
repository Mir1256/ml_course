"""ООП для ML-кода: dataclass-рамка, датасет с протоколом последовательности, трансформации через __call__ и Protocol."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Iterator, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class Box:
    """Неизменяемая рамка в пикселях. frozen -> хэшируема, можно класть в set; slots -> меньше памяти."""
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = "obj"

    def __post_init__(self):                       # валидация сразу при создании
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"вырожденная рамка: {self}")

    @property
    def area(self) -> float:                       # вычисляемый атрибут: box.area без скобок
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @classmethod
    def from_yolo(cls, cx, cy, w, h, img_w, img_h, label="obj") -> Box:   # альтернативный конструктор
        return cls((cx - w / 2) * img_w, (cy - h / 2) * img_h, (cx + w / 2) * img_w, (cy + h / 2) * img_h, label)


@dataclass
class Sample:
    image: np.ndarray                              # (H, W, 3) uint8
    boxes: list[Box] = field(default_factory=list) # НЕ boxes: list = [] — урок 19, изменяемое значение по умолчанию


# ---------- Трансформации: структурная типизация (Protocol) вместо наследования ----------
@runtime_checkable
class Transform(Protocol):
    """Всё, что можно вызвать как transform(sample) -> sample, — трансформация. Наследоваться не обязательно."""
    def __call__(self, sample: Sample) -> Sample: ...


class HFlip:
    def __call__(self, sample: Sample) -> Sample:
        w = sample.image.shape[1]
        boxes = [replace(b, x1=w - b.x2, x2=w - b.x1) for b in sample.boxes]   # replace — копия frozen-объекта с изменениями
        return Sample(sample.image[:, ::-1].copy(), boxes)


class Compose:
    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        return f"Compose({self.transforms!r})"


# ---------- Датасет: протокол последовательности -> работает с for, len, индексами и torch DataLoader ----------
class BaseDataset(ABC):
    """Абстрактный класс (как abstract class в C#): нельзя создать, пока не реализованы абстрактные методы."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Sample: ...

    def __iter__(self) -> Iterator[Sample]:        # общая реализация для всех наследников
        for i in range(len(self)):
            yield self[i]


class SyntheticDetection(BaseDataset):
    def __init__(self, n: int, transform: Transform | None = None, seed: int = 0):
        self.n, self.transform = n, transform
        self._rng = np.random.default_rng(seed)
        self._boxes = [[Box(10.0 * k, 10.0, 10.0 * k + 30, 50.0, "car") for k in range(1 + i % 3)] for i in range(n)]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Sample:
        if not 0 <= idx < self.n:
            raise IndexError(idx)                  # IndexError нужен, чтобы for по __getitem__ знал, где остановиться
        sample = Sample(np.zeros((64, 96, 3), np.uint8), list(self._boxes[idx]))
        return self.transform(sample) if self.transform else sample


if __name__ == "__main__":
    b = Box.from_yolo(0.5, 0.5, 0.25, 0.5, img_w=640, img_h=480, label="person")
    print(b, "| площадь:", b.area, "| хэшируема:", hash(b) == hash(Box(*[getattr(b, f) for f in ("x1", "y1", "x2", "y2")], "person")))
    try:
        Box(10, 10, 5, 20)
    except ValueError as e:
        print("валидация:", e)

    ds = SyntheticDetection(5, transform=Compose([HFlip()]))
    print("len:", len(ds), "| рамок на картинках:", [len(s.boxes) for s in ds])
    print("HFlip — это Transform (по протоколу):", isinstance(HFlip(), Transform))
    print("первая рамка после отражения:", ds[0].boxes[0])
    try:
        BaseDataset()
    except TypeError as e:
        print("абстрактный класс:", e)
