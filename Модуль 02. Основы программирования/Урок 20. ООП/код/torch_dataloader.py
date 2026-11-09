"""Тот же датасет в torch.utils.data.DataLoader: нужен свой collate_fn, потому что число рамок на картинках разное."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import Compose, HFlip, SyntheticDetection


class TorchDetection(Dataset):
    """Обёртка: torch ждёт только __len__ и __getitem__ — тот же «утиный» протокол, что у нашего датасета."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = self.base[idx]
        image = torch.from_numpy(s.image).permute(2, 0, 1).float() / 255           # HWC uint8 -> CHW float (урок 3)
        boxes = torch.tensor([[b.x1, b.y1, b.x2, b.y2] for b in s.boxes], dtype=torch.float32)
        return image, boxes


def detection_collate(batch):
    """Картинки одного размера складываем в тензор (N, C, H, W), рамки оставляем списком тензоров разной длины."""
    images, boxes = zip(*batch)
    return torch.stack(images), list(boxes)


loader = DataLoader(TorchDetection(SyntheticDetection(10, Compose([HFlip()]))), batch_size=4, collate_fn=detection_collate)
images, boxes = next(iter(loader))
print("батч картинок:", tuple(images.shape), "| рамок на картинках:", [len(b) for b in boxes])

try:
    next(iter(DataLoader(TorchDetection(SyntheticDetection(10)), batch_size=4)))
except RuntimeError as e:
    print("без своего collate_fn:", str(e).splitlines()[0][:90])
