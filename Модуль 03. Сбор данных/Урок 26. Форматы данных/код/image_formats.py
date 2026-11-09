"""Форматы картинок для CV: почему маски сегментации нельзя хранить в JPEG, и что занимает место."""
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import load_sample_image

photo = load_sample_image("china.jpg")
H, W = photo.shape[:2]
mask = np.zeros((H, W), np.uint8)                       # маска классов: 0 фон, 1 здание, 2 вода
mask[60:330, 40:260] = 1
mask[250:, 250:] = 2

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for name, kw in (("mask.png", {}), ("mask_q95.jpg", {"quality": 95}), ("mask_q75.jpg", {"quality": 75})):
        Image.fromarray(mask).save(tmp / name, **kw)
        back = np.asarray(Image.open(tmp / name))
        wrong = (back != mask).mean()
        print(f"{name:<13} {(tmp / name).stat().st_size / 1e3:6.1f} КБ | уникальных значений {np.unique(back).size:>3} | неверных пикселей {wrong:.2%}")

    for name, kw in (("photo.png", {}), ("photo_q95.jpg", {"quality": 95}), ("photo_q75.jpg", {"quality": 75}), ("photo.webp", {"quality": 90})):
        Image.fromarray(photo).save(tmp / name, **kw)
        back = np.asarray(Image.open(tmp / name).convert("RGB")).astype(np.float64)
        mse = np.mean((back - photo) ** 2)
        psnr = np.inf if mse == 0 else 10 * np.log10(255**2 / mse)
        print(f"{name:<13} {(tmp / name).stat().st_size / 1e3:6.1f} КБ | PSNR {psnr:5.1f} дБ")

    np.save(tmp / "depth.npy", np.random.default_rng(0).random((H, W)).astype(np.float16))
    print(f"карта глубины float16 в .npy: {(tmp / 'depth.npy').stat().st_size / 1e3:.0f} КБ (сырые {H * W * 2 / 1e3:.0f} КБ + заголовок)")
print(f"несжатое фото {H}x{W}x3 uint8: {photo.nbytes / 1e3:.0f} КБ")
