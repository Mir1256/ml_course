"""Визуальная отладка CV: рамки и маски — сначала с нуля в NumPy, потом через Matplotlib (OO API)."""
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_sample_image

PALETTE = np.array([[230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48], [145, 30, 180]], dtype=np.uint8)


# ---------- С нуля: рисуем прямо в массиве ----------
def draw_boxes_np(img: np.ndarray, boxes: np.ndarray, labels: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Растеризуем контуры рамок срезами: 4 полосы на рамку. Возвращает копию — исходник не трогаем."""
    out = img.copy()
    H, W = img.shape[:2]
    for (x1, y1, x2, y2), lab in zip(boxes.round().astype(int), labels):
        x1, x2 = np.clip([x1, x2], 0, W - 1)
        y1, y2 = np.clip([y1, y2], 0, H - 1)
        c = PALETTE[lab % len(PALETTE)]
        out[y1:y1 + thickness, x1:x2 + 1] = c
        out[max(y2 - thickness + 1, 0):y2 + 1, x1:x2 + 1] = c
        out[y1:y2 + 1, x1:x1 + thickness] = c
        out[y1:y2 + 1, max(x2 - thickness + 1, 0):x2 + 1] = c
    return out


def overlay_mask_np(img: np.ndarray, mask: np.ndarray, color, alpha: float = 0.5) -> np.ndarray:
    """Альфа-смешивание только внутри маски: out = α·color + (1-α)·img."""
    out = img.astype(np.float32)
    out[mask] = alpha * np.asarray(color, np.float32) + (1 - alpha) * out[mask]
    return out.round().astype(np.uint8)


# ---------- Matplotlib: то же, но рамки — векторные объекты поверх картинки ----------
def show_detections(ax, img, boxes, labels, scores=None, names=None, mask=None):
    ax.imshow(img)                                                  # uint8 RGB: значения 0..255 трактуются как цвета
    if mask is not None:
        ax.imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.45, interpolation="nearest")
    for i, ((x1, y1, x2, y2), lab) in enumerate(zip(boxes, labels)):
        color = PALETTE[lab % len(PALETTE)] / 255
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
        text = names[lab] if names else str(lab)
        if scores is not None:
            text += f" {scores[i]:.2f}"
        ax.text(x1, y1, text, color="white", fontsize=8, va="bottom", bbox=dict(facecolor=color, pad=1, lw=0))
    ax.set_axis_off()


if __name__ == "__main__":
    img = load_sample_image("china.jpg")                            # (427, 640, 3) uint8, RGB
    boxes = np.array([[30, 40, 250, 300], [300, 120, 600, 400], [420, 20, 520, 110]], float)
    labels = np.array([0, 1, 2])
    scores = np.array([0.91, 0.77, 0.42])
    names = ["building", "roof", "sky"]
    mask = (img[..., 2] > img[..., 0] + 20) & (np.arange(img.shape[0])[:, None] < 150)   # «небо»: синее и в верхней части

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(draw_boxes_np(overlay_mask_np(img, mask, (255, 200, 0)), boxes, labels))
    axes[0].set_title("NumPy: рисуем в пикселях"); axes[0].set_axis_off()
    show_detections(axes[1], img, boxes, labels, scores, names, mask)
    axes[1].set_title("Matplotlib: patches + text + маска")
    axes[2].imshow(img[..., ::-1]); axes[2].set_title("ошибка: каналы BGR вместо RGB"); axes[2].set_axis_off()
    fig.tight_layout(); fig.savefig("01_detections.png", dpi=80); plt.close(fig)

    # Ловушки imshow: диапазон значений и маленькие картинки
    small = np.random.default_rng(0).random((8, 8))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    axes[0].imshow(img.astype(np.float32)); axes[0].set_title("float 0..255: всё белое (клиппинг)")
    axes[1].imshow(img.astype(np.float32) / 255); axes[1].set_title("float 0..1: правильно")
    axes[2].imshow(small, cmap="gray"); axes[2].set_title("8x8, сглаживание по умолчанию")
    im = axes[3].imshow(small, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    axes[3].set_title("interpolation='nearest', vmin/vmax")
    fig.colorbar(im, ax=axes[3])
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout(); fig.savefig("02_imshow_pitfalls.png", dpi=80); plt.close(fig)

    # Сетка примеров: так смотрят батч перед обучением
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(2, 4, figsize=(12, 5))
    for ax in axes.flat:
        y, x = rng.integers(0, 427 - 160), rng.integers(0, 640 - 160)
        crop = img[y:y + 160, x:x + 160]
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
        ax.imshow(crop); ax.set_title(f"crop ({x},{y})", fontsize=8); ax.set_axis_off()
    fig.suptitle("Сетка случайных кропов с отражением — визуальная проверка аугментаций")
    fig.tight_layout(); fig.savefig("03_grid.png", dpi=80); plt.close(fig)
    print("сохранено 3 фигуры")
