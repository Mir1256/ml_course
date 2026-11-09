"""Зависимые события на практике: случайное разбиение кадров видео на train/val даёт утечку.

Соседние кадры почти одинаковы. Если кадры независимы — случайное разбиение честное.
Если нет — вероятность, что «близнец» валидационного кадра попал в train, огромна.
"""
import numpy as np

rng = np.random.default_rng(0)
n_videos, frames_per_video, val_share = 50, 200, 0.2

video_id = np.repeat(np.arange(n_videos), frames_per_video)
frame_idx = np.tile(np.arange(frames_per_video), n_videos)

# 1. Случайное разбиение по кадрам
is_val = rng.random(video_id.size) < val_share
neighbor_in_train = []
for v, f, val in zip(video_id, frame_idx, is_val):
    if val:
        same_video = video_id == v
        close = np.abs(frame_idx - f) <= 2                      # кадры в пределах ±2 почти идентичны
        neighbor_in_train.append(np.any(same_video & close & ~is_val))
print(f"разбиение по кадрам: у {np.mean(neighbor_in_train):.1%} val-кадров есть почти-копия в train")
print(f"теория: 1 - P(все 4 соседа тоже в val) = 1 - {val_share}^4 = {1 - val_share**4:.4f}")

# 2. Разбиение по видео (группам): соседние кадры всегда в одной части
val_videos = rng.choice(n_videos, size=int(n_videos * val_share), replace=False)
is_val_g = np.isin(video_id, val_videos)
leak = any(np.any(is_val_g[video_id == v]) and np.any(~is_val_g[video_id == v]) for v in range(n_videos))
print(f"разбиение по видео: есть ли видео сразу в train и val? {leak}")
