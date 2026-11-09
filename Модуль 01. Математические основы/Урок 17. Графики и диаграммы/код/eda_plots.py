"""EDA-графики аннотаций детекции на matplotlib. Каждый график отвечает на конкретный вопрос."""
import matplotlib.pyplot as plt
import numpy as np

from density import kde
from fake_annotations import make_annotations

df = make_annotations()
df = df[(df.w * df.h < 640 * 480) & (df.aspect.between(1 / 20, 20))]       # убираем невозможные рамки (урок 16)
order = df.category.value_counts().index

# 1. «Насколько несбалансированы классы?» — горизонтальные столбцы, отсортированные, с подписями
fig, ax = plt.subplots(figsize=(6, 3.5))
counts = df.category.value_counts()
ax.barh(counts.index[::-1], counts.values[::-1], color="#4C72B0")
for i, v in enumerate(counts.values[::-1]):
    ax.text(v, i, f" {v}", va="center", fontsize=8)
ax.set_xlabel("число рамок")
ax.set_title("Дисбаланс классов")
fig.tight_layout(); fig.savefig("01_classes.png", dpi=90); plt.close(fig)

# 2. «Как распределены размеры?» — одна величина в линейной и логарифмической шкале
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].hist(df.area, bins=100, color="#55A868")
axes[0].set_title("площадь: линейная шкала — всё в одном столбце")
log_area = np.log10(df.area)
axes[1].hist(log_area, bins="fd", density=True, color="#55A868", alpha=0.6, label="гистограмма")
grid = np.linspace(log_area.min(), log_area.max(), 300)
axes[1].plot(grid, kde(log_area.to_numpy(), grid), color="k", label="KDE")
for thr, name in ((32**2, "small"), (96**2, "large")):
    axes[1].axvline(np.log10(thr), ls="--", color="r")
    axes[1].text(np.log10(thr), axes[1].get_ylim()[1] * 0.9, f" {name}", color="r", fontsize=8)
axes[1].set_xlabel("log10(площадь)"); axes[1].legend()
axes[1].set_title("log-шкала: видно форму и границы COCO")
fig.tight_layout(); fig.savefig("02_area.png", dpi=90); plt.close(fig)

# 3. «Чем классы отличаются по размеру?» — boxplot по группам в log-шкале
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.boxplot([df.loc[df.category == c, "area"] for c in order], tick_labels=list(order), showfliers=False)
ax.set_yscale("log"); ax.set_ylabel("площадь, px² (log)")
ax.axhline(32**2, ls="--", color="r", lw=1)
ax.set_title("Размер объектов по классам (усы — 1.5 IQR, выбросы скрыты)")
fig.tight_layout(); fig.savefig("03_box_by_class.png", dpi=90); plt.close(fig)

# 4. «Какие формы рамок?» — 20 000 точек: scatter тонет, hexbin показывает плотность
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(df.w, df.h, s=2, alpha=0.5)
axes[0].set_title("scatter: перекрытие точек (overplotting)")
hb = axes[1].hexbin(np.log10(df.w), np.log10(df.h), gridsize=40, cmap="viridis", mincnt=1)
fig.colorbar(hb, ax=axes[1], label="число рамок")
axes[1].set_xlabel("log10(w)"); axes[1].set_ylabel("log10(h)")
axes[1].set_title("hexbin в log-шкале: кластеры форм")
fig.tight_layout(); fig.savefig("04_shapes.png", dpi=90); plt.close(fig)

# 5. «Где на кадре объекты?» — тепловая карта центров рамок
cx, cy = df.x + df.w / 2, df.y + df.h / 2
heat, _, _ = np.histogram2d(cy, cx, bins=(48, 64), range=((0, 480), (0, 640)))
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(heat, extent=(0, 640, 480, 0), cmap="magma")
fig.colorbar(im, ax=ax, label="число центров")
ax.set_title("Центры рамок на кадре 640x480")
fig.tight_layout(); fig.savefig("05_centers.png", dpi=90); plt.close(fig)

# 6. «Где модель хуже?» — ECDF уверенности по срезам вместо двух наложенных гистограмм
fig, ax = plt.subplots(figsize=(6, 3.5))
for night, color in ((False, "#4C72B0"), (True, "#C44E52")):
    s = np.sort(df.loc[df.night == night, "score"])
    ax.step(s, np.arange(1, len(s) + 1) / len(s), where="post", color=color, label="ночь" if night else "день")
ax.set_xlabel("уверенность модели"); ax.set_ylabel("доля рамок ≤ x"); ax.legend()
ax.set_title("ECDF: ночная кривая левее — модель менее уверена")
fig.tight_layout(); fig.savefig("06_ecdf.png", dpi=90); plt.close(fig)
print("сохранено 6 графиков")
