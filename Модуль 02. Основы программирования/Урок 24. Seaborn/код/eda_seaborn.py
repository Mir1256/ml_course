"""Seaborn для EDA: длинный формат данных, группировка через hue/col, статистика «из коробки» — и сверка её с ручной."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fake_annotations import make_annotations

sns.set_theme(style="whitegrid", context="notebook")
df = make_annotations()
df = df[(df.area < 640 * 480) & df.aspect.between(1 / 20, 20)].copy()
df["light"] = np.where(df.night, "ночь", "день")
df["size"] = pd.cut(df.area, [0, 32**2, 96**2, np.inf], labels=["small", "medium", "large"], right=False)
order = df.category.value_counts().index

# 1. Распределение с разбивкой: hue + log-шкала + нормировка внутри групп
g = sns.displot(df, x="area", hue="light", log_scale=True, element="step", stat="density", common_norm=False, height=3.5, aspect=1.6)
g.ax.set_title("Площадь рамок днём и ночью (плотности нормированы отдельно)")
g.savefig("01_hist_hue.png", dpi=80); plt.close("all")

# 2. Small multiples: одна фигура на значение col — сравнение без наложения
g = sns.catplot(df, x="category", y="score", col="light", kind="box", order=order, height=3.5, aspect=1.3, showfliers=False)
g.set_xticklabels(rotation=30)
g.figure.suptitle("Уверенность модели по классам", y=1.03)
g.savefig("02_box_cols.png", dpi=80); plt.close("all")

# 3. Тепловая карта сводной таблицы с подписями
pivot = df.pivot_table(index="category", columns="size", values="score", aggfunc="mean", observed=True).loc[order]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar_kws={"label": "средняя уверенность"})
ax.tick_params(axis="y", rotation=0)                    # подписи классов горизонтально, иначе наезжают друг на друга
ax.set_title("Где модель слабее: класс × размер")
fig.tight_layout(); fig.savefig("03_heatmap.png", dpi=80); plt.close(fig)

# 4. Столбцы со статистикой: по умолчанию seaborn рисует среднее и 95%-ДИ БУТСТРЕПОМ (урок 18)
fig, ax = plt.subplots(figsize=(6, 3.5))
sns.barplot(df, x="category", y="score", hue="light", order=order, hue_order=["день", "ночь"],
            errorbar=("ci", 95), n_boot=2000, seed=0, ax=ax)   # явный hue_order: иначе порядок — по появлению в данных
ax.tick_params(axis="x", rotation=30)
ax.set_title("Средняя уверенность и 95%-ДИ (бутстрап)")
fig.tight_layout(); fig.savefig("04_bar_ci.png", dpi=80)

# Сверка: достаём концы «усов» из графика и сравниваем со своим бутстрапом
rng = np.random.default_rng(0)
cat, light = order[0], "день"
values = df.loc[(df.category == cat) & (df.light == light), "score"].to_numpy()
boots = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
ours = np.quantile(boots, [0.025, 0.975])
whisker = ax.lines[0].get_ydata()                       # первая линия-ус: класс order[0], hue «день»
print(f"{cat}/{light}: ДИ seaborn [{min(whisker):.4f}, {max(whisker):.4f}] | свой бутстрап [{ours[0]:.4f}, {ours[1]:.4f}]")
plt.close(fig)

# 5. Попарные связи числовых признаков (на подвыборке: pairplot квадратичен по числу признаков)
sub = df.sample(1500, random_state=0).assign(log_area=lambda d: np.log10(d.area), log_aspect=lambda d: np.log10(d.aspect))
g = sns.pairplot(sub, vars=["log_area", "log_aspect", "score"], hue="light", corner=True, plot_kws={"s": 8, "alpha": 0.5}, height=2.2)
g.savefig("05_pairplot.png", dpi=70); plt.close("all")

# 6. Новый декларативный интерфейс seaborn.objects: слои + преобразования статистики
import seaborn.objects as so
(so.Plot(sub, x="log_area", y="score", color="light")
   .add(so.Dots(alpha=0.2, pointsize=2))
   .add(so.Line(linewidth=2), so.PolyFit(order=2))
   .label(title="Уверенность против log10(площади): точки + полиномиальный тренд")
   .save("06_objects.png", dpi=80, bbox_inches="tight"))
print("сохранено 6 графиков")
