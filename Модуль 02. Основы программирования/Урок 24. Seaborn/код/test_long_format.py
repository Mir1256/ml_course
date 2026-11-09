"""Seaborn работает с «длинным» форматом: одна строка — одно наблюдение. Проверяем переход wide <-> long."""
import pandas as pd


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Таблица метрик моделей (строка — эпоха, столбцы — модели) -> длинный формат для sns.lineplot(hue='model')."""
    return wide.reset_index(names="epoch").melt(id_vars="epoch", var_name="model", value_name="mAP")


def test_melt_and_back():
    wide = pd.DataFrame({"yolo26n": [0.31, 0.36], "yolo26s": [0.38, 0.44]})
    long = to_long(wide)
    assert long.shape == (4, 3) and set(long.model) == {"yolo26n", "yolo26s"}
    back = long.pivot(index="epoch", columns="model", values="mAP")
    assert back.loc[1, "yolo26s"] == 0.44
