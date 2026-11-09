"""Скейлеры с нуля в стиле scikit-learn: fit запоминает статистики train, transform применяет их к любым данным."""
import numpy as np


class StandardScalerNP:
    """z = (x - mean) / std — нулевое среднее, единичная дисперсия по каждому признаку."""
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0          # константный признак: делить на 0 нельзя, оставляем как есть
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, Z):
        return Z * self.scale_ + self.mean_


class MinMaxScalerNP:
    """x' = (x - min) / (max - min) в [0, 1] на train; на новых данных может выйти за [0, 1]."""
    def fit(self, X):
        self.min_, self.max_ = X.min(axis=0), X.max(axis=0)
        rng = self.max_ - self.min_
        self.range_ = np.where(rng == 0, 1.0, rng)
        return self

    def transform(self, X):
        return (X - self.min_) / self.range_


class RobustScalerNP:
    """x' = (x - median) / IQR — выбросы почти не влияют на статистики (урок 16)."""
    def fit(self, X):
        self.center_ = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        iqr = q75 - q25
        self.scale_ = np.where(iqr == 0, 1.0, iqr)
        return self

    def transform(self, X):
        return (X - self.center_) / self.scale_


def normalize_images(batch_nchw: np.ndarray, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> np.ndarray:
    """Поканальная нормализация батча в [0,1] ImageNet-статистиками — то, что делает torchvision Normalize."""
    m = np.asarray(mean, np.float32)[None, :, None, None]
    s = np.asarray(std, np.float32)[None, :, None, None]
    return (batch_nchw - m) / s
