"""Псевдоразметка с нуля (то, что делает SelfTrainingClassifier): учимся на метках, размечаем уверенные, повторяем."""
import numpy as np


def self_training(make_model, X, y, threshold=0.9, max_iter=10):
    """y: метки, -1 для неразмеченных. Возвращает обученную модель и историю числа размеченных объектов."""
    y = y.copy()
    history = [int((y != -1).sum())]
    for _ in range(max_iter):
        mask = y != -1
        model = make_model().fit(X[mask], y[mask])
        if mask.all():
            break
        proba = model.predict_proba(X[~mask])
        confident = proba.max(axis=1) >= threshold
        if not confident.any():
            break
        idx = np.flatnonzero(~mask)[confident]
        y[idx] = model.classes_[proba[confident].argmax(axis=1)]          # псевдометки — модель размечает сама себе
        history.append(int((y != -1).sum()))
    return model, y, history


if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.semi_supervised import SelfTrainingClassifier

    X, y = load_digits(return_X_y=True)
    X = X / 16
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    rng = np.random.default_rng(0)
    y_semi = np.full(len(ytr), -1)
    lab = rng.choice(len(ytr), 50, replace=False)
    y_semi[lab] = ytr[lab]
    make = lambda: LogisticRegression(max_iter=3000)
    model, y_pseudo, hist = self_training(make, Xtr, y_semi)
    added = y_semi == -1
    added &= y_pseudo != -1
    print("размечено по итерациям:", hist)
    print(f"точность псевдометок: {np.mean(y_pseudo[added] == ytr[added]):.3f} на {added.sum()} объектах")
    print(f"свой self-training: {model.score(Xte, yte):.3f} | sklearn: {SelfTrainingClassifier(make(), threshold=0.9).fit(Xtr, y_semi).score(Xte, yte):.3f}")
