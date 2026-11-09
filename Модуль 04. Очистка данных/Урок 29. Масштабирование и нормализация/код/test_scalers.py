import numpy as np

from scalers import MinMaxScalerNP, RobustScalerNP, StandardScalerNP, normalize_images


def test_standard_scaler_stats_and_inverse():
    X = np.random.default_rng(0).normal(5, 3, (100, 4))
    s = StandardScalerNP().fit(X)
    Z = s.transform(X)
    assert np.allclose(Z.mean(0), 0) and np.allclose(Z.std(0), 1)
    assert np.allclose(s.inverse_transform(Z), X)


def test_constant_feature_no_nan():
    X = np.c_[np.ones(10), np.arange(10.0)]
    for s in (StandardScalerNP(), MinMaxScalerNP(), RobustScalerNP()):
        assert np.isfinite(s.fit(X).transform(X)).all()


def test_fit_on_train_only_test_can_exceed_range():
    train, test = np.array([[0.0], [10.0]]), np.array([[20.0]])
    assert MinMaxScalerNP().fit(train).transform(test)[0, 0] == 2.0     # вне [0, 1] — это нормально


def test_imagenet_normalize_matches_torchvision():
    torch = __import__("pytest").importorskip("torch")
    from torchvision.transforms import Normalize
    x = np.random.default_rng(1).random((2, 3, 8, 8)).astype(np.float32)
    ref = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(torch.from_numpy(x)).numpy()
    assert np.allclose(normalize_images(x), ref, atol=1e-6)
