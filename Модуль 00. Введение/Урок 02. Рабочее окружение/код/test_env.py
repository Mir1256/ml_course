"""Тесты окружения. Запуск: uv run pytest "Модуль 00. Введение/Урок 02. Рабочее окружение/код" """
import sys

import pytest

from env_check import PACKAGES, versions


def test_python_version():
    assert sys.version_info >= (3, 11)


def test_all_packages_installed():
    missing = [name for name, ver in versions().items() if ver is None]
    assert not missing, f"не установлены: {missing}"
    assert set(versions()) == set(PACKAGES)


def test_numpy_blas_works():
    import numpy as np

    a = np.random.default_rng(0).normal(size=(200, 200))
    assert np.allclose(a @ np.linalg.inv(a), np.eye(200), atol=1e-8)


def test_gpu_kernel_runs():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA недоступна на этой машине")
    x = torch.arange(10, device="cuda", dtype=torch.float32)
    assert (x * 2).sum().item() == 90.0     # реальный запуск ядра, а не только is_available()
