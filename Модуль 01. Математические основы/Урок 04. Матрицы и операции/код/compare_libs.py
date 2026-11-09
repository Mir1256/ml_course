"""Сверка с SciPy, scikit-image-подобной логикой и (если есть) PyTorch."""
import time

import numpy as np
from scipy.signal import correlate2d

from matrices import conv2d_im2col, linear_layer, matmul_loop

rng = np.random.default_rng(1)

# 1. Скорость: тройной цикл против BLAS
A, B = rng.normal(size=(80, 80)), rng.normal(size=(80, 80))
t0 = time.perf_counter(); C1 = matmul_loop(A, B); t_loop = time.perf_counter() - t0
t0 = time.perf_counter(); C2 = A @ B; t_blas = time.perf_counter() - t0
print(f"matmul 80x80: цикл {t_loop * 1e3:.0f} мс, NumPy/BLAS {t_blas * 1e3:.3f} мс, совпадают: {np.allclose(C1, C2)}")

# 2. Свёртка: DL-«свёртка» = кросс-корреляция (ядро НЕ переворачивается)
img, ker = rng.normal(size=(32, 32)), rng.normal(size=(3, 3))
print("conv2d_im2col == scipy correlate2d(valid):", np.allclose(conv2d_im2col(img, ker), correlate2d(img, ker, mode="valid")))

try:
    import torch

    t_out = torch.nn.functional.conv2d(torch.from_numpy(img)[None, None], torch.from_numpy(ker)[None, None])
    print("conv2d_im2col == torch conv2d:          ", np.allclose(conv2d_im2col(img, ker), t_out[0, 0].numpy()))

    lin = torch.nn.Linear(3, 2, dtype=torch.float64)
    X = rng.normal(size=(5, 3))
    ours = linear_layer(X, lin.weight.detach().numpy(), lin.bias.detach().numpy())
    print("linear_layer == nn.Linear:              ", np.allclose(ours, lin(torch.from_numpy(X)).detach().numpy()))
except ImportError:
    print("torch не установлен — сверка с PyTorch пропущена")
