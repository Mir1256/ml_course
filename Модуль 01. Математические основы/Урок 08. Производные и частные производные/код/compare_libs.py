"""Три способа получить производную: символьно (SymPy), численно (наш код) и автоматически (PyTorch autograd)."""
import numpy as np
import sympy as sp

from derivatives import central_diff

x, y = sp.symbols("x y")
expr = sp.log(1 + sp.exp(-x * y)) + x**2 * sp.sin(y)              # похоже на логистическую потерю + регуляризатор
dx, dy = sp.diff(expr, x), sp.diff(expr, y)
print("∂/∂x =", sp.simplify(dx))
print("∂/∂y =", sp.simplify(dy))

point = {x: 0.7, y: -1.3}
sym = [float(dx.subs(point)), float(dy.subs(point))]
f = sp.lambdify((x, y), expr, "numpy")
num = [central_diff(lambda t: f(t, -1.3), 0.7, 1e-5), central_diff(lambda t: f(0.7, t), -1.3, 1e-5)]
print("SymPy   :", np.round(sym, 8))
print("численно:", np.round(num, 8))

try:
    import torch

    xt = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    yt = torch.tensor(-1.3, dtype=torch.float64, requires_grad=True)
    out = torch.log1p(torch.exp(-xt * yt)) + xt**2 * torch.sin(yt)
    out.backward()                                                  # обратный проход: цепное правило (урок 9)
    print("autograd:", np.round([xt.grad.item(), yt.grad.item()], 8))
except ImportError:
    print("torch не установлен — autograd пропущен")
