"""Маленький нейрон-сеть 2 -> 3 -> 1 на нашем Value и на PyTorch: градиенты должны совпасть."""
import random

import torch

from autograd import Value

random.seed(0)
W1 = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(3)]
b1 = [random.uniform(-1, 1) for _ in range(3)]
W2 = [random.uniform(-1, 1) for _ in range(3)]
b2 = random.uniform(-1, 1)
x, target = [0.5, -1.2], 0.7

# ---------- наш autograd ----------
vW1 = [[Value(w) for w in row] for row in W1]
vb1 = [Value(b) for b in b1]
vW2 = [Value(w) for w in W2]
vb2 = Value(b2)
hidden = [sum((w * xi for w, xi in zip(row, x)), b).tanh() for row, b in zip(vW1, vb1)]
out = sum((w * h for w, h in zip(vW2, hidden)), vb2)
loss = (out - target) ** 2
loss.backward()

# ---------- PyTorch ----------
tW1 = torch.tensor(W1, dtype=torch.float64, requires_grad=True)
tb1 = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
tW2 = torch.tensor(W2, dtype=torch.float64, requires_grad=True)
tb2 = torch.tensor(b2, dtype=torch.float64, requires_grad=True)
th = torch.tanh(tW1 @ torch.tensor(x, dtype=torch.float64) + tb1)
tloss = (tW2 @ th + tb2 - target) ** 2
tloss.backward()

ours = [v.grad for row in vW1 for v in row]
print("loss свой / torch:", round(loss.data, 10), round(tloss.item(), 10))
print("dL/dW1 свой :", [round(g, 8) for g in ours])
print("dL/dW1 torch:", [round(g, 8) for g in tW1.grad.flatten().tolist()])
print("dL/dW2 совпадает:", all(abs(v.grad - g) < 1e-12 for v, g in zip(vW2, tW2.grad.tolist())))
print("dL/db2 совпадает:", abs(vb2.grad - tb2.grad.item()) < 1e-12)
