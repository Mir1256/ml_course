"""Скалярный autograd с нуля (в духе micrograd А. Карпатого): граф вычислений + обратный проход по цепному правилу."""
from __future__ import annotations

import math


class Value:
    """Скаляр, который помнит, из чего он вычислен, и умеет протолкнуть градиент к своим «родителям»."""

    def __init__(self, data: float, parents: tuple[Value, ...] = (), op: str = ""):
        self.data = float(data)
        self.grad = 0.0                      # dL/d(self), заполняется обратным проходом
        self._parents = parents
        self._op = op
        self._backward = lambda: None        # как раздать self.grad родителям (локальные производные)

    # ---------- операции: прямой проход + правило для обратного ----------
    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad            # d(a+b)/da = 1
            other.grad += out.grad           # += а не = : узел может использоваться несколько раз (ветвление)
        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad   # d(ab)/da = b
            other.grad += self.data * out.grad   # d(ab)/db = a
        out._backward = _backward
        return out

    def __pow__(self, k: float) -> Value:
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad     # (e^x)' = e^x — используем уже посчитанное значение
        out._backward = _backward
        return out

    def log(self) -> Value:
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    # ---------- производные операции через базовые ----------
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * (other if isinstance(other, Value) else Value(other)) ** -1
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return Value(other) - self
    def __rtruediv__(self, other): return Value(other) / self

    def backward(self) -> None:
        """Обратный проход: топологическая сортировка графа, затем узлы в обратном порядке."""
        order, seen = [], set()

        def visit(v: Value):
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    visit(p)
                order.append(v)              # узел добавляется ПОСЛЕ всех, от кого он зависит
        visit(self)

        for v in order:
            v.grad = 0.0
        self.grad = 1.0                      # dL/dL = 1
        for v in reversed(order):            # от выхода к входам: к моменту v._backward() его grad уже полный
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"


if __name__ == "__main__":
    # Логистическая потеря одного объекта: L = log(1 + exp(-y * (w1*x1 + w2*x2 + b)))
    x1, x2, y = Value(2.0), Value(-1.0), Value(1.0)
    w1, w2, b = Value(0.5), Value(-0.3), Value(0.1)
    z = w1 * x1 + w2 * x2 + b
    L = (1 + (-y * z).exp()).log()
    L.backward()
    print("L =", round(L.data, 6))
    print("dL/dw1, dL/dw2, dL/db =", round(w1.grad, 6), round(w2.grad, 6), round(b.grad, 6))
    s = 1 / (1 + math.exp(y.data * z.data))           # аналитически: dL/dz = -y * σ(-yz)
    print("аналитически          =", round(-y.data * s * x1.data, 6), round(-y.data * s * x2.data, 6), round(-y.data * s, 6))

    # Ветвление: a используется дважды -> градиенты по двум путям складываются
    a = Value(3.0)
    f = a * a + a                                    # df/da = 2a + 1 = 7
    f.backward()
    print("f = a*a + a, df/da =", a.grad)

    # Затухание: производная цепочки из 10 tanh перемножает 10 локальных производных < 1
    v = Value(1.5)
    h = v
    for _ in range(10):
        h = h.tanh()
    h.backward()
    print(f"10 tanh подряд: d out / d in = {v.grad:.2e}")
