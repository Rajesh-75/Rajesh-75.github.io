"""
tensor_engine.py
==================================================================
A minimal reverse-mode automatic differentiation engine, built on
NumPy arrays only. No torch. No torch.autograd. Every backward()
rule below is written by hand.

The idea (same one PyTorch/micrograd use internally): every Tensor
remembers (a) which Tensors produced it (`_prev`) and (b) a closure
that knows how to push a gradient from itself backward onto those
parents (`_backward`). Calling `.backward()` on a scalar loss walks
this graph in reverse topological order and accumulates `.grad` on
every Tensor that needs one -- that walk, plus every individual
`_backward` rule, is the entire content of backpropagation. There is
no other mechanism; this IS what torch's C++ autograd engine does,
just in visible Python.
==================================================================
"""

from __future__ import annotations
import numpy as np


class Tensor:
    def __init__(self, data, _prev=(), requires_grad=False, _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.requires_grad = requires_grad
        self._prev = set(_prev)
        self._backward = lambda: None
        self._op = _op

    # -- bookkeeping -----------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = None

    def _ensure_grad(self):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

    @staticmethod
    def _unbroadcast(grad, shape):
        """Undo NumPy broadcasting on the way back: if a forward op
        broadcast e.g. a (n_embd,) bias up to (B,T,n_embd), the
        incoming gradient arrives at the broadcasted shape and must
        be summed back down to the parameter's own shape before it
        can be added to that parameter's .grad."""
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    # -- graph traversal ---------------------------------------------
    def backward(self):
        assert self.data.shape == () or self.data.size == 1, "backward() only from a scalar loss"
        topo, visited = [], set()

        def build(t):
            if id(t) not in visited:
                visited.add(id(t))
                for p in t._prev:
                    build(p)
                topo.append(t)

        build(self)
        self._ensure_grad()
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t._backward()

    # -- ops: each one does two things --------------------------------
    #    1. compute the forward value with numpy
    #    2. define _backward: how to route d(loss)/d(out) onto the inputs
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), _op="+")

        def _backward():
            if self.requires_grad or self.grad is not None:
                self._ensure_grad()
                self.grad += self._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad or other.grad is not None:
                other._ensure_grad()
                other.grad += self._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), _op="*")

        def _backward():
            if self.requires_grad or self.grad is not None:
                self._ensure_grad()
                self.grad += self._unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad or other.grad is not None:
                other._ensure_grad()
                other.grad += self._unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1.0)

    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = Tensor(self.data ** power, (self,), _op=f"**{power}")

        def _backward():
            if self.requires_grad or self.grad is not None:
                self._ensure_grad()
                self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def matmul(self, other):
        """d/dA (A@B) w.r.t upstream grad G: dA = G @ B.T, dB = A.T @ G.
        This is the single most important backward rule in a transformer --
        every Linear layer and every attention score/weighted-sum is a matmul.
        When A has extra leading (batch) dims that B doesn't (e.g. Linear:
        x is (B,T,C), weight is (C,out)), the raw gradient dB comes out with
        those batch dims still attached and must be summed over them --
        the same broadcasting-undo idea as _unbroadcast, applied here because
        matmul's broadcasting rules are batch-dim, not elementwise."""
        out = Tensor(self.data @ other.data, (self, other), _op="matmul")

        def _backward():
            if self.requires_grad or self.grad is not None:
                self._ensure_grad()
                grad_self = out.grad @ np.swapaxes(other.data, -1, -2)
                self.grad += self._unbroadcast(grad_self, self.data.shape)
            if other.requires_grad or other.grad is not None:
                other._ensure_grad()
                grad_other = np.swapaxes(self.data, -1, -2) @ out.grad
                other.grad += self._unbroadcast(grad_other, other.data.shape)
        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def transpose(self, *axes):
        out = Tensor(np.transpose(self.data, axes if axes else None), (self,), _op="T")
        inv = np.argsort(axes) if axes else None

        def _backward():
            if self.requires_grad or self.grad is not None:
                self._ensure_grad()
                self.grad += np.transpose(out.grad, inv if axes else None)
        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), _op="reshape")

        def _backward():
            if self.requires_grad or self.grad is not None:
                self._ensure_grad()
                self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), _op="sum")

        def _backward():
            if self.requires_grad or self.grad is not None:
                self._ensure_grad()
                g = out.grad
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis)
                self.grad += np.broadcast_to(g, self.data.shape)
        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def var(self, axis=None, keepdims=False):
        m = self.mean(axis=axis, keepdims=True)
        return ((self - m) ** 2).mean(axis=axis, keepdims=keepdims)

    def sqrt(self):
        return self ** 0.5

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), _op="tanh")

        def _backward():
            if self.requires_grad or self.grad is not None:
                self._ensure_grad()
                self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, (self,), _op="exp")

        def _backward():
            if self.requires_grad or self.grad is not None:
                self._ensure_grad()
                self.grad += e * out.grad
        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), _op="log")

        def _backward():
            if self.requires_grad or self.grad is not None:
                self._ensure_grad()
                self.grad += out.grad / self.data
        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def max(self, axis=None, keepdims=False):
        m = self.data.max(axis=axis, keepdims=True)
        out_data = m if keepdims else np.squeeze(m, axis=axis)
        out = Tensor(out_data, (self,), _op="max")
        mask = (self.data == m).astype(np.float64)
        mask = mask / mask.sum(axis=axis, keepdims=True)  # split gradient on ties

        def _backward():
            if self.requires_grad or self.grad is not None:
                self._ensure_grad()
                g = out.grad
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis)
                self.grad += mask * g
        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def masked_fill(self, mask, value):
        """mask: numpy boolean array, same-shaped (broadcastable). Positions
        where mask is True get replaced by `value`; gradient does not flow
        through filled positions (their contribution to the loss is fixed,
        not computed) -- exactly like torch's masked_fill in attention."""
        filled = np.where(mask, value, self.data)
        out = Tensor(filled, (self,), _op="masked_fill")
        keep = (~mask).astype(np.float64)

        def _backward():
            if self.requires_grad or self.grad is not None:
                self._ensure_grad()
                self.grad += keep * out.grad
        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def __getitem__(self, idx):
        """Used for embedding lookup (weight[input_ids]) and slicing. Backward
        must SCATTER-ADD into the original array, since the same row (e.g. a
        repeated token id) can be read multiple times in one batch and every
        read contributes to that row's gradient."""
        out = Tensor(self.data[idx], (self,), _op="getitem")

        def _backward():
            if self.requires_grad or self.grad is not None:
                self._ensure_grad()
                np.add.at(self.grad, idx, out.grad)
        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"


# ----------------------------------------------------------------------
# Free functions built from the primitive ops above -- these are the
# hand-written equivalents of F.softmax / F.cross_entropy / F.gelu.
# Every one of them is composed ONLY from Tensor ops that already know
# their own backward rule, so no new _backward closures are needed here
# -- the chain rule composes automatically through the graph.
# ----------------------------------------------------------------------
def softmax(x: Tensor, axis: int = -1) -> Tensor:
    shifted = x - x.max(axis=axis, keepdims=True)
    e = shifted.exp()
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    shifted = x - x.max(axis=axis, keepdims=True)
    return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()


def cross_entropy(logits: Tensor, targets: np.ndarray, ignore_index: int = -100) -> Tensor:
    """logits: Tensor of shape (N, C). targets: plain numpy int array (N,),
    entries equal to ignore_index are excluded from the loss (padding)."""
    log_probs = log_softmax(logits, axis=-1)          # (N, C), still a Tensor w/ graph
    valid = (targets != ignore_index)
    safe_targets = np.where(valid, targets, 0)
    n = logits.shape[0]
    picked = log_probs[np.arange(n), safe_targets]     # (N,) -- gather via __getitem__
    nll = -picked
    nll = nll * valid.astype(np.float64)               # zero out ignored positions
    denom = max(valid.sum(), 1)
    return nll.sum() * (1.0 / denom)


def new_gelu(x: Tensor) -> Tensor:
    c = np.sqrt(2.0 / np.pi)
    return (x * 0.5) * ((((x ** 3) * 0.044715 + x) * c).tanh() + 1.0)