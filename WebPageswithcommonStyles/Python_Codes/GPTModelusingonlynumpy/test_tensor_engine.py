import numpy as np
from tensor_engine import Tensor, softmax, cross_entropy, new_gelu

np.random.seed(0)


def numerical_grad(f, x_data, eps=1e-5):
    grad = np.zeros_like(x_data)
    it = np.nditer(x_data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x_data[idx]
        x_data[idx] = orig + eps
        plus = f(x_data)
        x_data[idx] = orig - eps
        minus = f(x_data)
        x_data[idx] = orig
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


def check(name, f_tensor, x_data):
    x = Tensor(x_data.copy(), requires_grad=True)
    out = f_tensor(x)
    out.backward()
    analytic = x.grad.copy()

    def f_np(data):
        return f_tensor(Tensor(data)).data.sum() if f_tensor(Tensor(data)).data.ndim else float(f_tensor(Tensor(data)).data)

    numeric = numerical_grad(lambda d: float(np.sum(f_tensor(Tensor(d)).data)), x_data)
    max_err = np.max(np.abs(analytic - numeric))
    status = "OK" if max_err < 1e-4 else "FAIL"
    print(f"[{status}] {name:22s} max|analytic-numeric| = {max_err:.2e}")
    assert max_err < 1e-4, f"{name} gradient mismatch"


# --- basic ops ---
check("add+mul", lambda x: (x * 2.0 + 3.0).sum(), np.random.randn(4, 3))
_W = np.random.randn(3, 5)  # fixed, so numerical/analytic checks compare the same function
check("matmul", lambda x: x.matmul(Tensor(_W)).sum(), np.random.randn(4, 3))
check("sum/mean", lambda x: x.mean(axis=-1).sum(), np.random.randn(4, 3))
check("var/sqrt", lambda x: x.var(axis=-1).sqrt().sum(), np.random.randn(4, 5) + 2.0)
check("tanh", lambda x: x.tanh().sum(), np.random.randn(4, 3))
check("exp/log", lambda x: (x.exp().log()).sum(), np.random.randn(4, 3) + 1.0)
check("pow", lambda x: (x ** 3).sum(), np.random.randn(4, 3) + 2.0)
check("max", lambda x: x.max(axis=-1).sum(), np.random.randn(4, 5))
check("softmax", lambda x: softmax(x, axis=-1).sum(), np.random.randn(4, 5))
check("gelu", lambda x: new_gelu(x).sum(), np.random.randn(4, 5))
# -1e9 fill makes finite-difference numerically ill-conditioned at masked
# positions (the function is flat/discontinuous there in float precision).
# Use a finite fill value instead -- same code path, sane numerics.
_mfmask = np.array([[True, False, True], [False, False, False]])
check("masked_fill", lambda x: x.masked_fill(_mfmask, 7.0).sum(), np.random.randn(2, 3))

# --- layernorm-style composite (mean/var/sqrt/broadcast all at once) ---
def ln(x):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return ((x - mean) / (var + 1e-5).sqrt()).sum()

check("layernorm-composite", ln, np.random.randn(4, 6) * 3 + 1)

# --- embedding lookup (getitem with repeated indices -> scatter-add check) ---
def embed_lookup(weight_data):
    w = Tensor(weight_data, requires_grad=True)
    ids = np.array([0, 1, 0, 2])  # index 0 repeated: tests scatter-add correctness
    picked = w[ids]
    return picked.sum().data.item() if False else picked

x = Tensor(np.random.randn(5, 4).copy(), requires_grad=True)
ids = np.array([0, 1, 0, 2])
out = x[ids].sum()
out.backward()
# row 0 should get gradient 2x (read twice), rows 1,2 get 1x, rows 3,4 get 0
expected_counts = np.array([2, 1, 1, 0, 0])
actual_counts = x.grad[:, 0]  # since d(sum)/d(each read elem) = 1
print(f"[{'OK' if np.allclose(actual_counts, expected_counts) else 'FAIL'}] embedding scatter-add   counts={actual_counts.tolist()} expected={expected_counts.tolist()}")
assert np.allclose(actual_counts, expected_counts)

# --- cross entropy vs manual softmax+NLL sanity ---
logits_data = np.random.randn(6, 10)
targets = np.array([2, 5, -100, 1, 9, -100])  # two ignored
logits = Tensor(logits_data.copy(), requires_grad=True)
loss = cross_entropy(logits, targets)
loss.backward()
analytic = logits.grad.copy()

def ce_scalar(data):
    lg = Tensor(data)
    lp = softmax(lg, axis=-1)
    valid = targets != -100
    safe_t = np.where(valid, targets, 0)
    n = data.shape[0]
    picked = lp.data[np.arange(n), safe_t]
    nll = -np.log(picked) * valid
    return nll.sum() / valid.sum()

numeric = numerical_grad(ce_scalar, logits_data.copy())
max_err = np.max(np.abs(analytic - numeric))
print(f"[{'OK' if max_err < 1e-4 else 'FAIL'}] cross_entropy          max|analytic-numeric| = {max_err:.2e}")
assert max_err < 1e-4

print("\nAll gradient checks passed.")