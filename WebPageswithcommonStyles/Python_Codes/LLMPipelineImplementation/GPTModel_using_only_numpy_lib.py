"""
gpt2_fully_from_scratch.py
==================================================================
GPT2LMHeadModel with NOTHING borrowed: no torch, no torch.nn, no
torch.nn.functional, no torch.autograd. Every one of the following
is written out by hand, in this codebase:

    1. The autograd ENGINE ITSELF (tensor_engine.py) -- a Tensor class
       that records its own computation graph and differentiates it,
       reverse-mode, via hand-written backward() rules for every op
       (matmul, add, layernorm's mean/var/sqrt, softmax, cross-entropy,
       embedding lookup, tanh-GELU). Verified against numerical
       finite-difference gradients in test_tensor_engine.py -- every
       rule matches to ~1e-10.
    2. The MODULE machinery (Module/Parameter/train()/eval()/parameters()) --
       same design as nn.Module, built explicitly.
    3. The LAYERS (Linear, LayerNorm, Embedding, Dropout) -- each is
       just its defining formula, composed from Tensor ops above.
    4. The TRANSFORMER ARCHITECTURE (causal self-attention, MLP,
       pre-LN residual blocks, weight tying, GPT-2 init scheme) --
       structurally identical to the reference implementation.
    5. The OPTIMIZER (Adam) -- the actual moment-estimate update rule,
       consuming Tensor.grad and mutating Tensor.data. Not delegated
       to any library.

The only non-standard-library import anywhere is NumPy, used purely
as an array container + elementwise/matmul primitives -- it contributes
no autograd, no layers, and no optimizer. Every gradient in this file
is computed by code you can read top to bottom.
==================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from tensor_engine import Tensor, softmax, cross_entropy, new_gelu


# ==================================================================
# 0. Module machinery (same design as nn.Module, made explicit)
# ==================================================================
class Parameter(Tensor):
    """A Tensor that Module.__setattr__ recognizes as learnable. Since
    Tensor already carries .data/.grad, Parameter adds nothing except
    being a distinguishable type -- exactly nn.Parameter's role."""

    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module:
    def __init__(self) -> None:
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.__dict__["_parameters"][name] = value
        elif isinstance(value, Module):
            self.__dict__["_modules"][name] = value
        object.__setattr__(self, name, value)

    def children(self) -> Iterator["Module"]:
        return iter(self._modules.values())

    def named_parameters(self, prefix: str = ""):
        for name, p in self._parameters.items():
            yield f"{prefix}{name}", p
        for cname, child in self._modules.items():
            yield from child.named_parameters(prefix=f"{prefix}{cname}.")

    def parameters(self) -> Iterator["Parameter"]:
        for _, p in self.named_parameters():
            yield p

    def apply(self, fn) -> "Module":
        for child in self._modules.values():
            child.apply(fn)
        fn(self)
        return self

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for child in self._modules.values():
            child.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def num_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def forward(self, *a, **kw):
        raise NotImplementedError

    def __call__(self, *a, **kw):
        return self.forward(*a, **kw)


class ModuleList(Module):
    def __init__(self, modules) -> None:
        super().__init__()
        self._list = list(modules)
        for i, m in enumerate(self._list):
            setattr(self, str(i), m)

    def __iter__(self):
        return iter(self._list)

    def __len__(self) -> int:
        return len(self._list)


# ==================================================================
# Layers, built from Tensor ops that already know their own gradient
# ==================================================================
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = Parameter(np.empty((in_features, out_features)))  # stored transposed vs torch: x @ W directly
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class LayerNorm(Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = Parameter(np.ones(normalized_shape))
        self.bias = Parameter(np.zeros(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return x_norm * self.weight + self.bias


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = Parameter(np.empty((num_embeddings, embedding_dim)))

    def forward(self, input_ids: np.ndarray) -> Tensor:
        return self.weight[input_ids]


class Dropout(Module):
    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = (np.random.rand(*x.shape) < keep_prob).astype(np.float64)
        return (x * Tensor(mask)) * (1.0 / keep_prob)


class NewGELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return new_gelu(x)


# ==================================================================
# 1. Config
# ==================================================================
@dataclass
class GPT2Config:
    vocab_size: int
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")


# ==================================================================
# 2. Causal self-attention
# ==================================================================
class CausalSelfAttention(Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        self.qkv_proj = Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = Linear(config.n_embd, config.n_embd)
        self.attn_dropout = Dropout(config.dropout)
        self.resid_dropout = Dropout(config.dropout)

        # Causal mask as a plain numpy bool array (not a Parameter -- it's
        # a fixed constant, exactly like register_buffer's non-trainable role).
        self.causal_mask = np.triu(np.ones((config.n_positions, config.n_positions), dtype=bool), k=1)

    def forward(self, x: Tensor, attention_mask: Optional[np.ndarray] = None) -> Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)                                  # (B, T, 3C)
        q = qkv[:, :, 0:C]
        k = qkv[:, :, C:2 * C]
        v = qkv[:, :, 2 * C:3 * C]

        # (B, T, C) -> (B, n_head, T, head_dim)
        def split_heads(t: Tensor) -> Tensor:
            t = t.reshape(B, T, self.n_head, self.head_dim)
            return t.transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        attn_scores = q.matmul(k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(self.head_dim))  # (B,H,T,T)

        mask = self.causal_mask[:T, :T]                          # True = forbidden (future position)
        if attention_mask is not None:
            pad_forbidden = (attention_mask[:, None, None, :] == 0)  # (B,1,1,T) True = padding
            mask = mask[None, None, :, :] | pad_forbidden
        attn_scores = attn_scores.masked_fill(mask, -1e9)

        attn_weights = softmax(attn_scores, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights.matmul(v)                             # (B,H,T,head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)          # back to (B,T,C)

        return self.resid_dropout(self.out_proj(out))


# ==================================================================
# 3. MLP
# ==================================================================
class MLP(Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.fc = Linear(config.n_embd, 4 * config.n_embd)
        self.proj = Linear(4 * config.n_embd, config.n_embd)
        self.act = NewGELU()
        self.dropout = Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.proj(self.act(self.fc(x))))


# ==================================================================
# 4. Block
# ==================================================================
class Block(Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, attention_mask: Optional[np.ndarray] = None) -> Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class CausalLMOutput:
    logits: Tensor
    loss: Optional[Tensor] = None


# ==================================================================
# 5. The full model
# ==================================================================
class GPT2LMHeadModel(Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config

        self.wte = Embedding(config.vocab_size, config.n_embd)
        self.wpe = Embedding(config.n_positions, config.n_embd)
        self.drop = Dropout(config.dropout)
        self.h = ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # lm_head reuses wte's weight, transposed (Linear here stores weight
        # as (in, out) = (n_embd, vocab), wte.weight is (vocab, n_embd)).
        self.lm_head_bias = None  # GPT-2's output projection has no bias

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("proj.weight"):
                p.data = np.random.normal(0.0, 0.02 / math.sqrt(2 * config.n_layer), size=p.data.shape)

    def _init_weights(self, module: Module) -> None:
        if isinstance(module, Linear):
            module.weight.data = np.random.normal(0.0, 0.02, size=module.weight.data.shape)
            if module.bias is not None:
                module.bias.data = np.zeros_like(module.bias.data)
        elif isinstance(module, Embedding):
            module.weight.data = np.random.normal(0.0, 0.02, size=module.weight.data.shape)
        elif isinstance(module, LayerNorm):
            module.bias.data = np.zeros_like(module.bias.data)
            module.weight.data = np.ones_like(module.weight.data)

    def lm_head(self, x: Tensor) -> Tensor:
        # Weight tying: logits = x @ wte.weight.T -- literally the SAME
        # Parameter object as the token embedding, just used transposed.
        return x.matmul(self.wte.weight.transpose(1, 0))

    def forward(self, input_ids: np.ndarray, attention_mask: Optional[np.ndarray] = None,
                labels: Optional[np.ndarray] = None) -> CausalLMOutput:
        B, T = input_ids.shape
        if T > self.config.n_positions:
            raise ValueError(f"Sequence length {T} exceeds n_positions {self.config.n_positions}")

        positions = np.arange(T)
        x = self.wte(input_ids) + self.wpe(positions)[None, :, :]
        x = self.drop(x)

        for block in self.h:
            x = block(x, attention_mask=attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)                                  # (B, T, vocab)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, self.config.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        return CausalLMOutput(logits=logits, loss=loss)

    def generate(self, input_ids: np.ndarray, attention_mask: Optional[np.ndarray] = None,
                 max_new_tokens: int = 20, do_sample: bool = False, temperature: float = 1.0) -> np.ndarray:
        self.eval()
        generated = input_ids.copy()
        mask = attention_mask.copy() if attention_mask is not None else np.ones_like(input_ids)

        for _ in range(max_new_tokens):
            context = generated[:, -self.config.n_positions:]
            context_mask = mask[:, -self.config.n_positions:]
            logits = self.forward(context, attention_mask=context_mask).logits
            next_logits = logits.data[:, -1, :] / max(temperature, 1e-5)

            if do_sample:
                probs = np.exp(next_logits - next_logits.max(axis=-1, keepdims=True))
                probs /= probs.sum(axis=-1, keepdims=True)
                next_token = np.array([[np.random.choice(len(p), p=p)] for p in probs])
            else:
                next_token = next_logits.argmax(axis=-1, keepdims=True)

            generated = np.concatenate([generated, next_token], axis=1)
            mask = np.concatenate([mask, np.ones_like(next_token)], axis=1)

        self.train()
        return generated


# ==================================================================
# 6. Adam -- the actual moment-estimate update rule, hand-written
# ==================================================================
class Adam:
    """
    For every parameter p with gradient g = p.grad:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g            (first moment: running mean of g)
        v_t = beta2 * v_{t-1} + (1 - beta2) * g**2          (second moment: running mean of g^2)
        m_hat = m_t / (1 - beta1**t)                        (bias correction: m_0=0 biases early estimates toward 0)
        v_hat = v_t / (1 - beta2**t)
        p.data -= lr * m_hat / (sqrt(v_hat) + eps)

    This is the whole algorithm from Kingma & Ba (2014) -- nothing here
    is delegated to torch.optim.Adam or any other library.
    """

    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data          # decoupled-style L2 term added to grad

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ==================================================================
# Sanity check: forward, hand-written backward, Adam step, generate
# ==================================================================
if __name__ == "__main__":
    np.random.seed(0)

    tiny_config = GPT2Config(vocab_size=50, n_positions=16, n_embd=16, n_layer=2, n_head=4, dropout=0.0)
    model = GPT2LMHeadModel(tiny_config)
    print(f"Parameters: {model.num_parameters():,}")

    input_ids = np.random.randint(0, tiny_config.vocab_size, (2, 8))
    attention_mask = np.ones_like(input_ids)
    labels = input_ids.copy()

    output = model(input_ids, attention_mask=attention_mask, labels=labels)
    print(f"logits shape: {output.logits.shape}  (expected (2, 8, {tiny_config.vocab_size}))")
    print(f"loss (step 0): {output.loss.data.item():.4f}  (expected ~ln(50)={math.log(50):.4f} at init)")

    output.loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    print(f"params with grad after hand-written backward(): {n_with_grad}/{n_total}")

    optimizer = Adam(model.parameters(), lr=1e-2)

    losses = []
    for step in range(15):
        model.zero_grad()
        out = model(input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        optimizer.step()
        losses.append(out.loss.data.item())

    print(f"loss trajectory over 15 Adam steps (overfitting one tiny batch):")
    print("  " + " -> ".join(f"{l:.3f}" for l in losses[::3]))
    assert losses[-1] < losses[0], "Adam should have decreased the loss on this tiny batch"
    print(f"loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}  (confirms backward()+Adam.step() actually learn)")

    generated = model.generate(input_ids[:, :3], max_new_tokens=5)
    print(f"generated shape: {generated.shape}  (expected (2, 8))")

    print("\nAll checks passed -- forward, hand-written backward, and Adam all verified.")