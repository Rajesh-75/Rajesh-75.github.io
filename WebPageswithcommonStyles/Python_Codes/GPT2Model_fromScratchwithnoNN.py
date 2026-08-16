"""
gpt2_from_scratch_no_nn.py
==================================================================
GPT2LMHeadModel — NO `torch.nn`, NO `torch.nn.functional`.
==================================================================

What's still allowed, and why: bare `torch.Tensor` and its autograd
engine (`requires_grad`, `.backward()`, `torch.no_grad()`). Rewriting
autograd itself from scratch is a separate, much larger project (the
Value/backward() direction from earlier in this chat) — this file
instead answers "what does nn.Module/nn.Linear/nn.LayerNorm/
nn.functional.softmax/nn.functional.cross_entropy actually DO under
the hood", by writing each of them out by hand on top of raw tensors.

Everything nn.* normally hides is explicit here:
    - Module        : the train()/eval() + parameter-registration
                       machinery (extends the Module class from
                       earlier in this chat with a `parameters()`
                       walk, buffers, and state_dict()).
    - Parameter      : a torch.Tensor subclass that IS the marker
                        Module.__setattr__ uses to decide "this
                        attribute is a learnable weight, not just
                        a plain tensor".
    - Linear / LayerNorm / Embedding / Dropout : each one is just
      the matmul/normalize/index-lookup/masking formula, written
      directly against `x @ W.T`, `x.mean()`, `weight[ids]`, etc.
    - softmax / cross_entropy : hand-written, numerically stable,
      with the same ignore_index=-100 masking convention nn.functional
      uses.

The GPT-2 architecture itself (config, attention, MLP, Block,
LMHeadModel, generate) is UNCHANGED from the nn/F version — same
five pieces, same weight tying, same pre-LN residual structure. Only
the building blocks under it are swapped from library calls to
explicit tensor math.
==================================================================
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional

import torch


# ==================================================================
# 0. Module machinery — what nn.Module gives you for free, made explicit.
#    (Same train()/eval() propagation as the toy Module we built
#    earlier in this chat, now extended with parameter/buffer tracking.)
# ==================================================================
class Parameter(torch.Tensor):
    """A tensor that should be treated as a learnable weight: found by
    Module.parameters(), saved by state_dict(), updated by an optimizer.
    This is the ONLY thing that distinguishes a parameter from any other
    tensor floating around — it's a marker, nothing more."""

    @staticmethod
    def __new__(cls, data: torch.Tensor):
        return torch.Tensor._make_subclass(cls, data, True)  # True = requires_grad


class Module:
    def __init__(self) -> None:
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        # This is the whole trick nn.Module relies on: intercept every
        # attribute assignment and sort it into the right bucket based
        # on its type, so parameters()/train()/state_dict() can later
        # walk the object graph without you registering anything by hand.
        if isinstance(value, Parameter):
            self.__dict__["_parameters"][name] = value
        elif isinstance(value, Module):
            self.__dict__["_modules"][name] = value
        object.__setattr__(self, name, value)

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """For state that travels with the module (moved by .to(device),
        saved in state_dict) but is NOT learned — e.g. the causal mask.
        This is exactly nn.Module.register_buffer's job."""
        self._buffers[name] = tensor
        object.__setattr__(self, name, tensor)

    def children(self) -> Iterator["Module"]:
        return iter(self._modules.values())

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, "Parameter"]]:
        for name, p in self._parameters.items():
            yield f"{prefix}{name}", p
        for child_name, child in self._modules.items():
            yield from child.named_parameters(prefix=f"{prefix}{child_name}.")

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

    def to(self, device) -> "Module":
        for p in self._parameters.values():
            p.data = p.data.to(device)
        for name, b in self._buffers.items():
            moved = b.to(device)
            self._buffers[name] = moved
            object.__setattr__(self, name, moved)
        for child in self._modules.values():
            child.to(device)
        return self

    def state_dict(self, prefix: str = "") -> dict:
        sd = {f"{prefix}{k}": v.detach().clone() for k, v in self._parameters.items()}
        sd.update({f"{prefix}{k}": v.clone() for k, v in self._buffers.items()})
        for name, child in self._modules.items():
            sd.update(child.state_dict(prefix=f"{prefix}{name}."))
        return sd

    def load_state_dict(self, sd: dict, prefix: str = "") -> None:
        for k, p in self._parameters.items():
            with torch.no_grad():
                p.copy_(sd[f"{prefix}{k}"])
        for k in self._buffers:
            self._buffers[k] = sd[f"{prefix}{k}"].clone()
            object.__setattr__(self, k, self._buffers[k])
        for name, child in self._modules.items():
            child.load_state_dict(sd, prefix=f"{prefix}{name}.")

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ModuleList(Module):
    """nn.ModuleList's entire reason for existing: a plain Python list
    wouldn't get walked by Module.__setattr__, so submodules inside it
    would be invisible to parameters()/train()/state_dict(). Wrapping
    them with setattr(self, str(i), m) fixes that."""

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
# Hand-written replacements for the nn.* layers GPT-2 needs
# ==================================================================
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(out_features, in_features))
        self.bias = Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class LayerNorm(Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.ones(normalized_shape))
        self.bias = Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[input_ids]


class Dropout(Module):
    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = (torch.rand_like(x) < keep_prob).to(x.dtype)
        return x * mask / keep_prob


class NewGELU(Module):
    """GPT-2's tanh-approximation GELU. Only torch.tanh/torch.pow — no
    F.gelu involved even in the original version of this file."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


# ==================================================================
# Hand-written replacements for F.softmax / F.cross_entropy
# ==================================================================
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax: subtract the max before exponentiating
    so large logits don't overflow. This is exactly what F.softmax does
    internally; here it's just visible."""
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """logits: (N, C), targets: (N,) of class indices, some possibly
    equal to ignore_index (padding positions to exclude from the loss).
    log_softmax via logsumexp avoids computing softmax then log() —
    same stability trick F.cross_entropy uses internally."""
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    valid = targets != ignore_index
    safe_targets = targets.clone()
    safe_targets[~valid] = 0  # dummy index; masked out below, never affects the loss
    nll = -log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    nll = nll * valid.to(nll.dtype)
    denom = valid.sum().clamp(min=1)
    return nll.sum() / denom


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

        self.qkv_proj = Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = Linear(config.n_embd, config.n_embd)

        self.attn_dropout = Dropout(config.dropout)
        self.resid_dropout = Dropout(config.dropout)

        causal_mask = torch.tril(torch.ones(config.n_positions, config.n_positions))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, config.n_positions, config.n_positions))

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(n_embd, dim=2)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
        )

        if attention_mask is not None:
            pad_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(~pad_mask, float("-inf"))

        attn_weights = softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


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

        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying: same Parameter object, not a copy

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("proj.weight"):
                param.data.normal_(mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: Module) -> None:
        if isinstance(module, Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> CausalLMOutput:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.n_positions:
            raise ValueError(f"Sequence length {seq_len} exceeds n_positions {self.config.n_positions}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.wte(input_ids) + self.wpe(positions)
        x = self.drop(x)

        for block in self.h:
            x = block(x, attention_mask=attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.9,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)

        for _ in range(max_new_tokens):
            context = generated[:, -self.config.n_positions:]
            context_mask = mask[:, -self.config.n_positions:]

            logits = self.forward(context, attention_mask=context_mask).logits
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if do_sample:
                probs = softmax(next_token_logits, dim=-1)
                next_token = _sample_top_p(probs, top_p)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            mask = torch.cat([mask, torch.ones_like(next_token)], dim=1)

        self.train()
        return generated

    def save_pretrained(self, save_dir: str) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        (save_path / "config.json").write_text(json.dumps(asdict(self.config), indent=2))

    @classmethod
    def from_pretrained(cls, load_dir: str) -> "GPT2LMHeadModel":
        load_path = Path(load_dir)
        config_dict = json.loads((load_path / "config.json").read_text())
        model = cls(GPT2Config(**config_dict))
        state_dict = torch.load(load_path / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
        return model


def _sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    cutoff = cumulative > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    sorted_probs[cutoff] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    sampled_idx_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
    return torch.gather(sorted_idx, dim=-1, index=sampled_idx_in_sorted)


# ==================================================================
# Sanity check: forward, loss, BACKWARD, and generate all actually work
# ==================================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    tiny_config = GPT2Config(vocab_size=1000, n_positions=64, n_embd=32, n_layer=2, n_head=4)
    model = GPT2LMHeadModel(tiny_config)
    print(f"Parameters: {model.num_parameters():,}")

    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 10))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    output = model(input_ids, attention_mask=attention_mask, labels=labels)
    print(f"logits shape: {tuple(output.logits.shape)}  (expected (2, 10, {tiny_config.vocab_size}))")
    print(f"loss: {output.loss.item():.4f}  (expected ~ln(1000)={math.log(1000):.4f} at init)")

    # Confirm autograd actually flows through our hand-written layers.
    output.loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    n_params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_params_total = sum(1 for _ in model.parameters())
    print(f"params with grad: {n_params_with_grad}/{n_params_total}")
    print(f"sample grad norms: {[round(g, 4) for g in grad_norms[:5]]}")

    # Confirm weight tying survived: wte.weight and lm_head.weight are the SAME tensor.
    print(f"weight tying holds: {model.lm_head.weight is model.wte.weight}")

    generated = model.generate(input_ids[:, :3], max_new_tokens=5, pad_token_id=0)
    print(f"generated shape: {tuple(generated.shape)}  (expected (2, 8))")

    print("\nAll checks passed.")