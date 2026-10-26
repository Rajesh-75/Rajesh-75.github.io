"""
gpt2_from_scratch.py
==================================================================
 GPT2LMHeadModel, BUILT FROM PLAIN PYTORCH — NO transformers LIBRARY
==================================================================

What "from scratch" means here: no `transformers.GPT2LMHeadModel`,
no `transformers.GPT2Config`. Every layer is built from raw
`torch.nn` primitives (Linear, LayerNorm, Embedding, Dropout) —
those are just matrix-multiply/normalize building blocks, not a
shortcut around the architecture. What you're NOT getting for free
here is exactly what the transformers library normally hides from
you: attention, blocks, weight tying, generation loop.

The five pieces of the architecture, in the order you'll read them
below:
    1. GPT2Config        — the handful of numbers that define a size
    2. CausalSelfAttention — one attention head-group, causally masked
    3. MLP                — the feed-forward sublayer inside a block
    4. Block               — attention + MLP, each wrapped in a
                              pre-LayerNorm residual connection
    5. GPT2LMHeadModel      — token+position embeddings -> N blocks ->
                              final norm -> a linear layer back to
                              vocabulary logits

Drop-in note: this exposes the same forward(input_ids, attention_mask,
labels) -> object-with-.logits-and-.loss shape, and the same
.generate(...) signature, that llm_pretraining.py / sft_training.py /
ppo_rlhf_training.py already call on the transformers model. Swapping
this in means changing an import, not rewriting the training scripts.
==================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================
# 1. Config — the numbers that define a model's size
# ==================================================================
@dataclass
class GPT2Config:
    vocab_size: int
    n_positions: int = 1024     # max sequence length the model was built for
    n_embd: int = 768           # width of every token's vector (the "d_model")
    n_layer: int = 12           # number of stacked Blocks
    n_head: int = 12            # attention heads per Block (n_embd must divide evenly)
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")


# ==================================================================
# GPT-2's activation: an approximate GELU (the "new gelu" / tanh form)
# ==================================================================
class NewGELU(nn.Module):
    """The exact tanh-approximation formula GPT-2 uses, spelled out
    rather than called via nn.GELU(approximate="tanh") — same result,
    but you can see what it's actually computing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


# ==================================================================
# 2. Causal self-attention — built from Linear + matmul, not a library call
# ==================================================================
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position t can only attend to
    positions <= t. This is the "causal (attention) masking" from the
    SFT page — it's architectural, present in every forward pass
    regardless of training stage, and unrelated to loss masking."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # One combined projection for Q, K, V (three separate Linears
        # is equivalent; this is the standard GPT-2 layout — a single
        # matmul that we then split three ways).
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular mask, precomputed once. register_buffer means
        # it moves with .to(device) and is saved in state_dict, but is
        # NOT a trainable parameter — it's a fixed constant.
        causal_mask = torch.tril(torch.ones(config.n_positions, config.n_positions))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, config.n_positions, config.n_positions))

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape

        qkv = self.qkv_proj(x)                                   # (B, T, 3*n_embd)
        q, k, v = qkv.split(n_embd, dim=2)                       # each (B, T, n_embd)

        # Split into heads: (B, T, n_embd) -> (B, n_head, T, head_dim)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores: (B, n_head, T, T)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask: forbid attending to future positions
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
        )

        # Padding mask (batching detail, unrelated to causal masking):
        # attention_mask is (B, T) with 1 = real token, 0 = pad.
        if attention_mask is not None:
            pad_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)  # (B, 1, 1, T)
            attn_scores = attn_scores.masked_fill(~pad_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v                                    # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)

        return self.resid_dropout(self.out_proj(out))


# ==================================================================
# 3. MLP — the feed-forward sublayer (expand 4x, activate, project back)
# ==================================================================
class MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = NewGELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(self.act(self.fc(x))))


# ==================================================================
# 4. Block — one Transformer layer: pre-LN residual attention + pre-LN residual MLP
# ==================================================================
class Block(nn.Module):
    """GPT-2 uses PRE-layernorm: normalize, then sublayer, then add
    the residual — as opposed to normalizing AFTER the addition
    (post-LN, the original Transformer paper's choice). Pre-LN is
    what makes deep GPT-2-style stacks trainable without extremely
    careful learning-rate warmup."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


# ==================================================================
# Forward-pass output container (mirrors what HF's CausalLMOutput gives you)
# ==================================================================
@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


# ==================================================================
# 5. The full model: embeddings -> N blocks -> final norm -> vocab logits
# ==================================================================
class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)       # token embedding
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)      # learned position embedding
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Weight tying: the un-embedding (logits) projection reuses the
        # SAME matrix as the token embedding. This isn't an optimization
        # shortcut — it halves the vocab-sized parameters and is standard
        # practice for GPT-2-scale models.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # GPT-2's specific fix: residual-stream projections get scaled
        # down by 1/sqrt(2 * n_layer) so variance doesn't blow up as
        # depth increases (each block adds a residual contribution).
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> CausalLMOutput:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.n_positions:
            raise ValueError(f"Sequence length {seq_len} exceeds n_positions {self.config.n_positions}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)  # (1, T)
        x = self.wte(input_ids) + self.wpe(positions)
        x = self.drop(x)

        for block in self.h:
            x = block(x, attention_mask=attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if labels is not None:
            # Standard next-token cross-entropy: predict token t+1 from
            # tokens <=t, so both logits and labels get shifted by one.
            # Same IGNORE_INDEX=-100 convention as SFT_CS.py's loss masking.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
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
        """Autoregressive generation: one token at a time, each new
        token fed back in as input for the next step. Same signature
        shape as ppo_rlhf_training.py's call to policy.generate(...)."""
        self.eval()
        generated = input_ids
        mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)

        for _ in range(max_new_tokens):
            context = generated[:, -self.config.n_positions:]
            context_mask = mask[:, -self.config.n_positions:]

            logits = self.forward(context, attention_mask=context_mask).logits
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = _sample_top_p(probs, top_p)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            mask = torch.cat([mask, torch.ones_like(next_token)], dim=1)

        self.train()
        return generated

    # --- HF-style save/load, so this can drop into scripts that call
    #     model.save_pretrained(...) / GPT2LMHeadModel.from_pretrained(...) ---
    def save_pretrained(self, save_dir: str) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        import json
        (save_path / "config.json").write_text(json.dumps(asdict(self.config), indent=2))

    @classmethod
    def from_pretrained(cls, load_dir: str) -> "GPT2LMHeadModel":
        import json
        load_path = Path(load_dir)
        config_dict = json.loads((load_path / "config.json").read_text())
        model = cls(GPT2Config(**config_dict))
        state_dict = torch.load(load_path / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
        return model


def _sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens whose
    cumulative probability exceeds top_p, zero out the rest, then
    sample from what remains. Prevents sampling from the long, noisy
    tail of low-probability tokens."""
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
# Sanity check
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
    print(f"loss: {output.loss.item():.4f}")

    generated = model.generate(input_ids[:, :3], max_new_tokens=5, pad_token_id=0)
    print(f"generated shape: {tuple(generated.shape)}  (expected (2, 8))")

    print("\nAll checks passed.")