"""
self_attention_numpy.py
========================
The ONE function that is actually unique to a Transformer — everything
else in llm_pretraining.py (AdamW, gradient accumulation, mixed
precision, checkpointing) is generic neural-network training machinery
that would apply equally to a CNN or an RNN.

This is what GPT2Attention._attn() computes internally, made explicit.
"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (subtract max before exponentiating)."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_self_attention(
    x: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    causal: bool = True,
) -> np.ndarray:
    """The self-attention computation, with nothing abstracted away.

    Args:
        x: Input token embeddings, shape (seq_len, d_model).
        W_q, W_k, W_v: Learned projection matrices, each (d_model, d_k).
        causal: If True, apply a causal mask so position i cannot attend
            to positions > i (required for autoregressive pretraining —
            a token must not "see the future" during training).

    Returns:
        Attention output, shape (seq_len, d_k) — a new representation of
        each token, computed as a weighted combination of ALL other
        tokens' Value vectors, where the weights come from how well each
        token's Query matches every other token's Key.
    """
    seq_len, d_model = x.shape
    d_k = W_q.shape[1]

    # Step 1: project the same input into three different roles.
    # This is the "self" in self-attention — Q, K, V all come from x.
    Q = x @ W_q  # "what am I looking for?"      shape (seq_len, d_k)
    K = x @ W_k  # "what do I contain?"          shape (seq_len, d_k)
    V = x @ W_v  # "what do I offer if attended-to?"  shape (seq_len, d_k)

    # Step 2: compatibility score between every pair of tokens.
    scores = Q @ K.T / np.sqrt(d_k)  # shape (seq_len, seq_len)

    # Step 3: causal mask — this is what makes pretraining autoregressive.
    # Position i is only allowed to attend to positions <= i.
    if causal:
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        scores = np.where(mask, -np.inf, scores)

    # Step 4: turn scores into a probability distribution per query token.
    attn_weights = softmax(scores, axis=-1)  # shape (seq_len, seq_len)

    # Step 5: each token's new representation is a weighted sum of ALL
    # tokens' Value vectors, weighted by how relevant each was.
    output = attn_weights @ V  # shape (seq_len, d_k)

    return output


if __name__ == "__main__":
    np.random.seed(0)
    seq_len, d_model, d_k = 5, 8, 8

    x = np.random.randn(seq_len, d_model)
    W_q = np.random.randn(d_model, d_k) * 0.1
    W_k = np.random.randn(d_model, d_k) * 0.1
    W_v = np.random.randn(d_model, d_k) * 0.1

    out = scaled_dot_product_self_attention(x, W_q, W_k, W_v, causal=True)
    print("Self-attention output shape:", out.shape)
    print(out)