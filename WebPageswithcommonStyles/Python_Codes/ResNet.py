"""
ResNet block, from scratch, with forward + backward pass fully spelled out.

The goal is to make two things visible in the code itself:
  1. FORWARD:  out = F(x) + x            -- the residual addition
  2. BACKWARD: d_x = d_F_path + d_identity_path
               (the gradient SPLITS at the addition node and both halves
                are summed back together before continuing to layer l-1)

No deep learning framework is used -- conv, ReLU, and their backward passes
are all implemented explicitly so nothing is hidden inside autograd.
"""

import numpy as np


class ReLU:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, d_out):
        return d_out * self.mask


class Conv2D:
    """3x3 'same' convolution, multi-channel, stride 1. Loops are kept
    explicit (rather than im2col) so the correspondence to the math is
    easy to trace, at the cost of speed."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        # He initialization -- correct choice here since ReLU follows every conv
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)
        self.k = kernel_size
        self.p = padding

    def forward(self, x):
        # x: (C_in, H, W)
        self.last_input = x
        C_in, H, W = x.shape
        C_out, k, p = self.W.shape[0], self.k, self.p
        x_padded = np.pad(x, ((0, 0), (p, p), (p, p)))
        out = np.zeros((C_out, H, W))
        for f in range(C_out):
            for i in range(H):
                for j in range(W):
                    patch = x_padded[:, i:i + k, j:j + k]          # (C_in, k, k)
                    out[f, i, j] = np.sum(patch * self.W[f]) + self.b[f]
        return out

    def backward(self, d_out, lr=0.01):
        C_in, H, W = self.last_input.shape
        C_out, k, p = self.W.shape[0], self.k, self.p
        x_padded = np.pad(self.last_input, ((0, 0), (p, p), (p, p)))
        out_h, out_w = d_out.shape[1], d_out.shape[2]

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        d_x_padded = np.zeros_like(x_padded)

        for f in range(C_out):
            db[f] = np.sum(d_out[f])
            for c in range(C_in):
                for ki in range(k):
                    for kj in range(k):
                        # dW: cross-correlation of the input patch with d_out
                        patch = x_padded[c, ki:ki + out_h, kj:kj + out_w]
                        dW[f, c, ki, kj] = np.sum(patch * d_out[f])
                        # d_x: scatter d_out back, weighted by the kernel value --
                        # equivalent to correlating d_out with the 180-rotated kernel
                        d_x_padded[c, ki:ki + out_h, kj:kj + out_w] += self.W[f, c, ki, kj] * d_out[f]

        d_x = d_x_padded[:, p:p + H, p:p + W] if p > 0 else d_x_padded
        self.W -= lr * dW
        self.b -= lr * db
        return d_x


class PlainBlock:
    """conv -> relu -> conv -> relu. No shortcut. Output is purely H(x)."""

    def __init__(self, channels):
        self.conv1 = Conv2D(channels, channels)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(channels, channels)
        self.relu_out = ReLU()

    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.relu_out.forward(out)
        return out

    def backward(self, d_out, lr=0.01):
        d = self.relu_out.backward(d_out)
        d = self.conv2.backward(d, lr)
        d = self.relu1.backward(d)
        d = self.conv1.backward(d, lr)      # this is the ONLY path back to x -- nothing else
        return d


class ResidualBlock:
    """conv -> relu -> conv, THEN add the original input, THEN relu."""

    def __init__(self, channels):
        self.conv1 = Conv2D(channels, channels)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(channels, channels)
        self.relu_out = ReLU()

    def forward(self, x):
        self.identity = x                    # <-- save x: this IS the skip connection
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)        # out is now F(x), before the addition
        self.F_x = out
        out = out + self.identity            # <<< THE RESIDUAL ADDITION: F(x) + x
        out = self.relu_out.forward(out)     # ReLU is applied AFTER the addition
        return out

    def backward(self, d_out, lr=0.01):
        d_add = self.relu_out.backward(d_out)

        # <<< THE ADDITION NODE'S BACKWARD RULE:
        # d(a+b)/da = 1 and d(a+b)/db = 1, so the incoming gradient is
        # COPIED UNCHANGED down both branches -- nothing is split proportionally.
        d_F_branch = d_add
        d_identity_branch = d_add

        # backprop through the conv path only
        d = self.conv2.backward(d_F_branch, lr)
        d = self.relu1.backward(d)
        d_conv_path = self.conv1.backward(d, lr)

        # <<< RECOMBINE: the two branches' gradients are SUMMED before
        # continuing to layer l-1. This sum is what guarantees d_x can
        # never be smaller than d_identity_branch alone.
        d_x = d_conv_path + d_identity_branch
        return d_x


if __name__ == "__main__":
    np.random.seed(0)
    C, H, W = 4, 8, 8
    x = np.random.randn(C, H, W) * 0.1

    print("=" * 60)
    print("PART 1 -- one residual block, forward pass")
    print("=" * 60)
    block = ResidualBlock(channels=C)
    out = block.forward(x)
    print(f"||x||             = {np.linalg.norm(x):.4f}   (the identity/skip value)")
    print(f"||F(x)||          = {np.linalg.norm(block.F_x):.4f}   (the conv path's own output)")
    print(f"||F(x) + x||      = {np.linalg.norm(block.F_x + x):.4f}   (what actually moves to the next block)")

    print()
    print("=" * 60)
    print("PART 2 -- same block, backward pass: watch the gradient split")
    print("=" * 60)
    d_out = np.ones_like(out)
    d_x = block.backward(d_out, lr=0.0)   # lr=0: inspect gradients without updating weights
    print(f"||d_out|| (incoming gradient)          = {np.linalg.norm(d_out):.4f}")
    print(f"||d_x||   (gradient reaching the input) = {np.linalg.norm(d_x):.4f}")
    print("d_x = d_conv_path + d_identity_branch  -- the identity branch guarantees")
    print("d_x can never be smaller than d_identity_branch alone, no matter how small")
    print("the conv path's own gradient becomes.")

    print()
    print("=" * 60)
    print("PART 3 -- stack 20 blocks with deliberately weak weights:")
    print("does the gradient survive to the input? Plain vs Residual.")
    print("=" * 60)
    depth = 20

    plain_blocks = [PlainBlock(channels=C) for _ in range(depth)]
    res_blocks = [ResidualBlock(channels=C) for _ in range(depth)]

    # Shrink every conv weight to simulate poorly-conditioned layers --
    # this is the scenario where the degradation problem shows up.
    for b in plain_blocks + res_blocks:
        b.conv1.W *= 0.3
        b.conv2.W *= 0.3

    # --- plain network ---
    h = x.copy()
    for b in plain_blocks:
        h = b.forward(h)
    d = np.ones_like(h)
    for b in reversed(plain_blocks):
        d = b.backward(d, lr=0.0)
    print(f"Plain CNN, {depth} layers:  ||gradient at input|| = {np.linalg.norm(d):.10f}")

    # --- residual network ---
    h = x.copy()
    for b in res_blocks:
        h = b.forward(h)
    d = np.ones_like(h)
    for b in reversed(res_blocks):
        d = b.backward(d, lr=0.0)
    print(f"ResNet,    {depth} layers:  ||gradient at input|| = {np.linalg.norm(d):.10f}")