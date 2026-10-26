"""
composite_pattern_contrast.py
==================================================================
Two implementations of the same problem -- propagate a train/eval
flag through a nested layer tree -- side by side in one file:

    Version A: no design pattern, plain if/isinstance branching.
    Version B: the Composite pattern (a uniform Module interface).

main() runs both and prints the same demonstration for each: normal
propagation working, then what happens when a new layer type
(BatchNorm) is added later without the traversal code's author in mind.
==================================================================
"""

import numpy as np


# ==================================================================
# VERSION A: no Composite pattern -- caller uses if/isinstance checks
# ==================================================================
class LinearNaive:
    def __init__(self, in_f, out_f):
        self.weight = np.random.randn(in_f, out_f) * 0.02


class DropoutNaive:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True


class BatchNormNaive:
    def __init__(self, dim):
        self.training = True  # also behaves differently in train vs eval


class MLPNaive:
    def __init__(self, with_batchnorm=False):
        self.fc = LinearNaive(8, 32)
        self.dropout = DropoutNaive(0.5)
        if with_batchnorm:
            self.bn = BatchNormNaive(32)
        self.proj = LinearNaive(32, 8)


class BlockNaive:
    def __init__(self, with_batchnorm=False):
        self.attn_dropout = DropoutNaive(0.3)
        self.mlp = MLPNaive(with_batchnorm=with_batchnorm)


class ModelNaive:
    def __init__(self, with_batchnorm=False):
        self.blocks = [BlockNaive(with_batchnorm=with_batchnorm), BlockNaive(with_batchnorm=with_batchnorm)]
        self.final_dropout = DropoutNaive(0.1)


def set_train_mode_naive(obj, mode):
    """The caller has to already know the shape of the tree and write
    a branch for every container type it might encounter. Crucially,
    this function was written BEFORE BatchNormNaive existed, so its
    MLPNaive branch only knows to recurse into .dropout -- it has no
    idea .bn might exist, and isinstance() gives no signal that it's
    missing something."""
    if isinstance(obj, ModelNaive):
        for block in obj.blocks:
            set_train_mode_naive(block, mode)
        set_train_mode_naive(obj.final_dropout, mode)
    elif isinstance(obj, BlockNaive):
        set_train_mode_naive(obj.attn_dropout, mode)
        set_train_mode_naive(obj.mlp, mode)
    elif isinstance(obj, MLPNaive):
        set_train_mode_naive(obj.dropout, mode)
        # NOTE: .fc and .proj are LinearNaive (no .training, correctly
        # skipped) but .bn -- if present -- is ALSO silently skipped,
        # because nobody updated this branch when BatchNormNaive was added.
    elif isinstance(obj, DropoutNaive):
        obj.training = mode
    else:
        raise TypeError(f"set_train_mode_naive doesn't know how to handle {type(obj).__name__}")


def demo_naive():
    print("=== Version A: if/isinstance based (no design pattern) ===")

    model = ModelNaive()
    set_train_mode_naive(model, False)
    print("Case 1 -- original tree, no BatchNorm:")
    print(f"  all dropouts set to eval: "
          f"{model.blocks[0].attn_dropout.training}, "
          f"{model.blocks[0].mlp.dropout.training}, "
          f"{model.blocks[1].mlp.dropout.training}, "
          f"{model.final_dropout.training}")

    print("\nCase 2 -- BatchNorm added later by someone extending MLPNaive:")
    model2 = ModelNaive(with_batchnorm=True)
    set_train_mode_naive(model2, False)  # runs with NO error -- that's the trap
    bn_state = model2.blocks[0].mlp.bn.training
    print(f"  set_train_mode_naive(model2, False) ran with no error or warning.")
    print(f"  model2.blocks[0].mlp.bn.training = {bn_state}  (still True -- SILENTLY missed)")
    print(f"  isinstance(obj, MLPNaive) matched fine; the branch just never looks at .bn.")
    print(f"  BatchNorm would run in training mode forever, including at inference time.")


# ==================================================================
# VERSION B: Composite pattern -- uniform Module interface
# ==================================================================
class Module:
    def __init__(self):
        self._modules = {}
        self.training = True

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self.__dict__.setdefault("_modules", {})[name] = value
        object.__setattr__(self, name, value)

    def train(self, mode=True):
        self.training = mode
        for child in self._modules.values():
            child.train(mode)  # every Module, leaf or container, answers the SAME call
        return self

    def eval(self):
        return self.train(False)


class Linear(Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = np.random.randn(in_f, out_f) * 0.02
    # no train() override needed -- Linear doesn't care about the flag,
    # and it still participates correctly via the inherited base method


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p


class BatchNorm(Module):
    """A brand-new layer type, added later. Zero changes required to
    Module, MLP, Block, or Model -- inheriting Module is the whole
    integration step."""
    def __init__(self, dim):
        super().__init__()


class MLP(Module):
    def __init__(self, with_batchnorm=False):
        super().__init__()
        self.fc = Linear(8, 32)
        self.dropout = Dropout(0.5)
        if with_batchnorm:
            self.bn = BatchNorm(32)  # <-- adding this required NO other code changes
        self.proj = Linear(32, 8)


class Block(Module):
    def __init__(self, with_batchnorm=False):
        super().__init__()
        self.attn_dropout = Dropout(0.3)
        self.mlp = MLP(with_batchnorm=with_batchnorm)


class Model(Module):
    def __init__(self, with_batchnorm=False):
        super().__init__()
        self.block0 = Block(with_batchnorm=with_batchnorm)
        self.block1 = Block(with_batchnorm=with_batchnorm)
        self.final_dropout = Dropout(0.1)


def demo_composite():
    print("\n=== Version B: Composite pattern ===")

    model = Model()
    model.eval()
    print("Case 1 -- original tree, no BatchNorm:")
    print(f"  all dropouts set to eval: "
          f"{model.block0.attn_dropout.training}, "
          f"{model.block0.mlp.dropout.training}, "
          f"{model.block1.mlp.dropout.training}, "
          f"{model.final_dropout.training}")

    print("\nCase 2 -- BatchNorm added later, same Model/Block/MLP classes:")
    model2 = Model(with_batchnorm=True)
    model2.eval()  # ONE call, no knowledge of tree shape needed
    print(f"  all dropouts + batchnorm set to eval: "
          f"{model2.block0.attn_dropout.training}, "
          f"{model2.block0.mlp.dropout.training}, "
          f"{model2.block0.mlp.bn.training}, "
          f"{model2.final_dropout.training}")
    print(f"  BatchNorm picked up automatically -- no traversal code was touched.")


# ==================================================================
def main():
    demo_naive()
    demo_composite()

    print("\n=== Summary ===")
    print("Version A: correctness depends on remembering to update the traversal")
    print("           function every time the tree grows. Forgetting fails silently.")
    print("Version B: correctness follows from the class hierarchy alone.")
    print("           New layer types are wired in for free by inheriting Module.")


if __name__ == "__main__":
    main()