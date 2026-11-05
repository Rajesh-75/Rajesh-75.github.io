"""Model construction and the freezing helpers."""
import torch.nn as nn
from torchvision import models

_ARCHS = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2),
}


def build_model(arch: str, num_classes: int, dropout: float, pretrained: bool = True) -> nn.Module:
    if arch not in _ARCHS:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {list(_ARCHS)}")
    fn, weights = _ARCHS[arch]
    model = fn(weights=weights if pretrained else None)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    return model


def set_trainable(model: nn.Module, layers) -> None:
    """Freeze everything, then unfreeze only the named top-level blocks."""
    for p in model.parameters():
        p.requires_grad = False
    for name in layers:
        for p in getattr(model, name).parameters():
            p.requires_grad = True


def set_train_mode(model: nn.Module) -> None:
    """model.train(), but frozen BatchNorm layers stay in eval mode so their
    ImageNet running statistics are not overwritten by medical batches."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not next(m.parameters()).requires_grad:
            m.eval()
