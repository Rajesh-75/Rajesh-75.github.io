"""
Fine-tuning a pretrained CNN with selectively frozen layers.

Demonstrates the standard transfer-learning pattern:
  1. Load a pretrained backbone (ResNet-50 here, but the freezing logic
     generalizes to any torchvision CNN).
  2. Freeze early layers (generic edge/texture/shape features) so their
     weights are NOT updated.
  3. Unfreeze later layers + replace the classification head, so only
     these get gradient updates.
  4. Use differential learning rates: a small LR for any unfrozen
     backbone layers, a larger LR for the new head — since the head
     starts from random init and needs to move faster.

This is the pattern used e.g. for adapting ImageNet-pretrained ResNet-50
to a domain-specific task (chest X-ray classification, etc.) with a
small labeled dataset.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models


# ---------------------------------------------------------------------
# 1. Build model + apply freezing
# ---------------------------------------------------------------------

def build_finetune_model(num_classes: int, freeze_until: str = "layer3"):
    """
    Loads a pretrained ResNet-50 and freezes everything up to (and
    including) `freeze_until`. Layers after it, plus the new head,
    remain trainable.

    ResNet-50's top-level children, in order, are:
        conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc

    freeze_until="layer3" is a common choice: keep the generic
    low/mid-level filters (edges, textures, simple shapes) fixed, and
    let layer4 (high-level, task-specific features) + fc adapt.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Replace the head first, so it's trainable by construction.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Walk the named children in definition order; freeze everything
    # up to and including `freeze_until`, leave the rest trainable.
    freezing = True
    for name, module in model.named_children():
        if freezing:
            for param in module.parameters():
                param.requires_grad = False
        if name == freeze_until:
            freezing = False  # everything after this stays trainable

    return model


def trainable_param_groups(model, backbone_lr=1e-5, head_lr=1e-3):
    """
    Splits trainable parameters into backbone vs. head groups so the
    optimizer can apply different learning rates to each — the
    unfrozen backbone layers should move slowly (they're already
    well-initialized), while the new head needs a larger LR to
    converge from random init.
    """
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    groups = [{"params": head_params, "lr": head_lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    return groups


# ---------------------------------------------------------------------
# 2. Training loop
# ---------------------------------------------------------------------

def train_one_epoch(model, loader: DataLoader, optimizer, criterion, device):
    model.train()  # BatchNorm/Dropout in frozen layers still matters —
                    # see note below.
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()   # gradients only accumulate on params with
                           # requires_grad=True; frozen params get none
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader: DataLoader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# ---------------------------------------------------------------------
# 3. Putting it together
# ---------------------------------------------------------------------

def finetune(train_loader, val_loader, num_classes, epochs=10,
             freeze_until="layer3", backbone_lr=1e-5, head_lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_finetune_model(num_classes, freeze_until=freeze_until).to(device)
    param_groups = trainable_param_groups(model, backbone_lr, head_lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    return model


if __name__ == "__main__":
    # Wire up your own DataLoaders (ImageFolder, custom Dataset, etc.)
    # then call, e.g.:
    #
    #   model = finetune(train_loader, val_loader, num_classes=2,
    #                     epochs=15, freeze_until="layer3")
    #
    pass
