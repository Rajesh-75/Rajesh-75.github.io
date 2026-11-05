"""Training step, evaluation, clinical metrics and probability calibration."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score

from .model import set_train_mode


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, amp, grad_clip) -> float:
    set_train_mode(model)
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            loss = criterion(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss: check the data and learning rates.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                       # so clipping sees true gradient norms
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.inference_mode()
def collect_logits(model, loader, device, amp):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(x.to(device, non_blocking=True))
        logits.append(out.float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def compute_metrics(logits, labels, class_names, temperature: float = 1.0) -> dict:
    k = len(class_names)
    probs = torch.softmax(logits / temperature, dim=1).numpy().astype(np.float64)
    y = labels.numpy()
    pred = probs.argmax(1)
    cm = confusion_matrix(y, pred, labels=list(range(k)))

    try:    # AUC is undefined if a class is absent from this split
        auc = roc_auc_score(y, probs[:, 1]) if k == 2 else roc_auc_score(
            y, probs, multi_class="ovr", average="macro", labels=list(range(k)))
    except ValueError:
        auc = float("nan")

    per_class = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        per_class[name] = {   # sensitivity/specificity are what clinicians ask for
            "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
            "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        }
    return {
        "acc": float((pred == y).mean()),
        "auc": float(auc),
        "nll": float(F.cross_entropy(logits / temperature, labels)),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def fit_temperature(logits, labels) -> float:
    """Temperature scaling: learn one scalar T so softmax(logits / T) is calibrated.
    Fit on the validation set only; it does not change the predicted class."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())
