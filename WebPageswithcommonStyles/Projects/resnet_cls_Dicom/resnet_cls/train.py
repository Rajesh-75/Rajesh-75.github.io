"""Entry point:  python -m resnet_cls.train --data-dir data --output-dir runs/exp1"""
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from .config import parse_args
from .data import build_datasets, class_weights, make_loader
from .engine import collect_logits, compute_metrics, fit_temperature, train_one_epoch
from .model import build_model, set_trainable
from .utils import atomic_save, seed_everything, setup_logging

PHASES = ("head", "full")


def build_optimizer(model, phase, cfg):
    if phase == "head":                                   # backbone frozen, train new head only
        return torch.optim.AdamW(model.fc.parameters(), lr=cfg.lr_head,
                                 weight_decay=cfg.weight_decay), None
    opt = torch.optim.AdamW([                              # discriminative learning rates
        {"params": model.layer3.parameters(), "lr": cfg.lr_layer3},
        {"params": model.layer4.parameters(), "lr": cfg.lr_layer4},
        {"params": model.fc.parameters(), "lr": cfg.lr_fc},
    ], weight_decay=cfg.weight_decay)
    return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.full_epochs)


def main(argv=None):
    cfg = parse_args(argv)
    out = Path(cfg.output_dir)
    log = setup_logging(out)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    seed_everything(cfg.seed, cfg.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg.amp and device.type == "cuda"
    log.info("device=%s amp=%s", device, amp)

    train_ds, val_ds, test_ds = build_datasets(cfg)
    classes = train_ds.classes
    log.info("classes=%s  train=%d val=%d test=%s", classes, len(train_ds), len(val_ds),
             len(test_ds) if test_ds else "none")
    train_dl, val_dl = make_loader(train_ds, cfg, True), make_loader(val_ds, cfg, False)

    model = build_model(cfg.arch, len(classes), cfg.dropout, cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_ds).to(device))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_auc, bad_epochs = -1.0, 0
    start_phase, start_epoch, ck = 0, 0, None
    if cfg.resume:                                         # continue an interrupted run
        ck = torch.load(cfg.resume, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        scaler.load_state_dict(ck["scaler"])
        start_phase, start_epoch = ck["phase_idx"], ck["epoch"]
        best_auc, bad_epochs = ck["best_auc"], ck["bad_epochs"]
        log.info("resumed from %s (phase=%s, epoch=%d)", cfg.resume, PHASES[start_phase], start_epoch)

    stop = False
    for pi, phase in enumerate(PHASES):
        if pi < start_phase:
            continue
        n_epochs = cfg.head_epochs if phase == "head" else cfg.full_epochs
        set_trainable(model, ["fc"] if phase == "head" else ["layer3", "layer4", "fc"])
        opt, sched = build_optimizer(model, phase, cfg)
        first = 0
        if ck is not None and pi == start_phase:
            opt.load_state_dict(ck["opt"])
            if sched is not None and ck["sched"] is not None:
                sched.load_state_dict(ck["sched"])
            first = start_epoch

        for epoch in range(first, n_epochs):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_dl, opt, criterion, scaler, device, amp, cfg.grad_clip)
            if sched is not None:
                sched.step()
            logits, labels = collect_logits(model, val_dl, device, amp)
            m = compute_metrics(logits, labels, classes)

            if m["auc"] > best_auc:                        # NaN compares False, so never "improves"
                best_auc, bad_epochs = m["auc"], 0
                atomic_save({"model": model.state_dict(), "classes": classes,
                             "config": asdict(cfg), "val_auc": best_auc}, out / "best.pt")
            elif phase == "full":
                bad_epochs += 1

            atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                         "sched": sched.state_dict() if sched else None,
                         "scaler": scaler.state_dict(), "phase_idx": pi, "epoch": epoch + 1,
                         "best_auc": best_auc, "bad_epochs": bad_epochs}, out / "last.pt")
            with open(out / "history.jsonl", "a") as f:
                f.write(json.dumps({"phase": phase, "epoch": epoch + 1, "train_loss": train_loss,
                                    "val": m, "secs": time.time() - t0}) + "\n")
            log.info("[%s] epoch %d/%d loss=%.4f val_acc=%.3f val_auc=%.3f (%.0fs)",
                     phase, epoch + 1, n_epochs, train_loss, m["acc"], m["auc"], time.time() - t0)

            if phase == "full" and bad_epochs >= cfg.patience:
                log.info("Early stopping: no val-AUC gain for %d epochs", cfg.patience)
                stop = True
                break
        if stop:
            break

    if not (out / "best.pt").exists():
        raise RuntimeError("No valid checkpoint saved (validation AUC was never defined).")

    # ---- Final evaluation with the best weights ----
    best = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model"])
    logits, labels = collect_logits(model, val_dl, device, amp)
    temperature = fit_temperature(logits, labels)          # calibrate on val, never on test
    best["temperature"] = temperature
    atomic_save(best, out / "best.pt")

    final = {"temperature": temperature, "val": compute_metrics(logits, labels, classes, temperature)}
    if test_ds is not None:                                # touched exactly once, at the end
        t_logits, t_labels = collect_logits(model, make_loader(test_ds, cfg, False), device, amp)
        final["test"] = compute_metrics(t_logits, t_labels, classes, temperature)
    (out / "metrics.json").write_text(json.dumps(final, indent=2))
    log.info("Done. val_auc=%.3f temperature=%.3f -> %s", final["val"]["auc"], temperature, out / "metrics.json")


if __name__ == "__main__":
    main()
