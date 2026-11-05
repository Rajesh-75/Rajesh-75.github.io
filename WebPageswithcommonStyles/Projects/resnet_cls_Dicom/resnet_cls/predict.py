"""Inference + Grad-CAM.  python -m resnet_cls.predict --checkpoint runs/exp1/best.pt img1.png img2.png"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import build_eval_transform
from .dicom_io import load_image, parse_window
from .model import build_model


class Predictor:
    def __init__(self, checkpoint: str, device: str | None = None):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cfg = ck["config"]
        self.classes = ck["classes"]
        self.temperature = ck.get("temperature", 1.0)       # calibration learned in train.py
        self.img_size = cfg["img_size"]
        self.window = parse_window(cfg.get("dicom_window", ""))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(cfg["arch"], len(self.classes), cfg["dropout"], pretrained=False)
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.tf = build_eval_transform(cfg["img_size"], cfg["grayscale"])   # identical to validation

    def _load(self, path) -> torch.Tensor:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(p)
        return self.tf(load_image(p, self.window).convert("RGB"))   # DICOM or ordinary image

    @torch.inference_mode()
    def predict(self, paths, batch_size: int = 16) -> list[dict]:
        results = []
        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            x = torch.stack([self._load(p) for p in chunk]).to(self.device)
            probs = torch.softmax(self.model(x).float() / self.temperature, dim=1).cpu()
            for p, pr in zip(chunk, probs):
                results.append({
                    "file": str(p),
                    "prediction": self.classes[int(pr.argmax())],
                    "probabilities": {c: round(float(v), 4) for c, v in zip(self.classes, pr)},
                })
        return results

    def explain(self, path, class_idx: int | None = None):
        """Grad-CAM heat-map (H x W in [0, 1]) showing which regions drove the prediction."""
        x = self._load(path).unsqueeze(0).to(self.device)
        store = {}

        def hook(_module, _inp, out):
            store["act"] = out
            out.register_hook(lambda g: store.__setitem__("grad", g))

        handle = self.model.layer4.register_forward_hook(hook)
        try:
            with torch.enable_grad():
                logits = self.model(x)
                idx = int(logits.argmax(1)) if class_idx is None else class_idx
                self.model.zero_grad()
                logits[0, idx].backward()
        finally:
            handle.remove()
        w = store["grad"].mean(dim=(2, 3), keepdim=True)            # channel importance
        cam = torch.relu((w * store["act"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()
    print(json.dumps(Predictor(args.checkpoint).predict(args.images), indent=2))
