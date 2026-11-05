"""Export to ONNX for deployment (TensorRT, ONNX Runtime, etc.)."""
import argparse
import json

import torch

from .predict import Predictor


def export_onnx(checkpoint: str, out_path: str, opset: int = 17) -> None:
    pred = Predictor(checkpoint, device="cpu")
    dummy = torch.randn(1, 3, pred.img_size, pred.img_size)
    torch.onnx.export(
        pred.model, dummy, out_path, opset_version=opset,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},   # variable batch size
    )
    # The ONNX graph outputs raw logits; the serving code needs these to turn them into
    # calibrated probabilities and to preprocess images identically to training.
    with open(out_path + ".json", "w") as f:
        json.dump({"classes": pred.classes, "temperature": pred.temperature,
                   "img_size": pred.img_size, "mean": [0.485, 0.456, 0.406],
                   "std": [0.229, 0.224, 0.225]}, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="model.onnx")
    args = ap.parse_args()
    export_onnx(args.checkpoint, args.out)
