import torch
from resnet_cls.engine import fit_temperature


def test_fit_temperature_accepts_inference_tensors():
    with torch.inference_mode():                     # same way collect_logits produces them
        logits = torch.randn(32, 3) * 5
    labels = torch.randint(0, 3, (32,))
    T = fit_temperature(logits, labels)
    assert torch.isfinite(torch.tensor(T)) and T > 0