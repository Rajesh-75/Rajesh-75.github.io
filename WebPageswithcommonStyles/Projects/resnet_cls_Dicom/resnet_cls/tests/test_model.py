import torch
from resnet_cls.model import build_model, set_train_mode, set_trainable


def test_output_shape():
    m = build_model("resnet50", 3, 0.3, pretrained=False).eval()
    assert m(torch.randn(2, 3, 224, 224)).shape == (2, 3)


def test_frozen_layers_and_batchnorm_mode():
    m = build_model("resnet50", 3, 0.3, pretrained=False)
    set_trainable(m, ["fc"])
    set_train_mode(m)
    assert all(not p.requires_grad for p in m.layer4.parameters())
    assert all(p.requires_grad for p in m.fc.parameters())
    assert not m.layer1[0].bn1.training      # frozen BN must stay in eval mode
    assert m.fc.training                     # trainable head is in train mode
