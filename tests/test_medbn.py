import torch
import torch.nn as nn

from adapt_algorithm import MedBatchNorm2d, configure_model, replace_batch_norm_with_medbn


class TinyBNNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def test_replace_batch_norm_with_medbn_preserves_affine_params():
    model = TinyBNNet()
    bn = model.block[1]
    with torch.no_grad():
        bn.weight.fill_(1.5)
        bn.bias.fill_(-0.25)
        bn.running_mean.fill_(0.3)
        bn.running_var.fill_(0.7)

    replace_batch_norm_with_medbn(model)
    medbn = model.block[1]

    assert isinstance(medbn, MedBatchNorm2d)
    assert torch.allclose(medbn.weight, torch.full_like(medbn.weight, 1.5))
    assert torch.allclose(medbn.bias, torch.full_like(medbn.bias, -0.25))
    assert torch.allclose(medbn.running_mean, torch.full_like(medbn.running_mean, 0.3))
    assert torch.allclose(medbn.running_var, torch.full_like(medbn.running_var, 0.7))


def test_medbn_works_with_configure_model():
    model = TinyBNNet()
    replace_batch_norm_with_medbn(model)
    configure_model(model)

    medbn = model.block[1]
    x = torch.randn(2, 3, 8, 8)
    y = model.predict(x)

    assert isinstance(medbn, MedBatchNorm2d)
    assert medbn.track_running_stats is False
    assert medbn.running_mean is None
    assert medbn.running_var is None
    assert y.shape == (2, 4, 8, 8)
    assert torch.isfinite(y).all()
