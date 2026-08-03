import torch
from torch import nn

from amp_design.tracking import ExponentialMovingAverage


def test_ema_updates_and_restores_parameters() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)
    original = model.weight.detach().clone()
    with ema.average_parameters(model):
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 2.0))
    torch.testing.assert_close(model.weight, original)
