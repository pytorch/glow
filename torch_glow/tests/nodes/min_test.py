import torch

from tests.utils import jitVsGlow


def test_elementwise_min():
    def test_f(a, b):
        return torch.min(a + a, b + b)

    jitVsGlow(test_f, torch.randn(7), torch.randn(7))
