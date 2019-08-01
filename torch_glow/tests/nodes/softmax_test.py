import torch
import torch.nn.functional as F

from tests.utils import jitVsGlow


def test_softmax_basic():
    """Basic test of the PyTorch SoftMax Node on Glow."""
    def softmax_basic(inputs):
        return F.softmax(inputs, dim=1)

    inputs = torch.randn(2, 3)
    jitVsGlow(softmax_basic, inputs, expected_fused_ops={"aten::softmax"})
