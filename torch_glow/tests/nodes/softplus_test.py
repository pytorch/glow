from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F
from tests import utils


class SimpleSoftPlusModel(torch.nn.Module):
    def __init__(self):
        super(SimpleSoftPlusModel, self).__init__()

    def forward(self, tensor):
        tensor = tensor + tensor
        return F.softplus(tensor)


class TestSoftPlus(utils.TorchGlowTestCase):
    def test_softplus(self):
        """Basic test of the PyTorch aten::softplus Node on Glow."""

        utils.compare_tracing_methods(
            SimpleSoftPlusModel(),
            torch.randn(4, 3),
            fusible_ops={"aten::softplus"},
        )
