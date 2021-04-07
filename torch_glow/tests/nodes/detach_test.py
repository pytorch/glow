from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class DetachModel(torch.nn.Module):
    def __init__(self):
        super(DetachModel, self).__init__()

    def forward(self, a):
        b = a.detach()
        return b + b


class TestDetach(utils.TorchGlowTestCase):
    def test_detach(self):
        """Test of the PyTorch detach Node on Glow."""

        x = torch.randn(5, 6, 7)
        x.requires_grad = True

        utils.compare_tracing_methods(DetachModel(), x, fusible_ops={"aten::detach"})
