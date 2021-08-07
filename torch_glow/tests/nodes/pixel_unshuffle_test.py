from __future__ import absolute_import, division, print_function, unicode_literals

import random
import torch
from tests import utils


class SimplePixelUnshuffleModel(torch.nn.Module):
    def __init__(self, downscale_factor):
        super(SimplePixelUnshuffleModel, self).__init__()
        self.downscale_factor = downscale_factor
        self.ps = torch.nn.PixelUnshuffle(self.downscale_factor)

    def forward(self, tensor):
        return self.ps(tensor)


class TestPixelUnshuffle(utils.TorchGlowTestCase):
    def test_pixel_unshuffle(self):
        """Test of the PyTorch pixel_unshuffle Node on Glow."""

        for _ in range(0, 20):
            c = random.randint(1, 3)
            r = random.randint(2, 5)
            w = random.randint(1, 100)
            h = random.randint(1, 100)
            b = random.randint(1, 10)

            utils.compare_tracing_methods(
                SimplePixelUnshuffleModel(r),
                torch.randn(b, c, w * r, h * r),
                fusible_ops={"aten::pixel_unshuffle"},
            )
