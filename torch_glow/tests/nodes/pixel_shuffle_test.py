from __future__ import absolute_import, division, print_function, unicode_literals

import random
import torch
from tests import utils


class SimplePixelShuffleModel(torch.nn.Module):
    def __init__(self, upscale_factor):
        super(SimplePixelShuffleModel, self).__init__()
        self.upscale_factor = upscale_factor
        self.ps = torch.nn.PixelShuffle(self.upscale_factor)

    def forward(self, tensor):
        return self.ps(tensor)


class TestPixelShuffle(utils.TorchGlowTestCase):
    def test_pixel_shuffle(self):
        """Test of the PyTorch pixel_shuffle Node on Glow."""

        for _ in range(0, 20):
            c = random.randint(1, 3)
            r = random.randint(2, 5)
            w = random.randint(1, 100)
            h = random.randint(1, 100)
            b = random.randint(1, 10)

            utils.compare_tracing_methods(
                SimplePixelShuffleModel(r),
                torch.randn(b, c * r ** 2, w, h),
                fusible_ops={"aten::pixel_shuffle"},
            )
