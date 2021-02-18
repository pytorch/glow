from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F
from tests import utils


class SimpleSoftmaxModel(torch.nn.Module):
    def __init__(self, dimension):
        super(SimpleSoftmaxModel, self).__init__()
        self.dimension = dimension

    def forward(self, tensor):
        return F.softmax(tensor, self.dimension)


class TestSoftmax(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (-2, [2, 3]),
            lambda: (-1, [2, 3]),
            lambda: (0, [2, 3]),
            lambda: (1, [2, 3]),
            lambda: (-3, [2, 3, 4]),
            lambda: (-2, [2, 3, 4]),
            lambda: (-1, [2, 3, 4]),
            lambda: (0, [2, 3, 4]),
            lambda: (1, [2, 3, 4]),
            lambda: (2, [2, 3, 4]),
        ]
    )
    def test_softmax(self, dim, input_dims):
        module = SimpleSoftmaxModel(dim)
        input = torch.randn(input_dims)
        utils.compare_tracing_methods(module, input)

    def test_softmax_oob_neg_dim(self):
        """Test out of bounds negative dimension index for the PyTorch SoftMax Node on Glow."""

        with self.assertRaises(IndexError):
            utils.compare_tracing_methods(SimpleSoftmaxModel(-3), torch.randn(2, 3))
