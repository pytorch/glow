from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleSizeModel(torch.nn.Module):
    def __init__(self, dimension):
        super(SimpleSizeModel, self).__init__()
        self.dimension = dimension

    def forward(self, tensor):
        return tensor.size(self.dimension)


class TestSize(utils.TorchGlowTestCase):
    # Need to be able to export lists from Glow fused nodes
    # Commented out both test cases for not triggering internal CI
    # @unittest.skip(reason="not ready")
    # def test_size_basic(self):
    #    """Test of the PyTorch aten::size Node on Glow."""

    #    def test_f(a):
    #        b = a + a.size(0)
    #        return b

    #    x = torch.zeros([4], dtype=torch.int32)

    #    utils.compare_tracing_methods(test_f, x, fusible_ops={"aten::size"})

    @utils.deterministic_expand(
        [
            lambda: (
                "basic",
                SimpleSizeModel(-1),
                torch.randn(2, 3, 4, dtype=torch.float32),
            )
        ]
    )
    def test_size(self, _, module, tensor):
        utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::size"})

    @utils.deterministic_expand(
        [
            lambda: (
                "oob",
                SimpleSizeModel(-4),
                torch.randn(2, 3, 4, dtype=torch.float32),
            )
        ]
    )
    def test_size_failure(self, _, module, tensor):
        with self.assertRaises(IndexError):
            utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::size"})
