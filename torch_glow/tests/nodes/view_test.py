from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleViewModule(torch.nn.Module):
    def __init__(self, *shape):
        super(SimpleViewModule, self).__init__()
        self.shape = shape

    def forward(self, tensor):
        return (tensor + tensor).view(self.shape)


class TestView(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (SimpleViewModule(2, -1), torch.rand(2, 3, 4)),
            lambda: (SimpleViewModule(-1, 2), torch.rand(2, 3, 4)),
        ]
    )
    def test_simple(self, module, tensor):
        utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::view"})
