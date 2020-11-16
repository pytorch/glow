from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils
import unittest
from parameterized import parameterized


class SimpleUnbindModule(torch.nn.Module):
    def __init__(self, axis):
        super(SimpleUnbindModule, self).__init__()
        self.axis = axis

    def forward(self, tensor):
        a = tensor + tensor
        return torch.unbind(a, self.axis)


class TestUnbind(unittest.TestCase):
    @parameterized.expand(
        [
            ("1d_axis_0", SimpleUnbindModule(0), torch.arange(8)),
            ("2d_axis_0", SimpleUnbindModule(0), torch.arange(8).reshape(4, 2)),
            ("2d_axis_1", SimpleUnbindModule(1), torch.arange(14).reshape(2, 7)),
            ("3d_axis_1", SimpleUnbindModule(1), torch.arange(30).reshape(2, 5, 3)),
        ]
    )
    def test_unbind_basic(self, _, module, tensor):
        """Test of the PyTorch unbind Node on Glow."""
        utils.compare_tracing_methods(
            module, tensor, fusible_ops={"aten::unbind"}, scripted=False
        )
