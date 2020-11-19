from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleViewModule(torch.nn.Module):
    def __init__(self, *shape):
        super(SimpleViewModule, self).__init__()
        self.shape = shape

    def forward(self, tensor):
        return (tensor + tensor).view(self.shape)


class TestView(unittest.TestCase):
    @parameterized.expand(
        [
            (SimpleViewModule(2, -1), torch.rand(2, 3, 4)),
            (SimpleViewModule(-1, 2), torch.rand(2, 3, 4)),
        ]
    )
    def test_simple(self, module, tensor):
        utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::view"})
