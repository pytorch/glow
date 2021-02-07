# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleShapeAsTensorModel(torch.nn.Module):
    def __init__(self):
        super(SimpleShapeAsTensorModel, self).__init__()

    def forward(self, tensor):
        result = torch._shape_as_tensor(tensor)
        return result + result


class TestShapeAsTensor(unittest.TestCase):
    @parameterized.expand(
        [
            ("single dimension", SimpleShapeAsTensorModel(), torch.randn(6)),
            ("multiple dimensions", SimpleShapeAsTensorModel(), torch.randn(3, 2, 4)),
        ]
    )
    def test_shape_as_tensor(self, _, module, tensor):
        """Test of the PyTorch ShapeAsTensor Node on Glow."""
        utils.compare_tracing_methods(
            module,
            tensor,
            fusible_ops={"aten::_shape_as_tensor"},
        )
