from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleToModel(torch.nn.Module):
    def __init__(self, *conversions):
        super(SimpleToModel, self).__init__()
        self.conversions = conversions

    def forward(self, tensor):
        for conversion_type in self.conversions:
            tensor = tensor.to(conversion_type)
        return tensor


class ToWithDeviceModel(torch.nn.Module):
    def __init__(self, *conversions):
        super(ToWithDeviceModel, self).__init__()
        self.conversions = conversions

    def forward(self, tensor):
        for conversion_type in self.conversions:
            tensor = tensor.to(device="cpu", dtype=conversion_type)
        return tensor


class TestTo(unittest.TestCase):
    @parameterized.expand(
        [
            ("to_int", SimpleToModel(torch.int), torch.randn(1, 2, 3, 4)),
            ("to_float", SimpleToModel(torch.float), torch.randn(1, 2, 3, 4)),
            (
                "to_int_to_float",
                SimpleToModel(torch.int, torch.float),
                torch.randn(1, 2, 3, 4),
            ),
            (
                "to_int_with_device",
                ToWithDeviceModel(torch.int),
                torch.randn(1, 2, 3, 4),
            ),
            ("to_cpu", SimpleToModel("cpu"), torch.randn(1, 2, 3, 4)),
            (
                "to_tensor",
                SimpleToModel(torch.randn(3, 4).type(torch.int32)),
                torch.randn(1, 2, 3, 4),
            ),
        ]
    )
    def test_to(self, _, module, tensor):
        utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::to"})
