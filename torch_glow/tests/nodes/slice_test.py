from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleSliceModel(torch.nn.Module):
    def __init__(self):
        super(SimpleSliceModel, self).__init__()

    def forward(self, tensor):
        other = (tensor + tensor)[1:]
        return other[0][1:]


class SimpleSliceWithStepModel(torch.nn.Module):
    def __init__(self):
        super(SimpleSliceWithStepModel, self).__init__()

    def forward(self, tensor):
        rank = len(tensor.shape)
        if rank == 2:
            other = (tensor + tensor)[::2]
            return other[:, ::2]
        elif rank == 3:
            return (tensor + tensor)[:, 3::4, :]
        elif rank == 4:
            return (tensor + tensor)[:, 3::4, :, 1::3]
        elif rank == 5:
            return (tensor + tensor)[:, 5:2, 2:2, 2::4]


class TestSlice(unittest.TestCase):
    def test_slice_basic(self):
        """Test of the PyTorch slice Node on Glow."""

        utils.compare_tracing_methods(
            SimpleSliceModel(), torch.rand((2, 3)), skip_to_glow=True
        )


class TestSliceWithStep(unittest.TestCase):
    @parameterized.expand(
        [
            ("basic", SimpleSliceWithStepModel(), torch.randn(8, 12)),
            ("3d", SimpleSliceWithStepModel(), torch.randn(6, 9, 7)),
            ("4d", SimpleSliceWithStepModel(), torch.randn(5, 8, 2, 7)),
            ("5d", SimpleSliceWithStepModel(), torch.randn(7, 9, 5, 13)),
        ]
    )
    def test_slice_with_step(self, _, module, tensor):
        utils.compare_tracing_methods(
            module, tensor, fusible_ops={"aten::slice"}, skip_to_glow=True
        )
