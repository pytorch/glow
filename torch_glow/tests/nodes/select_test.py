from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SelectModule(torch.nn.Module):
    def __init__(self, indices, axis, rank):
        super(SelectModule, self).__init__()
        self.indices = indices
        self.axis = axis
        self.rank = rank

    def forward(self, a):
        if self.rank == 2:
            if self.axis == 0:
                return (a + a)[self.indices[0], :]
            elif self.axis == 1:
                return (a + a)[:, self.indices[0]]
            else:
                return (a + a)[self.indices[0], self.indices[1]]
        elif self.rank == 3:
            if self.axis == 0:
                if len(self.indices) == 1:
                    return (a + a)[self.indices[0], :, :]
                else:
                    return (a + a)[self.indices[0], self.indices[1], :]
            elif self.axis == 1:
                if len(self.indices) == 1:
                    return (a + a)[:, :, self.indices[0]]
                else:
                    return (a + a)[:, self.indices[0], self.indices[1]]
            else:
                if len(self.indices) == 2:
                    return (a + a)[self.indices[0], :, self.indices[1]]
                else:
                    return (a + a)[self.indices[0], self.indices[1], self.indices[2]]


class TestComplexSelect(unittest.TestCase):
    @parameterized.expand(
        [
            ("2d_axis_0", SelectModule([1], 0, 2), torch.rand(2, 3)),
            ("2d_axis_1", SelectModule([2], 1, 2), torch.rand(2, 3)),
            ("2d_axis_0_1", SelectModule([0, 1], 2, 2), torch.rand(2, 3)),
            ("3d_axis_0", SelectModule([0], 0, 3), torch.rand(3, 4, 5)),
            ("3d_axis_0_1", SelectModule([2, 1], 0, 3), torch.rand(3, 4, 5)),
            ("3d_axis_1", SelectModule([0], 1, 3), torch.rand(3, 4, 5)),
            ("3d_axis_1_2", SelectModule([2, 1], 1, 3), torch.rand(3, 4, 5)),
            ("3d_axis_0_2", SelectModule([1, 3], 2, 3), torch.rand(3, 4, 5)),
            ("3d_axis_0_1_2", SelectModule([2, 0, 4], 1, 3), torch.rand(3, 4, 5)),
        ]
    )
    def test_f(self, _, module, tensor):
        """Test multidimensional tensors in the PyTorch Select Node on Glow."""
        utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::select"})
