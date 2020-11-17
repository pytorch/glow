from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleIndexModule(torch.nn.Module):
    def __init__(self, indices, type, rank):
        super(SimpleIndexModule, self).__init__()
        self.indices = indices
        self.type = type
        self.rank = rank

    def forward(self, a):
        if self.rank == 3:
            if self.type == 0:
                if len(self.indices) == 1:
                    return (a + a)[self.indices[0], :, :]
                else:
                    return (a + a)[self.indices[0], self.indices[1], :]
            elif self.type == 1:
                if len(self.indices) == 1:
                    return (a + a)[:, :, self.indices[0]]
                else:
                    return (a + a)[:, self.indices[0], self.indices[1]]
            else:
                if len(self.indices) == 2:
                    return (a + a)[self.indices[0], :, self.indices[1]]
                else:
                    return (a + a)[self.indices[0], self.indices[1], self.indices[2]]
        elif self.rank == 4:
            if len(self.indices) == 1:
                return (a + a)[:, self.indices[0], :, :]
            elif len(self.indices) == 2:
                return (a + a)[:, self.indices[0], :, self.indices[1]]
            elif len(self.indices) == 3:
                return (a + a)[self.indices[0], self.indices[1], :, self.indices[2]]
            else:
                return (a + a)[
                    self.indices[0], self.indices[1], self.indices[2], self.indices[3]
                ]


class TestIndex(unittest.TestCase):
    @parameterized.expand(
        [
            ("3d_0", SimpleIndexModule([[1, 0]], 0, 3), torch.rand(2, 3, 4)),
            ("3d_1", SimpleIndexModule([[1, 0], [2, 1]], 1, 3), torch.rand(2, 3, 4)),
            ("3d_2", SimpleIndexModule([[1, 0], [2, 1]], 2, 3), torch.rand(2, 3, 4)),
            ("3d_3", SimpleIndexModule([[1, 0]], 0, 3), torch.rand(4, 5, 3)),
            ("3d_4", SimpleIndexModule([[1, 0], [1, 2]], 1, 3), torch.rand(4, 5, 3)),
            ("3d_5", SimpleIndexModule([[1, 0], [1, 2]], 2, 3), torch.rand(4, 5, 3)),
            (
                "3d_6",
                SimpleIndexModule([[1, 0], [2, 1], [1, 1]], 0, 3),
                torch.rand(2, 3, 4),
            ),
            ("3d_7", SimpleIndexModule([[1, 0], [2, 1]], 1, 3), torch.rand(6, 5, 8)),
            ("3d_8", SimpleIndexModule([[1, 0], [2, 1]], 2, 3), torch.rand(6, 5, 8)),
            ("4d_1", SimpleIndexModule([[1, 0]], 0, 4), torch.rand(5, 3, 4, 6)),
            ("4d_2", SimpleIndexModule([[1, 0], [2, 1]], 0, 4), torch.rand(5, 3, 4, 6)),
            (
                "4d_3",
                SimpleIndexModule([[1, 0], [2, 1], [1, 1]], 0, 4),
                torch.rand(5, 3, 4, 6),
            ),
            (
                "4d_4",
                SimpleIndexModule([[1, 0], [2, 1], [0, 1], [2, 3]], 0, 4),
                torch.rand(5, 3, 4, 6),
            ),
            (
                "2d_indices",
                SimpleIndexModule(
                    [torch.tensor([[1, 0], [1, 2]]), torch.tensor([[1, 0], [1, 2]])],
                    1,
                    3,
                ),
                torch.rand(6, 5, 8),
            ),
            (
                "2d_indices",
                SimpleIndexModule(
                    [torch.tensor([[4, 0], [3, 2]]), torch.tensor([[2, 1], [1, 3]])],
                    2,
                    4,
                ),
                torch.rand(6, 5, 8, 7),
            ),
            (
                "2d_indices",
                SimpleIndexModule(
                    [torch.tensor([[1, 0], [1, 2]]), torch.tensor([[1, 0], [1, 2]])],
                    3,
                    4,
                ),
                torch.rand(6, 5, 3, 7),
            ),
        ]
    )
    def test_f(self, _, module, tensor):
        utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::index"})
