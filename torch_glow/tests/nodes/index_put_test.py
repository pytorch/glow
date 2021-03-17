from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class IndexPutModule(torch.nn.Module):
    def __init__(self, indices, accumulate=False):
        super(IndexPutModule, self).__init__()
        self.indices = indices
        self.accumulate = accumulate

    def forward(self, tensor, val):
        tensor.index_put_(self.indices, val, accumulate=self.accumulate)
        tensor = tensor + tensor
        return tensor


class TestIndexPut(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (
                "basic",
                IndexPutModule([torch.tensor([1, 1]), torch.tensor([0, 1])]),
                torch.zeros(2, 3),
                torch.tensor([1.0, 2.0]),
            ),
            lambda: (
                "3d_0",
                IndexPutModule(
                    [torch.tensor([1, 1]), torch.tensor([0, 1]), torch.tensor([0, 1])]
                ),
                torch.zeros(2, 3, 4),
                torch.tensor([1.0, 2.0]),
            ),
            lambda: (
                "3d_1",
                IndexPutModule(
                    [
                        torch.tensor([1, 1, 0]),
                        torch.tensor([0, 1, 1]),
                        torch.tensor([0, 1, 0]),
                    ]
                ),
                torch.zeros(2, 3, 4),
                torch.tensor([1.0, 2.0, 3.0]),
            ),
            lambda: (
                "broadcast_value_0",
                IndexPutModule(
                    [
                        torch.tensor([2, 0, 1]),
                        torch.tensor([1, 2, 0]),
                        torch.tensor([2, 0, 1]),
                    ]
                ),
                torch.zeros(5, 3, 4),
                torch.tensor([1.0]),
            ),
            lambda: (
                "broadcast_value_1",
                IndexPutModule(
                    [
                        torch.tensor([1, 1, 2]),
                        torch.tensor([0, 1, 2]),
                        torch.tensor([0, 1, 3]),
                    ]
                ),
                torch.zeros(5, 3, 4),
                torch.tensor([1.0]),
            ),
            lambda: (
                "broadcast_value_2",
                IndexPutModule(
                    [
                        torch.tensor([1, 1, 0]),
                        torch.tensor([0, 1, 0]),
                    ]
                ),
                torch.zeros(5, 3, 4),
                torch.tensor([1.0, 1.0, 1.0, 1.0]),
            ),
            lambda: (
                "accumulate_basic",
                IndexPutModule([torch.tensor([1, 2]), torch.tensor([0, 1])]),
                torch.zeros(4, 3),
                torch.tensor([1.0, 2.0]),
            ),
            lambda: (
                "accumulate_broadcast",
                IndexPutModule(
                    [
                        torch.tensor([1, 1, 2]),
                        torch.tensor([0, 1, 2]),
                        torch.tensor([0, 1, 3]),
                    ],
                    True,
                ),
                torch.ones(5, 4, 6),
                torch.tensor([5.0]),
            ),
            lambda: (
                "dim_0",
                IndexPutModule(
                    [
                        torch.tensor([1]),
                    ]
                ),
                torch.zeros(5, 3, 4),
                torch.tensor([5.0]),
            ),
            lambda: (
                "dim_1",
                IndexPutModule(
                    [
                        torch.tensor([1]),
                    ]
                ),
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                torch.tensor([-3.0, -4.0]),
            ),
            lambda: (
                "dim_2",
                IndexPutModule(
                    [
                        torch.tensor([1, 0]),
                    ]
                ),
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                torch.tensor([-3.0, -4.0]),
            ),
            lambda: (
                "dim_3",
                IndexPutModule(
                    [
                        torch.tensor([1, 0, 2]),
                    ]
                ),
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                torch.tensor([[-3.0], [-4.0], [-5.0]]),
            ),
        ]
    )
    def test_index_put(self, _, module, tensor, value):
        utils.compare_tracing_methods(
            module, tensor, value, fusible_ops={"aten::index_put_"}
        )
