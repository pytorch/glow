import torch
from tests import utils


class IndexSelectModule(torch.nn.Module):
    def __init__(self, dimension):
        super(IndexSelectModule, self).__init__()
        self.dimension = dimension

    def forward(self, tensor, index):
        return torch.index_select(tensor, self.dimension, index)


class TestIndexSelect(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("0-dim", torch.randn(3, 4), 0, torch.tensor([0, 2])),
            lambda: ("1-dim", torch.randn(3, 4), 1, torch.tensor([0, 2])),
            lambda: ("repeat index", torch.randn(3, 4), 1, torch.tensor([2, 2])),
        ]
    )
    def test_index_select(self, _, tensor, dimension, index):
        utils.compare_tracing_methods(
            IndexSelectModule(dimension),
            tensor,
            index,
            skip_to_glow=True,
            fusible_ops={"aten::index_select"},
        )
