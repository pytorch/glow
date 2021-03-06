import torch
from tests import utils


class RepeatModule(torch.nn.Module):
    def __init__(self, repeats):
        super(RepeatModule, self).__init__()
        self.repeats = repeats

    def forward(self, tensor):
        tensor = tensor + tensor
        return tensor.repeat(self.repeats)


class TestRepeat(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("basic", RepeatModule([4]), torch.randn(3)),
            lambda: ("basic", RepeatModule([3, 5]), torch.randn(3)),
            lambda: ("basic", RepeatModule([4, 3, 5]), torch.tensor(3)),
            lambda: ("2d", RepeatModule([4, 2]), torch.randn(5, 1)),
            lambda: ("2d", RepeatModule([4, 2, 6]), torch.randn(4, 3)),
            lambda: ("3d", RepeatModule([4, 4, 2]), torch.randn(6, 3, 4)),
            lambda: ("3d", RepeatModule([3, 1, 1]), torch.randn(3, 3, 4)),
            lambda: ("3d", RepeatModule([1, 5, 1]), torch.randn(5, 3, 4)),
            lambda: ("3d", RepeatModule([4, 2, 1, 5, 2, 10]), torch.randn(6, 3, 4)),
        ]
    )
    def test_repeat(self, _, module, tensor):
        utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::repeat"})
