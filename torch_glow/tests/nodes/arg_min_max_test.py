import torch
from tests import utils


class ArgMinModule(torch.nn.Module):
    def __init__(self, dim=None, keepDims=True):
        super(ArgMinModule, self).__init__()
        self.dim = dim
        self.keepDims = keepDims

    def forward(self, tensor):
        if self.dim:
            return torch.argmin(tensor, self.dim, self.keepDims)
        else:
            return torch.argmin(tensor)


class ArgMaxModule(torch.nn.Module):
    def __init__(self, dim=None, keepDims=True):
        super(ArgMaxModule, self).__init__()
        self.dim = dim
        self.keepDims = keepDims

    def forward(self, tensor):
        if self.dim:
            return torch.argmax(tensor, self.dim, self.keepDims)
        else:
            return torch.argmax(tensor)


class TestArgMin(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("basic", ArgMinModule(), torch.randn(4)),
            lambda: ("dimensions1", ArgMinModule(1, False), torch.randn(4, 4)),
            lambda: ("dimensions2", ArgMinModule(1), torch.randn(5, 5)),
        ]
    )
    def test_argmin_node(self, _, module, tensor):
        """Test of the PyTorch ArgMin node on Glow."""
        utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::argmin"})


class TestArgMax(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("basic", ArgMaxModule(), torch.randn(4)),
            lambda: ("dimensions1", ArgMaxModule(1, False), torch.randn(4, 4)),
            lambda: ("dimensions2", ArgMaxModule(1), torch.randn(5, 5)),
        ]
    )
    def test_argmax_node(self, _, module, tensor):
        """Test of the PyTorch ArgMax node on Glow."""
        utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::argmax"})
