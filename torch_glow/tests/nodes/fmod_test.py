from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from parameterized import parameterized
from tests import utils


class SimpleFmodModule(torch.nn.Module):
    def __init__(self):
        super(SimpleFmodModule, self).__init__()

    def forward(self, a, b):
        if b.size() == torch.Size([]):
            c = a.fmod(b.item())
        else:
            c = a.fmod(b)
        return c.fmod(torch.tensor(1.0, dtype=c.dtype))


class TestFmod(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (
                "int64_tensor",
                SimpleFmodModule(),
                torch.tensor([14]),
                torch.tensor([10]),
            ),
            lambda: (
                "int32_tensor",
                SimpleFmodModule(),
                torch.tensor([14], dtype=torch.int32),
                torch.tensor([10], dtype=torch.int32),
            ),
            lambda: (
                "float_tensor",
                SimpleFmodModule(),
                torch.randn(4),
                torch.tensor(0.3),
            ),
            lambda: (
                "basic_tensor",
                SimpleFmodModule(),
                torch.tensor([7.5]),
                torch.tensor([2.4]),
            ),
            lambda: (
                "int_number",
                SimpleFmodModule(),
                torch.tensor([14]),
                torch.tensor(10),
            ),
            lambda: ("basic", SimpleFmodModule(), torch.randn(4), torch.randn(4)),
            lambda: (
                "broadcast",
                SimpleFmodModule(),
                torch.randn(8, 3, 4, 2),
                torch.randn(4, 2),
            ),
            lambda: (
                "broadcast",
                SimpleFmodModule(),
                torch.randn(8, 3, 4, 2),
                torch.randn(1, 2),
            ),
            lambda: (
                "positive_broadcast",
                SimpleFmodModule(),
                torch.Tensor(8, 3, 4, 2).random_(0, 5),
                torch.Tensor(1, 2).random_(1, 5),
            ),
            lambda: (
                "positive_broadcast",
                SimpleFmodModule(),
                torch.Tensor(4, 2).random_(0, 5),
                torch.Tensor(8, 3, 4, 2).random_(1, 5),
            ),
        ]
    )
    def test_fmod(self, _, module, a, b):
        utils.compare_tracing_methods(module, a, b, fusible_ops={"aten::fmod"})
