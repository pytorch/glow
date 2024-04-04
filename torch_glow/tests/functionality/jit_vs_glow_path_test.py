# isort:skip_file

# pyre-ignore-all-errors
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F
import torch_glow
from glow.glow.torch_glow.tests.tests import utils


class TestJITVsGlowPath(utils.TorchGlowTestCase):
    def test_jit_vs_glow_path(self):
        """Basic test of the JIT vs. Glow logging feature."""

        torch_glow.enable_jit_vs_glow_compare()

        class TestModule(torch.nn.Module):
            def forward(self, input, weight):
                return F.linear((input + input), weight)

        n = 5
        in_features = 4
        out_features = 3

        input = torch.randn(n, in_features)
        weight = torch.randn(out_features, in_features)

        utils.compare_tracing_methods(
            TestModule(),
            input,
            weight,
            fusible_ops={"aten::add", "aten::linear"},
        )

    def test_jit_vs_glow_int_path(self):
        """Test JIT vs. Glow logging with int type"""

        torch_glow.enable_jit_vs_glow_compare()

        class TestModule(torch.nn.Module):
            def forward(self, a, b):
                c = a + b
                return c

        a = torch.randn(5, 6).to(dtype=torch.int32)
        b = torch.randn(5, 6).to(dtype=torch.int32)

        utils.compare_tracing_methods(TestModule(), a, b, fusible_ops={"aten::add"})

    def test_jit_vs_glow_inplace(self):
        """Test JIT vs. Glow logging with in-place op"""

        torch_glow.enable_jit_vs_glow_compare()

        class TestModule(torch.nn.Module):
            def forward(self, a, b):
                a += b
                return a

        a = torch.randn(5, 6)
        b = torch.randn(5, 6)

        utils.compare_tracing_methods(TestModule(), a, b, fusible_ops={"aten::add_"})
