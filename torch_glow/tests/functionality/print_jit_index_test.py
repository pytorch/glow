# pyre-ignore-all-errors
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from glow.glow.torch_glow.tests.tests import utils


class TestPrintJitNodeIndices(utils.TorchGlowTestCase):
    """Test printing PyTorch jit node indices."""

    def test_print_jit_indices(self):
        def test_f(a, b):
            c = a.add(b)
            return c.add(c)

        x = torch.randn(4)
        y = torch.randn(4)

        torch_glow.enableFusionPass_DO_NOT_USE_THIS()
        torch_glow.enable_printing_jit_node_indices()

        graph = torch.jit.trace(test_f, (x, y), check_trace=False)
        graph(x, y)
