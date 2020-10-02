from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch_glow


class TestPrintJitNodeIndices(unittest.TestCase):
    """Test printing PyTorch jit node indices."""

    def test_print_jit_indices(self):
        def test_f(a, b):
            c = a.add(b)
            return c.add(c)

        x = torch.randn(4)
        y = torch.randn(4)

        torch_glow.enableFusionPass()
        torch_glow.enable_printing_jit_node_indices()

        graph = torch.jit.trace(test_f, (x, y), check_trace=False)
        graph(x, y)
