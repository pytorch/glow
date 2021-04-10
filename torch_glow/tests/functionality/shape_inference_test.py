# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from tests import utils


class TestGlowShapeInference(utils.TorchGlowTestCase):
    def test_shape_inference_basics(self):
        """Test Glow shape inference basic usage."""

        def f(a):
            return a * a

        a = torch.randn(1)

        jit_f = torch.jit.trace(f, (a))
        jit_f_graph = jit_f.graph_for(a)

        args = (a,)

        actual = torch_glow.glow_shape_inference(
            jit_f_graph,
            args,
        )

        assert actual

    def test_shape_inference_input_mismatch(self):
        """Test Glow shape inference basic error handling."""

        def f(a):
            return a * a

        a = torch.randn(1)

        jit_f = torch.jit.trace(f, (a))
        jit_f_graph = jit_f.graph_for(a)

        # Input/args is empty, but the funciton expects one input.
        # Shape Inference should raise an exception in this case.
        args = ()

        self.assertRaises(
            Exception,
            lambda: torch_glow.glow_shape_inference(
                jit_f_graph,
                args,
            ),
        )

    def test_shape_inference_supported_symbols(self):
        """Test Glow shape inference unsupported symbols."""

        def f(a):
            return a * a

        a = torch.randn(1)

        jit_f = torch.jit.trace(f, (a))
        jit_f_graph = jit_f.graph_for(a)

        actual = torch_glow.glow_shape_inference_find_unsupported_symbols(jit_f_graph)
        expected = []
        self.assertEqual(set(expected), set(actual))

    def test_shape_inference_unsupported_symbols(self):
        """Test Glow shape inference unsupported symbols."""

        def f(a):
            # chain_matmul is currently not supported by shape inference engine
            return torch.matrix_power(torch.chain_matmul(a * 3, a + 4), 3)

        a = torch.randn(3, 3)

        jit_f = torch.jit.trace(f, (a))
        jit_f_graph = jit_f.graph_for(a)

        actual = torch_glow.glow_shape_inference_find_unsupported_symbols(jit_f_graph)
        expected = ["aten::chain_matmul", "aten::linalg_matrix_power"]
        self.assertEqual(set(expected), set(actual))

        blocklist = ["aten::chain_matmul"]
        actual = torch_glow.glow_shape_inference_find_unsupported_symbols(
            jit_f_graph, blocklist
        )
        expected = ["aten::linalg_matrix_power"]
        self.assertEqual(set(expected), set(actual))
