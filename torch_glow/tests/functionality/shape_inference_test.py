# isort:skip_file

# pyre-ignore-all-errors
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from glow.glow.torch_glow.tests.tests import utils


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

        args = (a,)

        actual = torch_glow.glow_shape_inference_find_unsupported_symbols(
            jit_f_graph, args
        )
        expected = []
        self.assertEqual(set(expected), set(actual))

    def test_shape_inference_unsupported_symbols(self):
        """Test Glow shape inference unsupported symbols."""

        def f(a):
            # linalg.multi_dot is currently not supported by shape inference engine
            return torch.matrix_power(torch.linalg.multi_dot([a * 3, a + 4]), 3)

        a = torch.randn(3, 3)

        jit_f = torch.jit.trace(f, (a))
        jit_f_graph = jit_f.graph_for(a)

        args = (a,)

        actual = torch_glow.glow_shape_inference_find_unsupported_symbols(
            jit_f_graph, args
        )
        expected = ["aten::linalg_multi_dot", "aten::linalg_matrix_power"]
        self.assertEqual(set(expected), set(actual))

        blocklist = ["aten::linalg_multi_dot"]
        actual = torch_glow.glow_shape_inference_find_unsupported_symbols(
            jit_f_graph, args, blocklist
        )
        expected = ["aten::linalg_matrix_power"]
        self.assertEqual(set(expected), set(actual))

    def test_shape_inference_unsupported_symbols_skip_fusion_group(self):
        """Test Glow shape inference unsupported symbols including skipping of
        symbols after a secondary fusion group."""

        def f(a, b):
            x1 = a * b
            x2 = x1 * b
            x3 = x2 * a
            x4 = x3 / b
            x5 = x4 / a
            x6 = x5 / b
            x7 = x6 * a
            x8 = x7 * b
            return x8 * torch.linalg.multi_dot([x8, x8])

        torch_glow.enableFusionPass_DO_NOT_USE_THIS()
        torch_glow.setFusionStartIndex(3)
        torch_glow.setFusionEndIndex(6)

        a = torch.randn(5, 5)
        b = torch.randn(5, 5)

        jit_f = torch.jit.trace(f, (a, b))

        jit_f_graph = jit_f.graph_for(a, b)

        torch_glow.clearFusionIndices()

        args = (a, b)

        # Don't skip nodes after the last fusion node.
        # in this case, one of the nodes (linalg.multi_dot) following the last fusion node
        # is not supported, and should be reported.
        actual = torch_glow.glow_shape_inference_find_unsupported_symbols(
            jit_f_graph, args, skip_last_fusion_node=False
        )
        expected = [
            "aten::linalg_multi_dot",
        ]
        self.assertEqual(set(expected), set(actual))

        # DO skip nodes after the last fusion node.
        # in this case, one of the nodes (linalg.multi_dot) following the last fusion node
        # is not supported, but is suppressed due to the skip_last_fusion_node flag.
        actual = torch_glow.glow_shape_inference_find_unsupported_symbols(
            jit_f_graph, args, skip_last_fusion_node=True
        )
        expected = []
        self.assertEqual(set(expected), set(actual))
