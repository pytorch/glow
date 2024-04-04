# isort:skip_file

# pyre-ignore-all-errors
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from glow.glow.torch_glow.tests.tests import utils
from glow.glow.torch_glow.tests.tests.utils import GLOW_FUSION_GROUP


class TestOnlyTensorOutputs(utils.TorchGlowTestCase):
    def test_only_tensor_outputs(self):
        """Test that Glow fuser only produces tensor outputs."""

        def f(a, b):
            x = (a + b).size(0)
            c = a.reshape(x, -1)
            return a + c

        torch_glow.disableFusionPass()

        a = torch.randn(5, 5)
        b = torch.randn(5, 5)

        jit_f = torch.jit.trace(f, (a, b))
        jit_f_graph = jit_f.graph_for(a, b)

        # By creating a graph with an aten::size (supported) feeding into an
        # unsupported op (prim::ListConstruct), we see that even if an op is
        # supported, if it produces a non-tensor output to the fusion group it
        # would not be fused.
        torch_glow.glowCustomFuseDebug_(
            jit_f_graph, ["prim::Constant", "aten::add", "aten::size", "aten::reshape"]
        )

        fusion_nodes = 0
        aten_sizes = 0
        for node in jit_f_graph.nodes():
            if node.kind() == GLOW_FUSION_GROUP:
                fusion_nodes += 1
            if node.kind() == "aten::size":
                aten_sizes += 1

        assert (
            fusion_nodes == 2
        ), "Expected two fusion nodes to be split up with aten::size between them"
        assert aten_sizes == 1, "Expected aten::size not to be fused"
