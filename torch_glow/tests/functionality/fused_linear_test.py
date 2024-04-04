# isort:skip_file

# pyre-ignore-all-errors
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from glow.glow.torch_glow.tests.tests import utils
from glow.glow.torch_glow.tests.tests.utils import graph_contains_str


graph_str = """
graph(%input : Tensor, %weight : Tensor, %bias : Tensor):
  %c : int = prim::Constant[value=4]()
  %d : int = prim::Constant[value=1]()
  %1 : int = aten::dim(%input)
  %2 : bool = aten::eq(%1, %c)
  %3 : Tensor = prim::If(%2)
    block0():
      %4 : Tensor = aten::t(%weight)
      %5 : int = prim::Constant[value=1]()
      %6 : Tensor = aten::mm(%input, %4)
      %7 : Tensor = aten::add(%bias, %6, %5)
      -> (%7)
    block1():
      %8 : Tensor = aten::t(%weight)
      %9 : Tensor = aten::matmul(%input, %8)
      %10 : Tensor = aten::add_(%9, %bias, %d)
      -> (%10)
  return (%3)
"""


class TestFuseLinear(utils.TorchGlowTestCase):
    def test_fuse_linear(self):
        """Test Glow's fuseBranchedLinearPattern JIT pass"""
        graph = torch._C.parse_ir(graph_str)
        assert not graph_contains_str(graph, "glow::fused_linear")
        torch_glow.fuseBranchedLinearPattern_(graph)
        assert graph_contains_str(graph, "glow::fused_linear")
