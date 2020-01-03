from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from tests.utils import graph_contains_str
import unittest

graph_str = """
graph(%x : Tensor):
  %1 : int = prim::Constant[value=4]()
  %2 : int = aten::dim(%x)
  %3 : bool = aten::eq(%1, %2)
  %e : str = prim::Constant[value="Exception"]()
  %4 : int = prim::If(%3)
    block0():
        %5 : int = prim::Constant[value=0]()
        -> (%5)
    block1():
        %6 : int = prim::Constant[value=0]()
        %7 = prim::RaiseException(%e)
        -> (%6)
  return (%x)
"""


class TestRemoveException(unittest.TestCase):
    def test_remove_exceptions(self):
        """Test Glow's removeExceptions JIT pass"""
        graph = torch._C.parse_ir(graph_str)
        assert(graph_contains_str(graph, "prim::RaiseException"))
        torch_glow.removeExceptions_(graph)
        assert(not graph_contains_str(graph, "prim::RaiseException"))
