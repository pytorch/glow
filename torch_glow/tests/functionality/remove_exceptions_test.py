# isort:skip_file

# pyre-ignore-all-errors
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from glow.glow.torch_glow.tests.tests import utils
from glow.glow.torch_glow.tests.tests.utils import graph_contains_str


def foo(x):
    y = x.dim()
    if y == 1:
        return x
    else:
        if x == 2:
            return x * 2
        else:
            raise RuntimeError("hi")


class TestRemoveException(utils.TorchGlowTestCase):
    def test_remove_exceptions(self):
        """Test Glow's removeExceptions JIT pass"""

        foo_jit = torch.jit.script(foo)
        graph = foo_jit.graph
        assert graph_contains_str(graph, "prim::RaiseException")
        torch_glow.removeExceptions_(graph)
        assert not graph_contains_str(graph, "prim::RaiseException")
