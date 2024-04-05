# isort:skip_file

# pyre-ignore-all-errors
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch_glow
from glow.glow.torch_glow.tests.tests import utils


class TestFuseParallelBranches(utils.TorchGlowTestCase):
    def test_fuse_parallel_branches_with_fusible_root(self):
        r"""Test GlowFuser fusing parallel branches with a common fusible root

                           a = add(x, y)
                            /        \
                b1 = add(a, x)      b2 = add(a, y)
                        \                 /
                res = TupleConstruct(b1, b2)

        This should be fused as

                        glow::FusionGroup_0
                                    |
                            TupleConstruct
        """

        def test_fuser(x, y):
            a = x + y
            branch1 = a + x
            branch2 = a + y
            res = (branch1, branch2)
            return res

        inputs = (torch.randn(2, 4), torch.randn(2, 4))
        traced = torch.jit.trace(test_fuser, inputs)
        torch_glow.glowCustomFuseDebug_(traced.graph)

        count = 0
        for node in traced.graph.nodes():
            if node.kind() == "glow::FusionGroup":
                count += 1
        assert count == 1, f"Expect 1 glow::FusionGroup, found {count}."

    # TODO: support fusing parallel branches without a common fusible root correctly
    @unittest.skip("Not supported yet")
    def test_fuse_parallel_branches_without_fusible_root(self):
        r"""Test GlowFuser fusing parallel branches without a common fusible root

                x = add(x, x)       y = add(y, y)
                        |                  |
                b1 = add(x, x)      b2 = add(y, y)
                        \                 /
                res = TupleConstruct(b1, b2)

        This should be fused as

                        glow::FusionGroup_0
                                    |
                            TupleConstruct

        """

        def test_fuser(x, y):
            x = x + x
            y = y + y
            branch1 = x + x
            branch2 = y + y
            res = (branch1, branch2)
            return res

        inputs = (torch.randn(2, 4), torch.randn(2, 4))
        traced = torch.jit.trace(test_fuser, inputs)
        torch_glow.glowCustomFuseDebug_(traced.graph)

        count = 0
        for node in traced.graph.nodes():
            if node.kind() == "glow::FusionGroup":
                count += 1
        assert count == 1, f"Expect 1 glow::FusionGroup, found {count}."
