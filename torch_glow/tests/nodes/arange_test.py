from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleArangeModule(torch.nn.Module):
    def __init__(self, end, start=0, step=1):
        super(SimpleArangeModule, self).__init__()
        self.start = start
        self.end = end
        self.step = step

    def forward(self, dummy):
        start = self.start(dummy) if callable(self.start) else self.start
        end = self.end(dummy) if callable(self.end) else self.end
        step = self.step(dummy) if callable(self.step) else self.step
        return torch.arange(start=start, end=end, step=step)


class TestArange(unittest.TestCase):
    """
    Tests for torch.arange glow fusion.

    Note that torch.arange is effectively a constant, so torch jit will try to
    compile it down to said constant. The tests in this class utilize a test
    function which takes a tensor as input, so that we can prevent that from
    happening. Otherwise, there would be nothing to fuse.
    """

    @parameterized.expand(
        [
            ("simple", SimpleArangeModule(end=lambda x: x.size(0)), torch.randn(10)),
            (
                "all_args",
                SimpleArangeModule(start=lambda x: x.size(0), end=30, step=1),
                torch.randn(10),
            ),
            (
                "floats",
                SimpleArangeModule(start=lambda x: x.size(0), end=30.5, step=0.8),
                torch.randn(10),
            ),
            (
                "negative_step",
                SimpleArangeModule(
                    start=lambda x: x.size(0), end=lambda x: x.size(1), step=-1.2
                ),
                torch.randn(10, 2),
            ),
        ]
    )
    def test_arange(self, _, module, dummy):
        """Testing arange with minimum parameters"""
        utils.compare_tracing_methods(module, dummy, fusible_ops={"aten::arange"})
