from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleLogModule(torch.nn.Module):
    def __init__(self, *dimensions):
        super(SimpleLogModule, self).__init__()

    def forward(
        self,
        a,
    ):
        b = torch.log(a)
        return torch.log(b)


class TestLog(unittest.TestCase):
    def test_log_basic(self):

        x = 1 / torch.rand(3, 4, 5)

        utils.compare_tracing_methods(
            SimpleLogModule(),
            x,
            fusible_ops={"aten::log"},
        )
