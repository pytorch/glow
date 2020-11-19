# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleSigmoidModel(torch.nn.Module):
    def __init__(self, inplace=False):
        super(SimpleSigmoidModel, self).__init__()
        self.inplace = inplace

    def forward(self, tensor):
        if self.inplace:
            other = tensor + tensor
            return other.sigmoid_()
        else:
            other = tensor + tensor
            return other.sigmoid()


class TestSigmoid(unittest.TestCase):
    @parameterized.expand(
        [
            ("basic", SimpleSigmoidModel(), torch.randn(6)),
            ("inplace", SimpleSigmoidModel(inplace=True), torch.randn(6)),
        ]
    )
    def test_sigmoid(self, _, module, tensor):
        utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::sigmoid"})
