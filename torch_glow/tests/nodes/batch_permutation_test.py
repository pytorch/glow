from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleBatchPermutationModule(torch.nn.Module):
    def forward(self, input, indices):
        return torch.ops._caffe2.BatchPermutation(input + input, indices)


class TestBatchPermutation(unittest.TestCase):
    def test_batch_permutation_basic(self):
        """Basic test of the _caffe2::BatchPermutation Node on Glow."""

        x = torch.randn(4, 2, 3)
        indices = torch.tensor([1, 3, 0, 2], dtype=torch.int32)

        utils.compare_tracing_methods(
            SimpleBatchPermutationModule(),
            x,
            indices,
            fusible_ops={"_caffe2::BatchPermutation"},
        )
