from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestBatchPermutation(unittest.TestCase):
    def test_batch_permutation_basic(self):
        """Basic test of the _caffe2::BatchPermutation Node on Glow."""

        def test_f(a, indices):
            return torch.ops._caffe2.BatchPermutation(a + a, indices)

        x = torch.randn(4, 2, 3)
        indices = torch.tensor([1, 3, 0, 2], dtype=torch.int32)

        jitVsGlow(test_f, x, indices, expected_fused_ops={"_caffe2::BatchPermutation"})
