from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F

import torch_glow
import unittest


class TestWeightFreezing(unittest.TestCase):

    # TODO: Enable this once compiled function caching works.

    @unittest.skip(reason="not ready")
    def test_weight_freezing(self):
        """Test weight freezing mechanism."""

        def conv2d(inputs, filters):
            conv = F.conv2d(inputs, filters, padding=1)
            return F.relu(conv)

        torch_glow.enableWeightFreezing()

        inputs = torch.randn(1, 4, 5, 5)
        filters = torch.randn(8, 4, 3, 3)

        conv2d_freeze = torch.jit.trace(conv2d, (inputs, filters))

        out1 = conv2d_freeze(inputs, filters)

        filters += 10

        out2 = conv2d_freeze(inputs, filters)

        assert(torch.allclose(out1, out2))
