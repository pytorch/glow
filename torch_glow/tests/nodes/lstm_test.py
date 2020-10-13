from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn as nn
from tests.utils import jitVsGlow


class TestLSTM(unittest.TestCase):
    def test_lstm_basic(self):
        """Basic test of the PyTorch lstm Node on Glow."""

        class SimpleLSTM(nn.Module):
            def __init__(self):
                super(SimpleLSTM, self).__init__()
                self.rnn = torch.nn.LSTM(12, 10, 1)
                w2 = torch.randn(40, 10)
                w1 = torch.randn(40, 12)
                b1 = torch.randn(40)
                b2 = torch.randn(40)
                self.rnn.training = False
                self.rnn.weight_ih_l0 = torch.nn.Parameter(w1)
                self.rnn.weight_hh_l0 = torch.nn.Parameter(w2)
                self.rnn.bias_ih_l0 = torch.nn.Parameter(b1)
                self.rnn.bias_hh_l0 = torch.nn.Parameter(b2)

            def forward(self, inputs, h, c):
                return self.rnn(inputs, (h, c))

        inputs = torch.randn(10, 3, 12)
        h = torch.randn(1, 3, 10)
        c = torch.randn(1, 3, 10)
        model = SimpleLSTM()

        jitVsGlow(model, inputs, h, c, expected_fused_ops={"aten::lstm"})

    def test_lstm_no_bias(self):
        """Basic test of the PyTorch lstm Node with no bias on Glow."""

        class SimpleNoBiasLSTM(nn.Module):
            def __init__(self):
                super(SimpleNoBiasLSTM, self).__init__()
                self.rnn = torch.nn.LSTM(5, 10, 1, bias=False)
                w2 = torch.randn(40, 10)
                w1 = torch.randn(40, 5)
                self.rnn.training = False
                self.rnn.weight_ih_l0 = torch.nn.Parameter(w1)
                self.rnn.weight_hh_l0 = torch.nn.Parameter(w2)

            def forward(self, inputs, h, c):
                return self.rnn(inputs, (h, c))

        inputs = torch.randn(10, 3, 5)
        h = torch.randn(1, 3, 10)
        c = torch.randn(1, 3, 10)
        model = SimpleNoBiasLSTM()
        jitVsGlow(model, inputs, h, c, expected_fused_ops={"aten::lstm"})
