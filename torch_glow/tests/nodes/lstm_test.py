# Copyright (c) Glow Contributors. See CONTRIBUTORS file.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from tests import utils


class TestLSTM(utils.TorchGlowTestCase):
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

        utils.compare_tracing_methods(
            model, inputs, h, c, fusible_ops={"aten::lstm"}, skip_to_glow=True
        )

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
        utils.compare_tracing_methods(
            model, inputs, h, c, fusible_ops={"aten::lstm"}, skip_to_glow=True
        )

    def test_lstm_batch_first(self):
        """Basic test of the PyTorch lstm Node with batch first."""

        class SimpleBatchFirstLSTM(nn.Module):
            def __init__(self):
                super(SimpleBatchFirstLSTM, self).__init__()
                self.rnn = torch.nn.LSTM(12, 10, 1, batch_first=True)
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

        inputs = torch.randn(3, 10, 12)
        h = torch.randn(1, 3, 10)
        c = torch.randn(1, 3, 10)
        model = SimpleBatchFirstLSTM()
        utils.compare_tracing_methods(
            model, inputs, h, c, fusible_ops={"aten::lstm"}, skip_to_glow=True
        )

    def test_lstm_bidirectional(self):
        """Bidirectional test of the PyTorch lstm Node on Glow."""

        class BidirectionalLSTM(nn.Module):
            def __init__(self):
                super(BidirectionalLSTM, self).__init__()
                self.rnn = torch.nn.LSTM(8, 10, 1, bidirectional=True)
                self.rnn.training = False

            def forward(self, inputs, h, c):
                return self.rnn(inputs, (h, c))

        inputs = torch.randn(5, 3, 8)
        h = torch.randn(2, 3, 10)
        c = torch.randn(2, 3, 10)
        model = BidirectionalLSTM()

        utils.compare_tracing_methods(
            model, inputs, h, c, fusible_ops={"aten::lstm"}, skip_to_glow=True
        )

    def test_lstm_two_layers(self):
        """2 layer test of the PyTorch lstm Node on Glow."""

        class MultipleLayersLSTM(nn.Module):
            def __init__(self):
                super(MultipleLayersLSTM, self).__init__()
                self.rnn = torch.nn.LSTM(10, 20, 2, bidirectional=False)
                self.rnn.training = False

            def forward(self, inputs, h, c):
                return self.rnn(inputs, (h, c))

        inputs = torch.randn(5, 3, 10)
        h = torch.randn(2, 3, 20)
        c = torch.randn(2, 3, 20)
        model = MultipleLayersLSTM()

        utils.compare_tracing_methods(
            model, inputs, h, c, fusible_ops={"aten::lstm"}, skip_to_glow=True
        )
