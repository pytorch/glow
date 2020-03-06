from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow
import unittest


class TestQuantizedLinear(unittest.TestCase):
    def test_quantized_linear_packed(self):
        """Basic test of the PyTorch quantized::linear Node on Glow."""

        q = torch.nn.quantized.Quantize(
            scale=1 / 25, zero_point=17, dtype=torch.quint8)
        dq = torch.nn.quantized.DeQuantize()

        linear = torch.nn.Linear(5, 5)

        linear.weight.data.fill_(1.2)
        linear.bias.data.fill_(3.0)

        model = torch.nn.Sequential(q, linear, dq)
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)

        x = torch.tensor(range(5), dtype=torch.float)
        x = torch.cat((x, x, x, x, x))
        x = torch.reshape(x, [5, 5])

        jitVsGlow(
            model,
            x,
            expected_fused_ops={
                "aten::quantize_per_tensor",
                "quantized::linear",
                "aten::dequantize",
            },
        )

    def test_quantized_linear_packed_dq_cut(self):
        """Basic test of the PyTorch quantized::linear Node on Glow, with dequantize excluded. """

        q = torch.nn.quantized.Quantize(
            scale=1 / 25, zero_point=17, dtype=torch.quint8)
        dq = torch.nn.quantized.DeQuantize()

        linear = torch.nn.Linear(5, 5)

        linear.weight.data.fill_(1.2)
        linear.bias.data.fill_(3.0)

        model = torch.nn.Sequential(q, linear, dq)
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)

        x = torch.tensor(range(5), dtype=torch.float)
        x = torch.cat((x, x, x, x, x))
        x = torch.reshape(x, [5, 5])

        jitVsGlow(
            model,
            x,
            expected_fused_ops={
                "aten::quantize_per_tensor",
                "quantized::linear",
            },
            black_list=[
                "aten::dequantize",
            ]
        )

    @unittest.skip(reason="random input could cause flaky")
    def test_quantized_linear_random_input(self):
        """Basic test of the PyTorch quantized::linear Node on Glow."""

        def test_f(inputs, weights, bias=None):
            q_int = torch.nn.quantized.Quantize(
                scale=1 / 13, zero_point=0, dtype=torch.qint8
            )
            q_uint = torch.nn.quantized.Quantize(
                scale=1 / 13, zero_point=10, dtype=torch.quint8
            )

            dq = torch.nn.quantized.DeQuantize()

            q_inputs = q_uint(inputs)
            q_weights = q_int(weights)

            return dq(torch.nn.quantized.functional.linear(q_inputs, q_weights, bias))

        for _ in range(100):
            inputs = torch.randn(7, 7)
            weights = torch.randn(7, 7)

            bias = torch.tensor([1, 1, 1, 1, 1, 1, 1], dtype=torch.float) * 0.1

            jitVsGlow(
                test_f,
                inputs,
                weights,
                bias,
                expected_fused_ops={
                    "glow::unpacked_quantized_linear",
                    "aten::quantize_per_tensor",
                    "aten::dequantize",
                },
            )

    def test_quantized_linear_packed_rowwise(self):
        """Basic test of the PyTorch quantized::linear Node with rowwise quantized
        packed weights on Glow."""

        linear = torch.nn.Linear(6, 5)
        linear.weight.data.random_(0, 100)
        linear.bias.data.random_(0, 10)

        x = torch.tensor(range(30), dtype=torch.float)
        x = torch.reshape(x, [5, 6])

        model = torch.quantization.QuantWrapper(linear)
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)

        jitVsGlow(model, x, expected_fused_ops={"aten::quantize_per_tensor",
                                                "quantized::linear",
                                                "aten::dequantize"})
