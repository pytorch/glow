from __future__ import absolute_import, division, print_function, unicode_literals

import torch_glow
import torch
import unittest

from collections import OrderedDict


def create_model(x, relu):
    """ x is an example input, relu is whether or not to include a fused relu"""

    with torch.no_grad():
        x_size = len(x.size())

        conv_op = None
        if x_size == 4:
            conv_op = torch.nn.Conv2d(3, 10, 3)
        elif x_size == 5:
            conv_op = torch.nn.Conv3d(3, 10, 3)
        else:
            print(f"Only 2d and 3d conv supported, got {x_size}d inputs")
            exit(1)

        conv_op.weight.random_(-1, 1)
        conv_op.bias.data.random_(-1, 1)

        model = None
        if relu:
            model = torch.nn.Sequential(
                OrderedDict([("conv", conv_op), ("relu", torch.nn.ReLU())])
            )
            model = torch.quantization.fuse_modules(model, [["conv", "relu"]])
        else:
            model = torch.nn.Sequential(OrderedDict([("conv", conv_op)]))

        model = torch.quantization.QuantWrapper(model)
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

        torch.quantization.prepare(model, inplace=True)
        model(x)
        torch.quantization.convert(model, inplace=True)

        return model


def run_to_glow(m, x):
    """Trace the model m with input x and call to_glow"""
    traced_m = torch.jit.trace(m, (x))

    spec = torch.classes.glow.GlowCompileSpec()
    spec.setBackend("Interpreter")
    sim = torch.classes.glow.SpecInputMeta()
    sim.setSpec("float", x.size())
    inputs = [sim]
    spec.addInputs(inputs)

    lowered_module = torch_glow.to_glow(traced_m._c, {"forward": spec})
    return lowered_module


class TestConvToGlow(unittest.TestCase):
    def test_conv2d_to_glow(self):
        x = torch.randn([1, 3, 30, 30])
        m = create_model(x, False)
        run_to_glow(m, x)

    def test_conv2d_relu_to_glow(self):
        x = torch.randn([1, 3, 30, 30])
        m = create_model(x, True)
        run_to_glow(m, x)

    def test_conv3d_to_glow(self):
        x = torch.randn([1, 3, 30, 30, 30])
        m = create_model(x, False)
        run_to_glow(m, x)

    def test_conv3d_relu_to_glow(self):
        x = torch.randn([1, 3, 30, 30, 30])
        m = create_model(x, True)
        run_to_glow(m, x)
