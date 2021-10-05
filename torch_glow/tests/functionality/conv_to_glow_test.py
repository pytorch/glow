# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

import torch
import torch_glow
from tests import utils


def create_model(x, relu, bias=True):
    """x is an example input, relu is whether or not to include a fused relu."""

    with torch.no_grad():
        x_size = len(x.size())

        conv_op = None
        if x_size == 4:
            if bias:
                conv_op = torch.nn.Conv2d(3, 10, 3)
            else:
                conv_op = torch.nn.Conv2d(3, 10, 3, bias=False)
        elif x_size == 5:
            conv_op = torch.nn.Conv3d(3, 10, 3)
        else:
            print(f"Only 2d and 3d conv supported, got {x_size}d inputs")
            exit(1)

        conv_op.weight.random_(-1, 1)
        if bias:
            conv_op.bias.data.random_(-1, 1)

        model = None
        if relu:
            model = torch.nn.Sequential(
                OrderedDict([("conv", conv_op), ("relu", torch.nn.ReLU())])
            )
            model = torch.ao.quantization.fuse_modules(model, [["conv", "relu"]])
        else:
            model = torch.nn.Sequential(OrderedDict([("conv", conv_op)]))

        model = torch.ao.quantization.QuantWrapper(model)
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")

        torch.ao.quantization.prepare(model, inplace=True)
        model(x)
        torch.ao.quantization.convert(model, inplace=True)

        return model


def run_to_glow(m, x):
    """Trace the model m with input x and call to_glow"""
    traced_m = torch.jit.trace(m, (x))

    spec = torch_glow.CompilationSpec()
    spec.get_settings().set_glow_backend("Interpreter")

    compilation_group = torch_glow.CompilationGroup()
    spec.compilation_groups_append(compilation_group)

    input_spec = torch_glow.InputSpec()
    input_spec.set_same_as(x)

    compilation_group.input_sets_append([input_spec])

    lowered_module = torch_glow.to_glow(traced_m, spec)

    return lowered_module


class TestConvToGlow(utils.TorchGlowTestCase):
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

    def test_conv2d_to_glow_empty_bias(self):
        x = torch.randn([1, 3, 30, 30])
        m = create_model(x, False, False)
        run_to_glow(m, x)
