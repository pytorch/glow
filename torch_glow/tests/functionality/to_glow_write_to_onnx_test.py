# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os

import torch
import torch_glow
from tests import utils


class Foo(torch.nn.Module):
    def __init__(self):
        super(Foo, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(6, 16, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        y = self.conv2(x)
        return y


class Bar(torch.nn.Module):
    def __init__(self, foo):
        super(Bar, self).__init__()
        self.foo = foo

    def forward(self, x):
        y = self.foo(x)
        return y


class Baz(torch.nn.Module):
    def __init__(self, foo):
        super(Baz, self).__init__()
        self.foo = foo

    def forward(self, x):
        y = self.foo(x)
        return (x, y)


def create_model(x, ModType):
    foo = Foo()
    foo = torch.ao.quantization.QuantWrapper(foo)
    foo.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
    torch.ao.quantization.prepare(foo, inplace=True)
    foo(x)
    torch.ao.quantization.convert(foo, inplace=True)
    model = ModType(foo)
    return model


class TestToGlowWriteToOnnx(utils.TorchGlowTestCase):
    def lower_and_write_to_onnx_helper(self, ModType, onnx_prefix):
        x = torch.randn(1, 3, 8, 8)
        model = create_model(x, ModType)

        spec = torch_glow.CompilationSpec()
        spec.get_settings().set_glow_backend("Interpreter")

        compilation_group = torch_glow.CompilationGroup()
        spec.compilation_groups_append(compilation_group)

        input_spec = torch_glow.InputSpec()
        input_spec.set_same_as(x)

        compilation_group.input_sets_append([input_spec])

        scripted_mod = torch.jit.trace(model, x)
        torch_glow.enable_write_to_onnx()
        torch_glow.set_onnx_file_name_prefix(onnx_prefix)
        torch_glow.enable_write_without_randomize()
        lowered_model = torch_glow.to_glow(scripted_mod, {"forward": spec})

        # Run Glow model
        g = lowered_model(x)

        # Run reference model
        t = model(x)

        self.assertEqual(type(g), type(t))
        self.assertEqual(len(g), len(t))

        for (gi, ti) in zip(g, t):
            self.assertTrue(torch.allclose(gi, ti))

        assert os.path.exists(onnx_prefix + ".onnxtxt")
        onnx_files = glob.glob(onnx_prefix + "*.onnx*")
        for f in onnx_files:
            os.remove(f)

    def test_lower_and_write_to_onnx_tensor_output(self):
        onnx_prefix = "write_to_onnx_test1"
        self.lower_and_write_to_onnx_helper(Bar, onnx_prefix)

    def test_lower_and_write_to_onnx_tuple_output(self):
        onnx_prefix = "write_to_onnx_test2"
        self.lower_and_write_to_onnx_helper(Baz, onnx_prefix)
