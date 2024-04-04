# isort:skip_file

# pyre-ignore-all-errors
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from glow.glow.torch_glow.tests.tests import utils


def get_compilation_spec(inputs):
    """helper function to get the compilation spec of the submodule"""
    spec = torch_glow.CompilationSpec()
    spec.get_settings().set_glow_backend("Interpreter")

    compilation_group = torch_glow.CompilationGroup()
    spec.compilation_groups_append(compilation_group)

    compilation_group.input_sets_append(torch_glow.input_specs_from_tensors(inputs))
    return spec


class QuantizedModule(torch.nn.Module):
    def forward(self, a, b):
        return torch.ops.quantized.add(a, b, scale=1.0 / 21, zero_point=10)


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.nn.quantized.Quantize(
            scale=1.0 / 21, zero_point=0, dtype=torch.qint8
        )
        self.dequant = torch.nn.quantized.DeQuantize()
        self.add = QuantizedModule()

    def forward(self, a, b):
        return self.dequant(self.add(self.quant(a), self.quant(b)))


class TestInputSpec(utils.TorchGlowTestCase):
    def test_input_spec(self):
        """Test setting quantized and non-quantized input specs."""
        with torch.no_grad():
            a = torch.tensor([[0.1]])
            b = torch.tensor([[0.1]])

            mod = TestModule()
            traced_model = torch.jit.trace(mod, (a, b))
            ref_result = traced_model(a, b)

            # test non-quantized input
            glow_mod = torch_glow.to_glow(traced_model, get_compilation_spec((a, b)))
            glow_result = glow_mod(a, b)
            self.assertTrue(torch.allclose(ref_result, glow_result))

            # test quantized input
            add_inputs = torch_glow.get_submod_inputs(mod, "add", (a, b))
            glow_mod = torch_glow.to_glow_selective(
                traced_model, {"add": get_compilation_spec(add_inputs)}
            )
            glow_result = glow_mod(a, b)
            self.assertTrue(torch.allclose(ref_result, glow_result))
