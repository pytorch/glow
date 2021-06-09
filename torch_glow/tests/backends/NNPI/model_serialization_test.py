from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.jit
import torch_glow
from tests import utils


class Bar(torch.nn.Module):
    def __init__(self):
        super(Bar, self).__init__()
        self.q = torch.nn.quantized.Quantize(
            scale=0.05, zero_point=1, dtype=torch.quint8
        )
        self.dq = torch.nn.quantized.DeQuantize()

    def forward(self, x, y):
        qx = self.q(x)
        qy = self.q(y)
        qz = torch.ops.quantized.add(qx, qy, 0.08, 0)
        return self.dq(qz)


@unittest.skip(reason="This test only works on HW")
class TestToGlowNNPIModelDumping(utils.TorchGlowTestCase):
    def test_serialization(self):
        with torch.no_grad():
            x = torch.randn([1, 4, 4, 4], dtype=torch.float32)
            y = torch.randn([1, 4, 4, 4], dtype=torch.float32)
            model = Bar()
            model = torch.jit.trace(model, (x, y))

            spec = torch_glow.CompilationSpec()
            spec_settings = spec.get_settings()
            spec_settings.set_glow_backend("NNPI")
            # Enabled the serialize in this spec
            spec_settings.set_enable_serialize(True)

            compilation_group = torch_glow.CompilationGroup()
            compilation_group_settings = compilation_group.get_settings()
            compilation_group_settings.set_replication_count(1)
            compilation_group_settings.backend_specific_opts_insert(
                "NNPI_IceCores", "1"
            )

            compilation_group.input_sets_append(
                torch_glow.input_specs_from_tensors([x, y])
            )

            spec.compilation_groups_append(compilation_group)
            torch_glow.disableFusionPass()
            torch_glow.enable_convert_to_fp16()

            # Enable global serialize
            # then compile(serialize) the model and save it
            torch_glow.enable_dump_serialized_model()
            glow_mod = torch_glow.to_glow(model, spec)
            res1 = glow_mod(x, y)
            torch.jit.save(glow_mod, "/tmp/serialize_to_glow.pt")

            # Enable global deserialize and disable serialize
            # and load(deserialize) the model to loaded_glow_mod
            torch_glow.enable_deserialize()
            torch_glow.disable_dump_serialized_model()
            loaded_glow_mod = torch.jit.load("/tmp/serialize_to_glow.pt")
            res2 = loaded_glow_mod(x, y)
            assert torch.allclose(res1, res2, 1e-5, 1e-5)
