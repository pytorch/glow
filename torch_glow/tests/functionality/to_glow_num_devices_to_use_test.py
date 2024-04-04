# pyre-ignore-all-errors
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from glow.glow.torch_glow.tests.tests import utils


class SimpleModule(torch.nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()

    def forward(self, x):
        y = x + x
        y = y + 2
        return y


class TestToGlowNumDevicesToUse(utils.TorchGlowTestCase):
    def devices_to_use_test_helper(self, input, num_replications):
        model = SimpleModule()

        spec = torch_glow.CompilationSpec()
        spec.get_settings().set_glow_backend("Interpreter")
        # Init with total number of devices.
        torch_glow.setGlowBackendNumDevices(6)

        compilation_group = torch_glow.CompilationGroup()
        spec.compilation_groups_append(compilation_group)

        input_spec = torch_glow.InputSpec()
        input_spec.set_same_as(input)
        compilation_group.input_sets_append([input_spec])
        compilation_group_settings = compilation_group.get_settings()
        compilation_group_settings.set_num_devices_to_use(3)
        compilation_group_settings.set_replication_count(num_replications)

        traced_mod = torch.jit.trace(model, input)
        lowered_model = torch_glow.to_glow(traced_mod, {"forward": spec})

        g = lowered_model(input)
        t = model(input)

        self.assertEqual(type(g), type(t))
        self.assertEqual(len(g), len(t))
        for gi, ti in zip(g, t):
            self.assertTrue(torch.allclose(gi, ti))

    def devices_to_use_test(self):
        self.devices_to_use_test_helper(input=torch.randn(4), num_replications=2)
