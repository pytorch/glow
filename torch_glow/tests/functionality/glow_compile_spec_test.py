from __future__ import absolute_import, division, print_function, unicode_literals

import torch_glow
import torch
import unittest


class TestGlowCompileSpec(unittest.TestCase):
    def test_glow_compile_spec(self):
        """Create glow compile spec."""

        dims = [2, 2]
        gcs = torch.classes.glow.GlowCompileSpec()
        gcs.setBackend("Interpreter")
        gcs.addInputTensor("float", dims)

        sim = torch.classes.glow.SpecInputMeta()
        sim.setSpec("float", dims)
        gcs.addInput(sim)
        inputs = [sim, sim]
        gcs.addInputs(inputs)

        t = torch.tensor(dims)
        sim.setSpecFromTensor(t)
