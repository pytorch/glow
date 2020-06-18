from __future__ import absolute_import, division, print_function, unicode_literals

import torch_glow
import torch
import unittest


class TestGlowCompileSpec(unittest.TestCase):
    def test_glow_compile_spec(self):
        """Test glow compile spec basics."""

        dims = [2, 2]
        gcs = torch.classes.glow.GlowCompileSpec()
        gcs.setBackend("Interpreter")

        # Test SpecInputMeta setters
        sim = torch.classes.glow.SpecInputMeta()
        sim.set(dims, torch.float32)
        t = torch.tensor(dims)
        sim.setSameAs(t)

        # Test adding input methods
        gcs.addInputTensor(dims, torch.float32)
        gcs.addInput(sim)
        inputs = [sim, sim]
        gcs.addInputs(inputs)
