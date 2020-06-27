from __future__ import absolute_import, division, print_function, unicode_literals

import torch_glow
import torch
import unittest


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(5, 5)

    def forward(self, x):
        return self.linear(x)


def run_model(m, input, randomize):
    if randomize:
        torch_glow.enable_randomize_constants()
    else:
        torch_glow.disable_randomize_constants()

    torch_glow.disableFusionPass()
    traced_m = torch.jit.trace(m, input)

    spec = torch.classes.glow.GlowCompileSpec()
    spec.setBackend("Interpreter")
    sim = torch.classes.glow.SpecInputMeta()
    sim.setSameAs(input)
    spec.addInputs([sim])

    glow_m = torch_glow.to_glow(traced_m._c, {"forward": spec})
    return glow_m.forward(input)


class TestRandomizeWeights(unittest.TestCase):
    def test_randomize_weights(self):
        m = Model()
        input = torch.randn(5)
        normal1 = run_model(m, input, False)
        normal2 = run_model(m, input, False)
        rand1 = run_model(m, input, True)
        rand2 = run_model(m, input, True)

        assert(torch.allclose(normal1, normal2))
        assert(not torch.allclose(normal1, rand1))
        assert(not torch.allclose(normal1, rand2))
        assert(not torch.allclose(normal2, rand1))
        assert(not torch.allclose(normal2, rand2))
        assert(not torch.allclose(rand1, rand2))
