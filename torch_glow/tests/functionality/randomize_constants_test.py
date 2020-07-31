# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch_glow
from torch_glow import InputMeta, CompilationOptions, GlowCompileSpec
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(5, 10)

    def forward(self, x):
        return self.linear(x)


def run_model(m, input, randomize):
    torch_glow.disableFusionPass()
    traced_m = torch.jit.trace(m, input)

    input_meta = InputMeta()
    input_meta.set_same_as(input)
    inputs = [input_meta]
    options = CompilationOptions()
    options.backend = "Interpreter"
    options.randomize_constants = randomize
    spec = GlowCompileSpec()
    spec.set(inputs, options)

    glow_m = torch_glow.to_glow(traced_m, {"forward": spec})
    return glow_m.forward(input)


class TestRandomizeWeights(unittest.TestCase):
    def test_randomize_weights(self):
        m = Model()
        input = torch.randn(5)
        normal1 = run_model(m, input, False)
        normal2 = run_model(m, input, False)
        rand = run_model(m, input, True)

        assert torch.allclose(normal1, normal2)
        assert not torch.allclose(normal1, rand)
