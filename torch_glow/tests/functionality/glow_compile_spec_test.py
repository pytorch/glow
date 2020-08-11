# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

from torch_glow import InputMeta, CompilationOptions, GlowCompileSpec
import torch
import unittest


class TestGlowCompileSpec(unittest.TestCase):
    def test_glow_compile_spec(self):
        """Test glow compile spec basics."""

        dims = [2, 2]
        input_meta = InputMeta()
        input_meta.set(dims, torch.float32)
        inputs = [input_meta, input_meta]

        options = CompilationOptions()
        options.backend = "Interpreter"
        spec = GlowCompileSpec()
        spec.set(inputs, options)
