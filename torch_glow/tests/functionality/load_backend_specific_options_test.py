# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import tempfile
import unittest

import torch
import torch_glow


class TestLoadBackendSpecificOptions(unittest.TestCase):
    def test_backend_specific_options(self):
        """Test loading backend specific options from YAML file."""

        def test_f(a, b):
            return a.add(b)

        x = torch.randn(4)
        y = torch.randn(4)

        # Create YAML file with backend options
        with tempfile.NamedTemporaryFile() as options_fd:
            options_fd.write(b"interpreter-memory: 4194304\n")
            options_fd.flush()

            # Run Glow
            torch_glow.loadBackendSpecificOptions(options_fd.name)
            torch_glow.enableFusionPass()
            glow_trace = torch.jit.trace(test_f, (x, y), check_trace=False)
            glow_trace(x, y)
