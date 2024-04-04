# isort:skip_file

# pyre-ignore-all-errors
from __future__ import absolute_import, division, print_function, unicode_literals

import tempfile

import torch
import torch_glow
from glow.glow.torch_glow.tests.tests import utils


class TestLoadBackendSpecificOptions(utils.TorchGlowTestCase):
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
            torch_glow.enableFusionPass_DO_NOT_USE_THIS()
            glow_trace = torch.jit.trace(test_f, (x, y), check_trace=False)
            glow_trace(x, y)
