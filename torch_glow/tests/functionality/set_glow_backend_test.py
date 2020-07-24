# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch_glow


class TestSetGlowBackend(unittest.TestCase):
    def test_set_glow_backend(self):
        """Test setting the Glow backend type"""

        backend_name_before = torch_glow.getGlowBackendName()
        backend_num_devices_before = torch_glow.getGlowBackendNumDevices()

        torch_glow.setGlowBackend("CPU")
        torch_glow.setGlowBackendNumDevices(4)

        assert torch_glow.getGlowBackendName() == "CPU"
        assert torch_glow.getGlowBackendNumDevices() == 4

        # reset everything
        torch_glow.setGlowBackend(backend_name_before)
        torch_glow.setGlowBackendNumDevices(backend_num_devices_before)
