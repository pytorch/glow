# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import torch_glow


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        help="Glow backend to use for testing",
    )
    parser.addoption(
        "--load-backend-specific-opts",
        type=str,
        default=None,
        help="YAML file containing backend specific options",
    )
    parser.addoption(
        "--load-device-configs",
        type=str,
        default=None,
        help="YAML file containing device specific options",
    )


def pytest_sessionstart(session):
    backend = session.config.getoption("--backend")
    if backend:
        torch_glow.setGlowBackend(backend)
    # Load YAML file with backend specific options
    be_opts = session.config.getoption("--load-backend-specific-opts")
    if be_opts:
        torch_glow.loadBackendSpecificOptions(be_opts)
    # Load YAML file with device specific options
    de_opts = session.config.getoption("--load-device-configs")
    if de_opts:
        torch_glow.loadDeviceConfigs(de_opts)
