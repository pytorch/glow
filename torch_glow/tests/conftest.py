from __future__ import absolute_import, division, print_function, unicode_literals

import torch_glow


def pytest_addoption(parser):
    parser.addoption("--backend", action="store", default=None)


def pytest_sessionstart(session):
    backend = session.config.getoption("--backend")
    if backend:
        torch_glow.setGlowBackend(backend)
