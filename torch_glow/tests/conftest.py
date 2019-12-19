# contents of conftest.py
import pytest
import torch
import torch_glow


def pytest_addoption(parser):
    parser.addoption("--backend", action="store", default=None)


def pytest_sessionstart(session):
    backend = session.config.getoption("--backend")
    if backend:
        torch_glow.setGlowBackend(backend)
