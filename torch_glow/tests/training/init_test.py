from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
import os

import pytest


@pytest.mark.skip(reason="Need something like EraseNumberTypes to re-enable")
def test_init():
    """Basic init for training."""

    trainer = torch_glow.TorchGlowTrainingWrapper()

    # test settings
    trainer.parameters(3, 10)
    trainer.config(0.1, 0.2, 0.3, 0.4, 64)

    x = torch.randn(1, 3, 224, 244)
    y = torch.randint(0, 1, (1, 1000))

    assert trainer.init(
        os.environ['TOP_DIR'] + "/tests/models/pytorchModels/resnet18.pt",
        [x], "Interpreter", True)

    assert trainer.train(x, y)
