from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow


def test_init():
    """Basic init for training."""

    trainer = torch_glow.TorchGlowTrainingWrapper()

    # test settings
    trainer.settings(True, True)
    trainer.parameters(3, 10)
    trainer.config(0.1, 0.2, 0.3, 0.4, 64)

    x = torch.randn(3, 224, 244)

    trainer.train(x, x)
    #  TODO assert trainer.init(
    #  "../../../tests/models/pytorchModels/resnet18.pt",
    #  [x], "Interpreter", True)
