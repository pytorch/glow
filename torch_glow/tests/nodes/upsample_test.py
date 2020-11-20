# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleUpsampleModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(SimpleUpsampleModel, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, tensor):
        return torch.nn.Upsample(*self.args, **self.kwargs)(tensor)


class TestUpsample(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "3d_2x_size_nearest",
                SimpleUpsampleModel(size=(8, 10, 12)),
                torch.rand(2, 3, 4, 5, 6),
            ),
            (
                "3d_2x_scale_factor_nearest",
                SimpleUpsampleModel(scale_factor=(2, 2, 2)),
                torch.rand(2, 3, 4, 5, 6),
            ),
            (
                "3d_2x_single_scale_factor_nearest",
                SimpleUpsampleModel(scale_factor=2),
                torch.rand(2, 3, 4, 5, 6),
            ),
            (
                "3d_not_2x_single_scale_factor_nearest",
                SimpleUpsampleModel(scale_factor=5),
                torch.rand(2, 3, 4, 5, 6),
            ),
            (
                "3d_not_2x_scale_factor_nearest",
                SimpleUpsampleModel(scale_factor=(1, 2, 3)),
                torch.rand(2, 3, 4, 5, 6),
            ),
            (
                "3d_not_2x_size_nearest",
                SimpleUpsampleModel(size=(10, 12, 13)),
                torch.rand(2, 3, 4, 5, 6),
            ),
        ]
    )
    def test_upsample_nearest3d(self, name, module, tensor):
        utils.compare_tracing_methods(
            module,
            tensor,
            fusible_ops=["aten::upsample_nearest3d"],
        )

    @parameterized.expand(
        [
            (
                "2d_2x_scale_factor_nearest",
                SimpleUpsampleModel(scale_factor=(2, 2)),
                torch.rand(1, 1, 3, 3),
            ),
            (
                "2d_not_2x_scale_factor_nearest",
                SimpleUpsampleModel(scale_factor=(3, 4)),
                torch.rand(1, 1, 3, 3),
            ),
            (
                "2d_2x_single_scale_factor_nearest",
                SimpleUpsampleModel(scale_factor=2),
                torch.rand(1, 1, 3, 3),
            ),
            (
                "2d_not_2x_single_scale_factor_nearest",
                SimpleUpsampleModel(scale_factor=3),
                torch.rand(1, 1, 3, 3),
            ),
            (
                "2d_2x_size_nearest",
                SimpleUpsampleModel(size=(6, 6)),
                torch.rand(1, 1, 3, 3),
            ),
            (
                "2d_not_2x_size_nearest",
                SimpleUpsampleModel(size=(4, 8)),
                torch.rand(1, 1, 3, 3),
            ),
        ]
    )
    def test_upsample_nearest2d(self, name, module, tensor):
        utils.compare_tracing_methods(
            module,
            tensor,
            fusible_ops=["aten::upsample_nearest2d"],
        )
