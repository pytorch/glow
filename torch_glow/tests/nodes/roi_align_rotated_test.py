# Copyright (c) Glow Contributors. See CONTRIBUTORS file.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


def rand_rotated_rois(N, H, W, count, horizontal=False):
    W -= 1
    H -= 1

    rois = torch.rand((count, 6))

    for i in range(count):
        rois[i][0] = (N * rois[i][0]) // 1  # batch index

        rois[i][1] *= W - 1  # center_x
        rois[i][2] *= H - 1  # center_y
        rois[i][1] += 1
        rois[i][2] += 1

        rois[i][3] *= W - rois[i][1]  # width
        rois[i][4] *= H - rois[i][2]  # height

        rois[i][5] *= 0 if horizontal else 360 - 180  # angle

        assert rois[i][1] >= 1 and rois[i][1] < W
        assert rois[i][2] >= 1 and rois[i][2] < H
        assert rois[i][1] + rois[i][3] <= W
        assert rois[i][2] + rois[i][4] <= H
        assert rois[i][3] > 0
        assert rois[i][4] > 0

    return rois


class SimpleRoiAlignRotatedModel(torch.nn.Module):
    def __init__(
        self,
        order,
        spatial_scale=1.0,
        pooled_h=6,
        pooled_w=6,
        sampling_ratio=2,
        aligned=True,
    ):
        super(SimpleRoiAlignRotatedModel, self).__init__()
        self.kwargs = {
            "order": order,
            "spatial_scale": spatial_scale,
            "pooled_h": pooled_h,
            "pooled_w": pooled_w,
            "sampling_ratio": sampling_ratio,
            "aligned": aligned,
        }

    def forward(self, features, rois):
        return torch.ops._caffe2.RoIAlignRotated(features, rois, **self.kwargs)


class TestRoiAlignRotated(utils.TorchGlowTestCase):
    """TODO: Combine with TestRoiAlign"""

    @utils.deterministic_expand(
        [
            lambda: (
                "basic",
                SimpleRoiAlignRotatedModel("NCHW"),
                torch.randn(1, 3, 16, 20),
            ),
            lambda: (
                "nhwc",
                SimpleRoiAlignRotatedModel("NHWC"),
                torch.randn(1, 16, 20, 3),
            ),
            lambda: (
                "batched",
                SimpleRoiAlignRotatedModel("NCHW"),
                torch.randn(4, 3, 16, 20),
            ),
            lambda: (
                "horizontal",
                SimpleRoiAlignRotatedModel("NCHW"),
                torch.randn(4, 3, 16, 20),
                True,
            ),
            lambda: (
                "scaled",
                SimpleRoiAlignRotatedModel("NCHW", spatial_scale=0.0625),
                torch.randn(1, 3, 224, 224),
            ),
            lambda: (
                "unaligned",
                SimpleRoiAlignRotatedModel("NCHW", aligned=False),
                torch.randn(1, 3, 16, 20),
            ),
            lambda: (
                "dynamic_sampling",
                SimpleRoiAlignRotatedModel("NCHW", sampling_ratio=0),
                torch.randn(1, 3, 16, 20),
            ),
        ]
    )
    def test_roi_align_rotated(self, _, module, features, horizontal=False):
        order = module.kwargs.get("order")
        kwargs = {k: v for k, v in zip(order, features.size())}
        kwargs.pop("C")
        rois = rand_rotated_rois(count=250, horizontal=horizontal, **kwargs)
        utils.compare_tracing_methods(
            module, features, rois, fusible_ops={"_caffe2::RoIAlignRotated"}
        )
