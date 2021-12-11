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

import torch
from tests import utils


def rand_rois(N, H, W, count):
    rois = torch.rand((count, 5))

    for i in range(count):
        rois[i][0] = (N * rois[i][0]) // 1  # batch index

        rois[i][1] *= W - 1  # x1
        rois[i][2] *= H - 1  # y1

        f = rois[i][3]
        if f == 0:  # enforce 0 < f < 1
            f = 1e-3
        rois[i][3] = rois[i][1] + f * (W - rois[i][1])  # x2
        f = rois[i][4]
        if f == 0:  # enforce 0 < f < 1
            f = 1e-3
        rois[i][4] = rois[i][2] + f * (H - rois[i][2])  # y2

        assert rois[i][1] > 0 and rois[i][1] < W - 1
        assert rois[i][2] > 0 and rois[i][2] < H - 1
        assert rois[i][3] > rois[i][1] and rois[i][3] < W
        assert rois[i][4] > rois[i][2] and rois[i][3] < W

    return rois


class SimpleRoiAlignModel(torch.nn.Module):
    def __init__(
        self,
        order,
        spatial_scale=1.0,
        pooled_h=6,
        pooled_w=6,
        sampling_ratio=2,
        aligned=True,
    ):
        super(SimpleRoiAlignModel, self).__init__()
        self.kwargs = {
            "order": order,
            "spatial_scale": spatial_scale,
            "pooled_h": pooled_h,
            "pooled_w": pooled_w,
            "sampling_ratio": sampling_ratio,
            "aligned": aligned,
        }

    def forward(self, features, rois):
        return torch.ops._caffe2.RoIAlign(features, rois, **self.kwargs)


class TestRoiAlign(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("basic", SimpleRoiAlignModel("NCHW"), torch.randn(1, 3, 16, 20)),
            lambda: ("nhwc", SimpleRoiAlignModel("NHWC"), torch.randn(1, 16, 20, 3)),
            lambda: ("batched", SimpleRoiAlignModel("NCHW"), torch.randn(4, 3, 16, 20)),
            lambda: (
                "scaled",
                SimpleRoiAlignModel("NCHW", spatial_scale=0.0625),
                torch.randn(1, 3, 224, 224),
            ),
            lambda: (
                "unaligned",
                SimpleRoiAlignModel("NCHW", aligned=False),
                torch.randn(1, 3, 16, 20),
            ),
            lambda: (
                "dynamic_sampling",
                SimpleRoiAlignModel("NCHW", sampling_ratio=0),
                torch.randn(1, 3, 16, 20),
            ),
        ]
    )
    def test_roi_align(self, _, module, features):
        order = module.kwargs.get("order")
        kwargs = {k: v for k, v in zip(order, features.size())}
        kwargs.pop("C")
        rois = rand_rois(count=250, **kwargs)
        utils.compare_tracing_methods(
            module, features, rois, fusible_ops={"_caffe2::RoIAlign"}
        )

    def test_roi_align_fp16(self):
        """Test of the _caffe2::RoiAlign Node on Glow."""

        N, C, H, W = 1, 3, 16, 20

        features = torch.randn(N, C, H, W)
        rois = rand_rois(N, H, W, 250)

        # atol/rtol must be high because maximum delta can be high due to shifts
        # in sampling points due to fp16 rounding of coordinates.
        utils.compare_tracing_methods(
            SimpleRoiAlignModel("NCHW"),
            features,
            rois,
            fusible_ops={"_caffe2::RoIAlign"},
            fp16=True,
            atol=1e-1,
            rtol=1e-1,
        )
