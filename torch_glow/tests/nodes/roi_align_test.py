from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


def rand_rois(N, H, W, num_rois):
    rois = torch.rand((num_rois, 5))

    for i in range(num_rois):
        rois[i][0] = (N * rois[i][0]) // 1  # batch index

        rois[i][1] *= W - 1  # x1
        rois[i][2] *= H - 1  # y1

        rois[i][3] *= W - rois[i][1]  # x2
        rois[i][3] += rois[i][1]
        rois[i][4] *= H - rois[i][2]  # y2
        rois[i][4] += rois[i][2]

        assert rois[i][1] > 0 and rois[i][1] < W - 1
        assert rois[i][2] > 0 and rois[i][2] < H - 1
        assert rois[i][3] > rois[i][1] and rois[i][3] < W
        assert rois[i][4] > rois[i][2] and rois[i][3] < W

    return rois


class TestRoiAlign(unittest.TestCase):
    def test_roi_align_basic(self):
        """Test of the _caffe2::RoiAlign Node on Glow."""

        def test_f(features, rois):
            return torch.ops._caffe2.RoIAlign(
                features,
                rois,
                order="NCHW",
                spatial_scale=1.0,
                pooled_h=6,
                pooled_w=6,
                sampling_ratio=2,
                aligned=True,
            )

        N, C, H, W = 1, 3, 16, 20

        features = torch.randn(N, C, H, W)
        rois = rand_rois(N, H, W, 250)

        jitVsGlow(test_f, features, rois, expected_fused_ops={"_caffe2::RoIAlign"})

    def test_roi_align_nhwc(self):
        """Test of the _caffe2::RoiAlign Node on Glow."""

        def test_f(features, rois):
            return torch.ops._caffe2.RoIAlign(
                features,
                rois,
                order="NHWC",
                spatial_scale=1.0,
                pooled_h=6,
                pooled_w=6,
                sampling_ratio=2,
                aligned=True,
            )

        N, C, H, W = 1, 3, 16, 20

        features = torch.randn(N, H, W, C)
        rois = rand_rois(N, H, W, 250)

        jitVsGlow(test_f, features, rois, expected_fused_ops={"_caffe2::RoIAlign"})

    def test_roi_align_batched(self):
        """Test of the _caffe2::RoiAlign Node on Glow."""

        def test_f(features, rois):
            return torch.ops._caffe2.RoIAlign(
                features,
                rois,
                order="NCHW",
                spatial_scale=1.0,
                pooled_h=6,
                pooled_w=6,
                sampling_ratio=2,
                aligned=True,
            )

        N, C, H, W = 4, 3, 16, 20

        features = torch.randn(N, C, H, W)
        rois = rand_rois(N, H, W, 250)

        jitVsGlow(test_f, features, rois, expected_fused_ops={"_caffe2::RoIAlign"})

    def test_roi_align_scaled(self):
        """Test of the _caffe2::RoiAlign Node on Glow."""

        def test_f(features, rois):
            return torch.ops._caffe2.RoIAlign(
                features,
                rois,
                order="NCHW",
                spatial_scale=0.0625,
                pooled_h=6,
                pooled_w=6,
                sampling_ratio=2,
                aligned=True,
            )

        N, C, H, W = 1, 3, 224, 224

        features = torch.randn(N, C, H, W)
        rois = rand_rois(N, H, W, 10)

        jitVsGlow(test_f, features, rois, expected_fused_ops={"_caffe2::RoIAlign"})

    def test_roi_align_unaligned(self):
        """Test of the _caffe2::RoiAlign Node on Glow."""

        def test_f(features, rois):
            return torch.ops._caffe2.RoIAlign(
                features,
                rois,
                order="NCHW",
                spatial_scale=1.0,
                pooled_h=6,
                pooled_w=6,
                sampling_ratio=2,
                aligned=False,
            )

        N, C, H, W = 1, 3, 16, 20

        features = torch.randn(N, C, H, W)
        rois = rand_rois(N, H, W, 250)

        jitVsGlow(test_f, features, rois, expected_fused_ops={"_caffe2::RoIAlign"})

    def test_roi_align_dynamic_sampling(self):
        """Test of the _caffe2::RoiAlign Node on Glow."""

        def test_f(features, rois):
            return torch.ops._caffe2.RoIAlign(
                features,
                rois,
                order="NCHW",
                spatial_scale=1.0,
                pooled_h=6,
                pooled_w=6,
                sampling_ratio=0,
                aligned=True,
            )

        N, C, H, W = 1, 3, 16, 20

        features = torch.randn(N, C, H, W)
        rois = rand_rois(N, H, W, 250)

        jitVsGlow(test_f, features, rois, expected_fused_ops={"_caffe2::RoIAlign"})

    def test_roi_align_fp16(self):
        """Test of the _caffe2::RoiAlign Node on Glow."""

        def test_f(features, rois):
            return torch.ops._caffe2.RoIAlign(
                features,
                rois,
                order="NCHW",
                spatial_scale=1.0,
                pooled_h=6,
                pooled_w=6,
                sampling_ratio=2,
                aligned=True,
            )

        N, C, H, W = 1, 3, 16, 20

        features = torch.randn(N, C, H, W)
        rois = rand_rois(N, H, W, 250)

        # atol/rtol must be high because maximum delta can be high due to shifts
        # in sampling points due to fp16 rounding of coordinates.
        jitVsGlow(
            test_f,
            features,
            rois,
            expected_fused_ops={"_caffe2::RoIAlign"},
            use_fp16=True,
            atol=1e-1,
            rtol=1e-1,
        )
