from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


def rand_rotated_rois(N, H, W, num_rois, horizontal=False):
    W -= 1
    H -= 1

    rois = torch.rand((num_rois, 6))

    for i in range(num_rois):
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


class TestRoiAlignRotated(unittest.TestCase):
    def test_roi_align_rotated_basic(self):
        """Test of the _caffe2::RoIAlignRotated Node on Glow."""

        def test_f(features, rois):
            return torch.ops._caffe2.RoIAlignRotated(
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
        rois = rand_rotated_rois(N, H, W, 250)

        jitVsGlow(
            test_f, features, rois, expected_fused_ops={"_caffe2::RoIAlignRotated"}
        )

    def test_roi_align_rotated_nhwc(self):
        """Test of the _caffe2::RoIAlignRotated Node on Glow."""

        def test_f(features, rois):
            return torch.ops._caffe2.RoIAlignRotated(
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
        rois = rand_rotated_rois(N, H, W, 250)

        jitVsGlow(
            test_f, features, rois, expected_fused_ops={"_caffe2::RoIAlignRotated"}
        )

    def test_roi_align_rotated_horizontal(self):
        """Test of the _caffe2::RoIAlignRotated Node on Glow."""

        def test_f(features, rois):
            return torch.ops._caffe2.RoIAlignRotated(
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
        rois = rand_rotated_rois(N, H, W, 250, horizontal=True)

        jitVsGlow(
            test_f, features, rois, expected_fused_ops={"_caffe2::RoIAlignRotated"}
        )

    def test_roi_align_rotated_batched(self):
        """Test of the _caffe2::RoIAlignRotated Node on Glow."""

        def test_f(features, rois):
            return torch.ops._caffe2.RoIAlignRotated(
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
        rois = rand_rotated_rois(N, H, W, 250)

        jitVsGlow(
            test_f, features, rois, expected_fused_ops={"_caffe2::RoIAlignRotated"}
        )

    def test_roi_align_rotated_scaled(self):
        """Test of the _caffe2::RoIAlignRotated Node on Glow."""

        def test_f(features, rois):
            return torch.ops._caffe2.RoIAlignRotated(
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
        rois = rand_rotated_rois(N, H, W, 10)

        jitVsGlow(
            test_f, features, rois, expected_fused_ops={"_caffe2::RoIAlignRotated"}
        )

    def test_roi_align_rotated_unaligned(self):
        """Test of the _caffe2::RoIAlignRotated Node on Glow."""

        def test_f(features, rois):
            return torch.ops._caffe2.RoIAlignRotated(
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
        rois = rand_rotated_rois(N, H, W, 250)

        jitVsGlow(
            test_f, features, rois, expected_fused_ops={"_caffe2::RoIAlignRotated"}
        )

    def test_roi_align_rotated_dynamic_sampling(self):
        """Test of the _caffe2::RoIAlignRotated Node on Glow."""

        def test_f(features, rois):
            return torch.ops._caffe2.RoIAlignRotated(
                features,
                rois,
                order="NCHW",
                spatial_scale=1.0,
                pooled_h=6,
                pooled_w=6,
                sampling_ratio=0,
                aligned=True,
            )

        N, C, H, W = 1, 3, 224, 224

        features = torch.randn(N, C, H, W)
        rois = rand_rotated_rois(N, H, W, 10)

        jitVsGlow(
            test_f, features, rois, expected_fused_ops={"_caffe2::RoIAlignRotated"}
        )

    def test_roi_align_rotated_fp16(self):
        """Test of the _caffe2::RoIAlignRotated Node on Glow."""

        def test_f(features, rois):
            return torch.ops._caffe2.RoIAlignRotated(
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
        rois = rand_rotated_rois(N, H, W, 250)

        # atol/rtol must be high because maximum delta can be high due to shifts
        # in sampling points due to fp16 rounding of coordinates.
        jitVsGlow(
            test_f,
            features,
            rois,
            expected_fused_ops={"_caffe2::RoIAlignRotated"},
            use_fp16=True,
            atol=0.5,
            rtol=0.5,
        )
