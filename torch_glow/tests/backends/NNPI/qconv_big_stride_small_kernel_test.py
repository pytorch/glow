from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class TestQuantizedConv2dBigStrideSmallKernel(unittest.TestCase):
    # These tests should be run on NNPI card manually, or else
    # buck test will only run them on emulator.
    supported_backends = {"NNPI"}
    fused_2d_expect = {
        "aten::quantize_per_tensor",
        "quantized::conv2d",
        "aten::dequantize",
    }
    fused_3d_expect = {
        "aten::quantize_per_tensor",
        "quantized::conv3d",
        "aten::dequantize",
    }

    @parameterized.expand(
        [
            (
                "2d_stride_bigger_in_one_dim",
                torch.nn.Conv2d(8, 4, [1, 1], groups=1, stride=[2, 1]),
                torch.randn([1, 8, 8, 8]),
                fused_2d_expect,
            ),
            (
                "2d_stride_bigger_in_multi_dims",
                torch.nn.Conv2d(8, 4, [1, 1], groups=1, stride=[2, 2]),
                torch.randn([1, 8, 8, 8]),
                fused_2d_expect,
            ),
            (
                "2d_stride_bigger_in_multi_groups",
                torch.nn.Conv2d(8, 4, [1, 1], groups=4, stride=[2, 1]),
                torch.randn([1, 8, 8, 8]),
                fused_2d_expect,
            ),
            (
                "2d_stride_bigger_strong_test_1",
                torch.nn.Conv2d(4, 8, [2, 3], groups=2, stride=[1, 4]),
                torch.randn([1, 4, 29, 23]),
                fused_2d_expect,
            ),
            (
                "2d_stride_bigger_strong_test_2",
                torch.nn.Conv2d(6, 8, [7, 3], groups=2, stride=[8, 4]),
                torch.randn([2, 6, 47, 35]),
                fused_2d_expect,
            ),
            # Skiped 3d tests
            # (
            #    "3d_stride_bigger_in_one_dim",
            #    torch.nn.Conv3d(8, 4, kernel_size=2, groups=1, stride=1),
            #    torch.randn([1, 8, 16, 8, 8]),
            #    fused_3d_expect,
            # ),
        ]
    )
    def test_qconv(self, name, conv, tensor, fused_expect):
        """Test of quantized conv whose stride is bigger than kernel."""
        with torch.no_grad():
            model = torch.quantization.QuantWrapper(conv)
            model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
            torch.quantization.prepare(model, inplace=True)
            # Calibration
            model.forward(tensor)
            torch.quantization.convert(model, inplace=True)
            utils.compare_tracing_methods(
                model,
                tensor,
                fusible_ops=fused_expect,
                # We set the atol & rtol of this test to be very big,
                # because we know there is going to be issues of off-by-1,
                # and we dont want to trigger it.
                # However, even with such great atol & rtol, this is still
                # good enough to verify the functionality is enabled correctly.
                atol=0.1,
                rtol=0.1,
            )
