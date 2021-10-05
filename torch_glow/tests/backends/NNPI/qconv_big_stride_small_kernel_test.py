from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class TestQuantizedConv2dBigStrideSmallKernel(utils.TorchGlowTestCase):
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

    @utils.deterministic_expand(
        [
            lambda: (
                "2d_stride_bigger_in_one_dim",
                torch.nn.Conv2d(8, 4, [1, 1], groups=1, stride=[2, 1]),
                torch.randn([1, 8, 8, 8]),
            ),
            lambda: (
                "2d_stride_bigger_in_multi_dims",
                torch.nn.Conv2d(8, 4, [1, 1], groups=1, stride=[2, 2]),
                torch.randn([1, 8, 8, 8]),
            ),
            lambda: (
                "2d_stride_bigger_in_multi_groups",
                torch.nn.Conv2d(8, 4, [1, 1], groups=4, stride=[2, 1]),
                torch.randn([1, 8, 8, 8]),
            ),
            lambda: (
                "2d_stride_bigger_strong_test_1",
                torch.nn.Conv2d(4, 8, [2, 3], groups=2, stride=[1, 4]),
                torch.randn([1, 4, 29, 23]),
            ),
            lambda: (
                "2d_stride_bigger_strong_test_2",
                torch.nn.Conv2d(6, 8, [7, 3], groups=2, stride=[8, 4]),
                torch.randn([2, 6, 47, 35]),
            ),
        ]
    )
    def test_qconv_2d(self, name, conv, tensor):
        """Test of quantized conv2d whose stride is bigger than kernel."""
        with torch.no_grad():
            model = torch.ao.quantization.QuantWrapper(conv)
            model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
            torch.ao.quantization.prepare(model, inplace=True)
            # Calibration
            model.forward(tensor)
            torch.ao.quantization.convert(model, inplace=True)
            utils.compare_tracing_methods(
                model,
                tensor,
                fusible_ops=self.fused_2d_expect,
                # We set the atol & rtol of this test to be very big,
                # because we know there is going to be issues of off-by-1,
                # and we dont want to trigger it.
                # However, even with such great atol & rtol, this is still
                # good enough to verify the functionality is enabled correctly.
                atol=0.1,
                rtol=0.1,
            )

    # Skiped 3d tests
    @utils.deterministic_expand(
        [
            lambda: (
                "3d_stride_bigger_in_one_dim",
                torch.nn.Conv3d(8, 4, kernel_size=2, groups=1, stride=1),
                torch.randn([1, 8, 16, 8, 8]),
            ),
        ]
    )
    @unittest.skip(reason="qconv3d channelwise is not yet supported on NNPI")
    def test_qconv_3d(self, name, conv, tensor):
        """Test of quantized conv3d whose stride is bigger than kernel."""
        with torch.no_grad():
            model = torch.ao.quantization.QuantWrapper(conv)
            model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
            torch.ao.quantization.prepare(model, inplace=True)
            # Calibration
            model.forward(tensor)
            torch.ao.quantization.convert(model, inplace=True)
            utils.compare_tracing_methods(
                model,
                tensor,
                fusible_ops=self.fused_3d_expect,
                atol=0.1,
                rtol=0.1,
            )
