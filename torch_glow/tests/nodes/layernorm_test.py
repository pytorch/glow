from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F
from tests import utils


class SimpleLayerNormModule(torch.nn.Module):
    def __init__(self, normalized_shape):
        super(SimpleLayerNormModule, self).__init__()
        self.normalized_shape = normalized_shape

    def forward(self, input, weight=None, bias=None):
        return F.layer_norm(input, self.normalized_shape, weight, bias)


class LayerNormNHCWLayout(torch.nn.Module):
    def __init__(self, normalized_shape, stride=1, padding=0, dilation=1, groups=1):
        super(LayerNormNHCWLayout, self).__init__()
        self.normalized_shape = normalized_shape
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, inputs, filters, biasConv=None, weight=None, bias=None):
        conv = F.conv2d(
            inputs,
            filters,
            bias=biasConv,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return F.layer_norm(
            conv.permute(0, 2, 1, 3), self.normalized_shape, weight, bias
        )


class LayerNormNHCWLayoutWithConvAfter(torch.nn.Module):
    def __init__(self, normalized_shape, stride=1, padding=0, dilation=1, groups=1):
        super(LayerNormNHCWLayoutWithConvAfter, self).__init__()
        self.normalized_shape = normalized_shape
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(
        self,
        inputs,
        filters,
        filters2,
        biasConv=None,
        biasConv2=None,
        weight=None,
        bias=None,
    ):
        conv = F.conv2d(
            inputs,
            filters,
            bias=biasConv,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        t = F.layer_norm(conv.permute(0, 2, 1, 3), self.normalized_shape, weight, bias)
        return F.conv2d(
            t.permute(0, 2, 1, 3),
            filters2,
            biasConv2,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class TestLayerNorm(utils.TorchGlowTestCase):
    def test_layernorm_basic(self):
        """Basic test of the PyTorch layernorm Node on Glow."""

        inputs = torch.randn(1, 4, 5, 5)
        weight = torch.randn(5)
        bias = torch.randn(5)

        utils.compare_tracing_methods(
            SimpleLayerNormModule([5]),
            inputs,
            weight,
            bias,
            fusible_ops={"aten::layer_norm"},
        )

    def test_layernorm_no_bias(self):
        """Test of the PyTorch aten::layer_norm without weights and bias."""

        inputs = torch.randn(1, 4, 5, 5)

        utils.compare_tracing_methods(
            SimpleLayerNormModule([5, 5]), inputs, fusible_ops={"aten::layer_norm"}
        )

    def test_layernorm_layout(self):
        """Test of the PyTorch aten::layer_norm with NHCW layout."""

        inputs = torch.randn(1, 6, 5, 6)
        kernel = torch.randn(3, 6, 2, 2)
        # This unit test build a graph like conv => permute => layer_norm
        # Since in Glow we always guess 4 dims input tensor to be NCHW,
        # After the permutation, the layout of layer_norm's input would be
        # NHCW, which is not a supported layout, and we should mitigate this by
        # setting accept_all_layouts to be true.
        utils.compare_tracing_methods(
            LayerNormNHCWLayout([5]),
            inputs,
            kernel,
            fusible_ops={"aten::layer_norm", "aten::permute", "aten::_convolution"},
            accept_all_layouts=True,
        )

    def test_layernorm_layout_with_conv_after(self):
        """Test of the PyTorch aten::layer_norm with NHCW layout and conv after
        layer_norm."""

        inputs = torch.randn(1, 8, 5, 6)
        kernel = torch.randn(4, 8, 2, 2)
        kernel2 = torch.randn(2, 4, 2, 2)
        # This unit test build a graph like conv => permute => layer_norm
        # => conv. Since in Glow we always guess 4 dims input tensor to be NCHW,
        # After the permutation, the layout of layer_norm's input would be
        # NHCW. If we simply ignore the layout checking of layer_norm, still
        # the second conv will complain about layout mismatch. We should
        # mitigate this by setting accept_all_layouts to be true.
        utils.compare_tracing_methods(
            LayerNormNHCWLayoutWithConvAfter([5]),
            inputs,
            kernel,
            kernel2,
            fusible_ops={"aten::layer_norm", "aten::permute", "aten::_convolution"},
            accept_all_layouts=True,
        )
