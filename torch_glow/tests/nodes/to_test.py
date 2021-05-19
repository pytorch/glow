from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleToModel(torch.nn.Module):
    def __init__(self, *conversions):
        super(SimpleToModel, self).__init__()
        self.conversions = conversions

    def forward(self, tensor):
        for conversion_type in self.conversions:
            tensor = tensor.to(conversion_type)
        return tensor


class ToWithDeviceModel(torch.nn.Module):
    def __init__(self, *conversions):
        super(ToWithDeviceModel, self).__init__()
        self.conversions = conversions

    def forward(self, tensor):
        for conversion_type in self.conversions:
            tensor = tensor.to(device="cpu", dtype=conversion_type)
        return tensor


class SimplePrimToModel(torch.nn.Module):
    def __init__(self, conversion, device=None):
        super().__init__()
        self.device = None
        self.conversion = conversion
        if self.device is None:
            self.forward = self._forward_dtype
        else:
            self.forward = self._forward_device_dtype

    def _forward_device_dtype(self, dummy):
        return torch.ops.prim.NumToTensor(dummy.size(0)).to(
            device=self.device, dtype=self.conversion
        )

    def _forward_dtype(self, dummy):
        return torch.ops.prim.NumToTensor(dummy.size(0)).to(self.conversion)


class TestTo(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("to_int", SimpleToModel(torch.int), torch.randn(1, 2, 3, 4)),
            lambda: ("to_float", SimpleToModel(torch.float), torch.randn(1, 2, 3, 4)),
            lambda: (
                "to_int_to_float",
                SimpleToModel(torch.int, torch.float),
                torch.randn(1, 2, 3, 4),
            ),
            lambda: (
                "to_int_with_device",
                ToWithDeviceModel(torch.int),
                torch.randn(1, 2, 3, 4),
            ),
            lambda: ("to_cpu", SimpleToModel("cpu"), torch.randn(1, 2, 3, 4)),
            lambda: (
                "to_tensor",
                SimpleToModel(torch.randn(3, 4).type(torch.int32)),
                torch.randn(1, 2, 3, 4),
            ),
        ]
    )
    def test_to(self, _, module, tensor):
        utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::to"})

    @utils.deterministic_expand(
        [
            lambda: (
                "to_prim_dtype",
                SimplePrimToModel(torch.float),
                torch.randn(5, 6, 7),
            ),
            lambda: ("to_prim_device", SimplePrimToModel("cpu"), torch.randn(5, 6, 7)),
            lambda: (
                "to_prim_device_with_dtype",
                SimplePrimToModel(torch.float, "cuda"),
                torch.randn(5, 6, 7),
            ),
        ]
    )
    def test_to_prim(self, _, module, tensor):
        utils.compare_tracing_methods(
            module,
            tensor,
            fusible_ops={"prim::NumToTensor", "aten::to"},
            scripted=True,
        )
