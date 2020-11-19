from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleTypeasModel(torch.nn.Module):
    def __init__(self):
        super(SimpleTypeasModel, self).__init__()

    def forward(self, tensor, other=None):
        # TODO: Understand and document the utility of the self-conversion test
        # as well as the additional tensor + tensor step
        other = tensor if other is None else other
        if tensor.dtype != torch.bool:
            tensor = tensor + tensor
        typed = tensor.type_as(other)
        return typed + typed


class TestTypeAs(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "to_int32",
                SimpleTypeasModel(),
                torch.randn(4),
                torch.zeros(4, dtype=torch.int32),
            ),
            (
                "from_int32",
                SimpleTypeasModel(),
                torch.randn(4).to(dtype=torch.int32),
                torch.zeros(4),
            ),
            (
                "from_bool",
                SimpleTypeasModel(),
                torch.randn(4).to(dtype=torch.bool),
                torch.zeros(4),
            ),
            ("self", SimpleTypeasModel(), torch.randn(4), None, False),
            (
                "f2f2",
                SimpleTypeasModel(),
                torch.randn(4, 2),
                torch.randn(8, 3, 4, 2),
                False,
            ),
            (
                "f2i2",
                SimpleTypeasModel(),
                torch.randn(4, 2),
                torch.randn(8, 3, 4, 2).to(dtype=torch.int32),
            ),
        ]
    )
    def test_typeas(self, _, module, tensor, other=None, should_fuse=True):
        if other is not None:
            utils.compare_tracing_methods(
                module,
                tensor,
                other,
                fusible_ops={"aten::type_as"} if should_fuse else {},
            )
        else:
            utils.compare_tracing_methods(
                module, tensor, fusible_ops={"aten::type_as"} if should_fuse else {}
            )
