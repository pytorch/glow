# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear1 = torch.nn.Linear(5, 3)
        self.linear2 = torch.nn.Linear(3, 1)

    def forward(self, x):
        return self.linear2(self.linear1(x))


class ConvModel(torch.nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1)
        self.conv2 = torch.nn.Conv2d(3, 3, 1)
        self.linear = torch.nn.Linear(5, 5)

    def forward(self, x):
        return self.linear(self.conv2(self.conv1(x)))


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = ConvModel()
        self.linear = LinearModel()

    def forward(self, x):
        return self.linear(self.conv(x))


def test_fuse_necessary_getattrs_only():
    m = Model()
    x = torch.randn(1, 3, 5, 5)

    torch_glow.disableFusionPass()

    jit_m = torch.jit.trace(m, x)
    jit_m_graph = jit_m.graph_for(x)

    # don't fuse aten::_convolutions
    torch_glow.glowCustomFuseDebug_(
        jit_m_graph,
        ["prim::Constant", "prim::GetAttr", "aten::t", "aten::matmul", "aten::add_"],
    )

    return m(x)
