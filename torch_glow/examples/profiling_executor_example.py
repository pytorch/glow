# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
import torch_glow


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


torch._C._jit_set_profiling_mode(True)
torch_glow.enableFusionPass()

m = Model()

m_jit = torch.jit.script(m)

x = torch.randn(10)

# No Glow fusion node
print("initial jit ir")
print(m_jit.graph_for(x))

m_jit(x)
m_jit(x)
m_jit(x)

# Contains Glow fusion node
print("final jit ir")
print(m_jit.graph_for(x))
