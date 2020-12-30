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


# Custom Relu implementation in pytorch along with onnx export definition
# Refer: https://github.com/onnx/tutorials/tree/master/PyTorchCustomOperator
# Refer: https://pytorch.org/docs/stable/onnx.html#custom-operators

import torch
from torch.onnx.symbolic_helper import parse_args
import torch.utils.cpp_extension

op_source = """
#include <torch/script.h>

torch::Tensor custom_relu(torch::Tensor input, double alpha, double beta) {
  input *= alpha;
  input += beta;
  input = input.clamp_max(1).clamp_min(0);
  return input.clone();
}

TORCH_LIBRARY(op_domain, m) {
  m.def("custom_relu", &custom_relu);
}
"""

# Compile and load the custom op
torch.utils.cpp_extension.load_inline(
    name="custom_relu",
    cpp_sources=op_source,
    is_python_module=False,
    verbose=True,
)

# Wrapper module for custom relu C++ op
class CustomRelu(torch.nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, x):
        return torch.ops.op_domain.custom_relu(x, self.alpha, self.beta)

# ONNX export symbolic helper
@parse_args('v', 'f', 'f')
def custom_relu_onnx(g, x, alpha, beta):
    return g.op("OpDomain::CustomRelu", x, alpha_f=alpha, beta_f=beta)

torch.onnx.register_custom_op_symbolic('op_domain::custom_relu', custom_relu_onnx, 9)

# Create a sequenction FC+Relu model
model = torch.nn.Sequential(
    torch.nn.Linear(3, 3),
    CustomRelu(0.5, 0.2))
model.eval()

# Random input for tracing
inp = torch.rand(3, 3)

torch.onnx.export(model, (inp, ), "mlp_custom.onnx", input_names=["x"])

out = model(inp)
out.detach().numpy().tofile("mlp_output.raw")
