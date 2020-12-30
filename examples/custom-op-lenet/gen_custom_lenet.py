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

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch.onnx.symbolic_helper import parse_args
import torch.utils.cpp_extension

op_source = """
#include <torch/script.h>
#include <torch/nn/functional/pooling.h>

namespace F = torch::nn::functional;

torch::Tensor scaled_tanh(torch::Tensor input, double scale, double amplitude) {
  auto scaled = input * scale;
  auto t = torch::tanh(scaled);
  auto ampt = amplitude * t;
  return ampt;
}

torch::Tensor l2pool(torch::Tensor input, std::vector<int64_t> kernel_size) {
  return F::lp_pool2d(input, F::LPPool2dFuncOptions(2, kernel_size));
}

TORCH_LIBRARY(lenetops, m) {
  m.def("scaled_tanh", &scaled_tanh);
  m.def("l2pool", &l2pool);
}
"""

# Compile and load the custom op
torch.utils.cpp_extension.load_inline(
    name="lenetops_pytorch",
    cpp_sources=op_source,
    is_python_module=False,
    verbose=True,
)

class ScaledTanh(torch.nn.Module):
    def __init__(self, scale, amplitude):
        super().__init__()
        self.scale = float(scale)
        self.amplitude = float(amplitude)

    def forward(self, x):
        return torch.ops.lenetops.scaled_tanh(x, self.scale, self.amplitude)

# ONNX export symbolic helper
@parse_args('v', 'f', 'f')
def scaled_tanh_helper(g, x, scale, amplitude):
    return g.op("LenetOps::ScaledTanh", x, scale_f=scale, amplitude_f=amplitude)

torch.onnx.register_custom_op_symbolic('lenetops::scaled_tanh', scaled_tanh_helper, 9)

class L2Pool(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = list(kernel_size)

    def forward(self, x):
        return torch.ops.lenetops.l2pool(x, self.kernel_size)

# ONNX export symbolic helper
@parse_args('v', 'is')
def l2pool_helper(g, x, kernel_size):
    return g.op("LenetOps::L2Pool", x, kernel_size_i=kernel_size)

torch.onnx.register_custom_op_symbolic('lenetops::l2pool', l2pool_helper, 9)

class LeNetCustom(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
                # Input = 1x1x28x28
                nn.Conv2d(1, 6, (5, 5), padding=2),
                # c1 = 1x6x28x28
                ScaledTanh(2.0, 0.5),
                L2Pool((2, 2)),
                # p1 = 1x6x14x14

                nn.Conv2d(6, 16, (5, 5)),
                # c2 = 1x16x10x10
                ScaledTanh(2.0, 0.5),
                L2Pool((2, 2)))
                # p2 = 1x16x5x5

        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16*5*5, 120),
                ScaledTanh(0.5, 1.0),
                nn.Linear(120, 84),
                ScaledTanh(0.5, 1.0),
                nn.Linear(84, 10))

    def forward(self, x):
        x = self.feature_extractor(x)
        y = self.classifier(x)

        return y

# download and create datasets
train_dataset = datasets.MNIST(root='mnist_data', train=True, transform=transforms.ToTensor())
valid_dataset = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transforms.ToTensor())

# define the data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=32, shuffle=False)

model = LeNetCustom()

torch.manual_seed(42)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Epochs loop
for e in range(5):
    # Training loop
    model.train()
    running_loss = 0
    for X, y_true in train_loader:
        optimizer.zero_grad()

        # Forward pass
        y = model(X)
        y_hat = F.softmax(y, dim=1)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

        # Back propagation
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    print("Epoch %d loss=%f " % (e, epoch_loss), end='')

    with torch.no_grad():
        # Validation loop
        model.eval()
        running_loss = 0
        for X, y_true in valid_loader:
            # Forward pass
            y = model(X)
            y_hat = F.softmax(y, dim=1)
            loss = criterion(y_hat, y_true)
            running_loss += loss.item() * X.size(0)

        epoch_loss = running_loss / len(valid_loader.dataset)
        print("Validation loss=%f" % epoch_loss)

with torch.no_grad():
    model.eval()
    inp = valid_loader.dataset[0][0]
    batch = torch.unsqueeze(inp, 0)
    torch.onnx.export(model, batch, "lenet_mnist_custom.onnx", verbose=True)
