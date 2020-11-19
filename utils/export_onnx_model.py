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

import argparse

import torch.onnx
import torchvision
from torch.autograd import Variable


# Export ONNX model from PyTorch
# Refer to https://pytorch.org/docs/stable/onnx.html


class PyTorchPretrainedModel(object):
    def __init__(self, model_name):
        self.model_name = model_name
        method_to_call = getattr(torchvision.models, self.model_name)
        self.model = method_to_call(pretrained=True)
        self.model_parameters_num = len(list(self.model.state_dict()))

    def export_onnx_model(
        self, input_name, output_name, batch_size, model_path, verbose
    ):
        dummy_input = Variable(torch.randn(batch_size, 3, 224, 224))
        input_names = [input_name] + [
            "learned_%d" % i for i in range(self.model_parameters_num)
        ]
        output_names = [output_name]
        torch.onnx.export(
            self.model,
            dummy_input,
            model_path,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
        )


if __name__ == "__main__":
    # For more pretrained model in PyTorch, refer to:
    # https://pytorch.org/docs/stable/torchvision/models.html
    parser = argparse.ArgumentParser("ONNX model exported from PyTorch.")
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--model_path", type=str, default="resnet18.onnx")
    parser.add_argument("--model_input_name", type=str, default="data")
    parser.add_argument("--model_output_name", type=str, default="output")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    pytorch_model = PyTorchPretrainedModel(args.model_name)
    pytorch_model.export_onnx_model(
        args.model_input_name,
        args.model_output_name,
        args.batch_size,
        args.model_path,
        args.verbose,
    )
