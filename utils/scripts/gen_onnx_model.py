# Copyright (c) 2017-present, Facebook, Inc.
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

## This is a helper script that generates ONNX models.
## The generated model will be used for ONNX importer unittest:
## ./tests/unittests/onnxImporterTest.cpp
## Run $>python gen_onnx_model.py to get the ONNX model.

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np

# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

W = np.array([[[[1., 1.],  # (1, 1, 2, 2) tensor for convolution weights
                [1., 1.]]]]).astype(np.float32)

B = np.array([2.]).astype(np.float32)


# Convolution with padding. "data" represents the input data,
# which will be provided by ONNX importer unittests.
node_def = onnx.helper.make_node(
    'Conv',
    inputs=['data', 'W', 'B'],
    outputs=['y'],
    kernel_shape=[2, 2],
    strides=[1, 1],
    pads=[1, 1, 1, 1],
    name="conv1"
)

weight_tensor = helper.make_tensor(
            name='W',
            data_type=TensorProto.FLOAT,
            dims=(1, 1, 2, 2),
            vals=W.reshape(4).tolist()
            )

bias_tensor = helper.make_tensor(
            name='B',
            data_type=TensorProto.FLOAT,
            dims=(1,),
            vals=B.reshape(1).tolist()
            )
# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    "test-model",
    inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, [1, 1, 3, 3]),
            helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 1, 2, 2]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [1,])],
    outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4, 4])]
)

graph_def.initializer.extend([weight_tensor])
graph_def.initializer.extend([bias_tensor])
graph_def.initializer[0].name = 'W'
graph_def.initializer[1].name = 'B'
# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-conv')

with open('simpleConv.onnxtxt', 'w') as f:
  f.write(str(model_def));

onnx.checker.check_model(model_def)
print('The model is checked!')
